from __future__ import annotations

import functools
import hashlib
import inspect
import json
from typing import Any, Callable, ParamSpec, Type, TypeVar, cast, overload

from joblib import Memory
from pydantic import BaseModel

P = ParamSpec("P")
M = TypeVar("M")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")
TNode = TypeVar("TNode", bound= BaseModel)
BaseM = TypeVar("BaseM", bound = BaseModel)
def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))

@overload
def cache_pydantic_structured(
    *,
    memory: Memory,
    model: Type[BaseM],
    fn: Callable[P, BaseM],
    ignore: list[str] | None = None,
    dump_exclude: set[str] | None = None,
) -> Callable[P, BaseM]:
    ...
@overload
def cache_pydantic_structured(
    *,
    memory: Memory,
    model: None,
    fn: Callable[P, Any],
    ignore: list[str] | None = None,
    dump_exclude: set[str] | None = None,
) -> Callable[P, Any]:
    ...
def cache_pydantic_structured(
    *,
    memory: Memory,
    model: Type[BaseM] | None,
    fn: Callable[P, BaseM],
    ignore: list[str] | None = None,
    dump_exclude: set[str] | None = None,
) -> Callable[P, BaseM | None]:
    """
    Cache a function that returns a Pydantic model using joblib, while:

    - hashing arguments + the model's JSON schema (schema changes invalidate cache)
    - caching ONLY model_dump() (not the Pydantic object itself)
    - reconstructing the model via model_validate() on cache hit
    - preserving the original function signature
    - supporting joblib ignore=... (e.g. to ignore an agent/self argument)

    This is designed for LLM structured-output calls, especially when
    `include_raw=True` is used internally.

    --------------------------------------------------------------------
    Generic example
    --------------------------------------------------------------------

    >>> from joblib import Memory
    >>> from pydantic import BaseModel
    >>>
    >>> class Result(BaseModel):
    ...     answer: str
    ...     # raw: dict | None = None   # if present, exclude from cache
    >>>
    >>> class Agent:
    ...     def llm_call(self, prompt: str) -> Result:
    ...         # internally:
    ...         # self.llm.with_structured_output(Result, include_raw=True).invoke(...)
    ...         return Result(answer="hello")
    ...
    ...     @staticmethod
    ...     def entry(agent: "Agent", *, prompt: str) -> Result:
    ...         return agent.llm_call(prompt)
    >>>
    >>> mem = Memory(location="/tmp/cache", verbose=0)
    >>> cached_entry = cache_pydantic_structured(
    ...     memory=mem,
    ...     model=Result,
    ...     fn=Agent.entry,
    ...     ignore=["agent"],          # do not hash the agent instance
    ...     dump_exclude={"raw"},      # do not cache raw LLM payloads
    ... )
    >>>
    >>> agent = Agent()
    >>> out = cached_entry(agent, prompt="hi")
    >>> isinstance(out, Result)
    True

    --------------------------------------------------------------------
    Notes
    --------------------------------------------------------------------
    - The wrapped function must return an instance of `model`.
    - Only the dumped dict is cached; raw LLM responses should be excluded.
    - The cache key includes:
        * function arguments (except ignored ones)
        * the model JSON schema hash
    - The returned callable has the SAME signature as `fn`.

    """
    ignore = ignore or []
    dump_exclude = dump_exclude or set()
    cached_entry = cached(memory, fn, ignore=ignore)

    @functools.wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> BaseM:
        # We assume first positional arg is the agent/self (so ignore=["agent"] works)
        # if not args:
        #     raise TypeError("Expected first positional argument to be the agent/self")
        payload = cached_entry(*args, **kwargs)
        return model.model_validate(payload) if model is not None else payload

    wrapped.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return cast(Callable[P, BaseM], wrapped)


if __name__ == "__main__":
    from pathlib import Path
    from joblib import Memory
    from pydantic import BaseModel
    import shutil
    import tempfile

    # ----------------------------
    # Fake Pydantic output model
    # ----------------------------
    class FakeSelection(BaseModel):
        used_node_ids: list[str]
        score: float
        # raw: dict | None = None   # uncomment to test dump_exclude

    # ----------------------------
    # Fake agent (no LLM involved)
    # ----------------------------
    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def _select_used_evidence(self, question: str, candidates: list[dict], out_model: Type[BaseM]) -> BaseM:
            # simulate "LLM" work
            self.calls += 1
            outdict = dict(used_node_ids=[c["node_id"] for c in candidates],
                score=len(question),)
            return out_model.model_validate(outdict)

        @staticmethod
        def entry(agent: "FakeAgent", question: str, candidates: list[dict], out_schema, out_model: Type[BaseM]) -> BaseM:
            return agent._select_used_evidence(question=question, candidates=candidates, out_model=out_model)

    # ----------------------------
    # Temp cache dir
    # ----------------------------
    tmpdir = Path(tempfile.mkdtemp(prefix="joblib-test-"))
    try:
        mem = Memory(location=str(tmpdir), verbose=0)

        # Wrap with your caching helper
        cached_entry = cache_pydantic_structured(
            memory=mem,
            model=FakeSelection,
            fn=FakeAgent.entry,
            ignore=["agent", "out_model"],
            # dump_exclude={"raw"},  # if you add raw above
        )

        agent = FakeAgent()
        candidates = [
            {"node_id": "n1", "text": "hello"},
            {"node_id": "n2", "text": "world"},
        ]

        # ----------------------------
        # First call → cache MISS
        # ----------------------------
        out1 = cached_entry(agent=agent, question="hi", candidates=candidates, out_schema = FakeSelection.model_json_schema, out_model=FakeSelection)
        print("out1:", out1)
        print("agent.calls after first:", agent.calls)
        assert agent.calls == 1

        # ----------------------------
        # Second call → cache HIT
        # ----------------------------
        out2 = cached_entry(agent=agent, question="hi", candidates=candidates, out_schema = FakeSelection.model_json_schema, out_model=FakeSelection)
        print("out2:", out2)
        print("agent.calls after second:", agent.calls)
        assert agent.calls == 1  # no increment → cache hit

        # ----------------------------
        # Different input → cache MISS
        # ----------------------------
        out3 = cached_entry(agent=agent, question="different", candidates=candidates, out_schema = FakeSelection.model_json_schema, out_model=FakeSelection)
        print("out3:", out3)
        print("agent.calls after third:", agent.calls)
        assert agent.calls == 2

        # ----------------------------
        # Schema change invalidates cache
        # ----------------------------
        class FakeSelectionV2(BaseModel):
            used_node_ids: list[str]
            score: float
            extra: int

        cached_entry_v2 = cache_pydantic_structured(
            memory=mem,
            model=FakeSelectionV2,
            fn=FakeAgent.entry,
            ignore=["agent"],
        )

        out4 = cached_entry_v2(agent=agent, question="hi", candidates=candidates)
        print("out4 (v2):", out4)
        print("agent.calls after schema change:", agent.calls)
        assert agent.calls == 3  # schema hash forces miss

        print("\n✅ All cache tests passed")

    finally:
        shutil.rmtree(tmpdir)