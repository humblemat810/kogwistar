from __future__ import annotations

"""Workflow step resolvers.

This module provides a registry-based step resolver that can be used by
`WorkflowRuntime` (or any workflow executor) that expects:

    step_resolver(op_name: str) -> Callable[[StepContext], RunResult]

Design goals
------------
* Keep step implementations out of the orchestrator.
* Allow the orchestrator to inject runtime dependencies via `ctx.state["_deps"]`.
* Keep the step resolver contract stable: *handlers return RunResult*.

Dependency injection
--------------------
Handlers are expected to retrieve dependencies from `ctx.state["_deps"]`, e.g.:

    deps = ctx.state["_deps"]
    conversation_engine = deps["conversation_engine"]

The orchestrator should populate `_deps` in the workflow initial_state.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Union


Json = Any

# Import your real RunResult types from runtime/models
from graph_knowledge_engine.workflow.runtime import RunFailure, RunResult, RunSuccess, StepContext

RawStepFn = Callable[[StepContext], Union[Json, RunResult]]

@dataclass
class MappingStepResolver:
    handlers: Dict[str, RawStepFn]
    default: Optional[RawStepFn] = None

    def __init__(self, handlers: Optional[Mapping[str, RawStepFn]] = None, *, default: Optional[RawStepFn] = None) -> None:
        self.handlers = dict(handlers or {})
        self.default = default

    def register(self, op: str) -> Callable[[RawStepFn], RawStepFn]:
        def _decorator(fn: RawStepFn) -> RawStepFn:
            self.handlers[op] = fn
            return fn
        return _decorator

    def resolve(self, op: str) -> Callable[[StepContext], RunResult]:
        raw = self.handlers.get(op) or self.default
        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        def _wrapped(ctx: StepContext) -> RunResult:
            try:
                out = raw(ctx)
                if isinstance(out, (RunSuccess, RunFailure)):
                    return out
                else:
                    raise TypeError("Resolver must return RunResult")
                # return out  # field name might be `data`/`payload` in your codebase
            except Exception as e:
                return RunFailure(conversation_node_id=ctx.state.get('workflow_node_id') , errors=[str(e)])  # match your RunFailure fields
        return _wrapped

    def __call__(self, op: str) -> Callable[[StepContext], RunResult]:
        return self.resolve(op)
    
    
default_resolver = MappingStepResolver()


def _deps(ctx: StepContext) -> Dict[str, Any]:
    deps = ctx.state.get("_deps")
    if not isinstance(deps, dict):
        raise RuntimeError("StepContext.state['_deps'] must be a dict of injected dependencies")
    return deps


@default_resolver.register("start")
def _start(ctx: StepContext) -> RunResult:
    ctx.state.setdefault("op_log", []).append("start")
    ctx.state["started"] = True
    return RunSuccess(conversation_node_id=None, outputs=[{"started": True}])


@default_resolver.register("memory_retrieve")
def _memory_retrieve(ctx: StepContext) -> RunResult:
    """Retrieve candidate memories.

    Writes:
      - state['memory_raw'] : MemoryRetrievalResult (non-serializable, runtime-only)
      - state['memory']     : jsonable mirror
    """
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("memory_retrieve")

    from .memory_retriever import MemoryRetriever
    from .workflow.serialize import to_jsonable

    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
    )

    tool_runner = deps.get("tool_runner")
    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    if tool_runner is None:
        # Fallback: run directly without tool recording.
        mem = mem_retriever.retrieve(
            user_id=ctx.state["user_id"],
            current_conversation_id=ctx.state["conversation_id"],
            query_embedding=ctx.state["embedding"],
            user_text=ctx.state["user_text"],
            context_text="",
        )
    else:
        mem = tool_runner.run_tool(
            conversation_id=ctx.state["conversation_id"],
            user_id=ctx.state["user_id"],
            turn_node_id=ctx.state["turn_node_id"],
            turn_index=ctx.state["turn_index"],
            tool_name="memory_retrieve",
            args={"n_results": getattr(mem_retriever, "n_results", None)},
            handler=lambda: mem_retriever.retrieve(
                user_id=ctx.state["user_id"],
                current_conversation_id=ctx.state["conversation_id"],
                query_embedding=ctx.state["embedding"],
                user_text=ctx.state["user_text"],
                context_text="",
            ),
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    ctx.state["memory_raw"] = mem
    memj = to_jsonable(mem)
    ctx.state["memory"] = memj
    return RunSuccess(conversation_node_id=None, outputs=[memj])


@default_resolver.register("kg_retrieve")
def _kg_retrieve(ctx: StepContext) -> RunResult:
    """Retrieve KG facts/links based on query and memory seed ids."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("kg_retrieve")

    from .knowledge_retriever import KnowledgeRetriever
    from .workflow.serialize import to_jsonable

    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )

    mem_raw = ctx.state.get("memory_raw")
    seed_ids = list(getattr(mem_raw, "seed_kg_node_ids", []) or []) if mem_raw is not None else []

    tool_runner = deps.get("tool_runner")
    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    if tool_runner is None:
        kg = kg_retriever.retrieve(
            user_text=ctx.state["user_text"],
            context_text="",
            query_embedding=ctx.state["embedding"],
            seed_kg_node_ids=seed_ids,
        )
    else:
        kg = tool_runner.run_tool(
            conversation_id=ctx.state["conversation_id"],
            user_id=ctx.state["user_id"],
            turn_node_id=ctx.state["turn_node_id"],
            turn_index=ctx.state["turn_index"],
            tool_name="kg_retrieve",
            args={"max_retrieval_level": max_retrieval_level, "seed_kg_node_ids": seed_ids},
            handler=lambda: kg_retriever.retrieve(
                user_text=ctx.state["user_text"],
                context_text="",
                query_embedding=ctx.state["embedding"],
                seed_kg_node_ids=seed_ids,
            ),
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    ctx.state["kg_raw"] = kg
    kgj = to_jsonable(kg)
    ctx.state["kg"] = kgj
    return RunSuccess(conversation_node_id=None, outputs=[kgj])


@default_resolver.register("memory_pin")
def _memory_pin(ctx: StepContext) -> RunResult:
    """Pin selected memory into the conversation graph."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("memory_pin")

    from .memory_retriever import MemoryRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")

    mem_raw = ctx.state.get("memory_raw")
    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
    )

    out = None
    if (
        mem_raw is not None
        and getattr(mem_raw, "selected", None)
        and getattr(mem_raw, "memory_context_text", None)
    ):
        out = mem_retriever.pin_selected(
            user_id=ctx.state["user_id"],
            current_conversation_id=ctx.state["conversation_id"],
            turn_node_id=ctx.state["turn_node_id"],
            mem_id=ctx.state["mem_id"],
            turn_index=ctx.state["turn_index"],
            self_span=ctx.state["self_span"],
            selected_memory=getattr(mem_raw, "selected"),
            memory_context_text=getattr(mem_raw, "memory_context_text"),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    ctx.state["memory_pin_raw"] = out
    outj = {
        "memory_context_node_id": getattr(getattr(out, "memory_context_node", None), "id", None)
        if out
        else None,
        "pinned_edge_ids": [e.id for e in getattr(out, "pinned_edges", [])] if out else [],
    }
    ctx.state["memory_pin"] = outj
    return RunSuccess(conversation_node_id=None, outputs=[outj])


@default_resolver.register("kg_pin")
def _kg_pin(ctx: StepContext) -> RunResult:
    """Pin selected KG nodes/edges (as pointers) into the conversation graph."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("kg_pin")

    from .knowledge_retriever import KnowledgeRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )

    kg_raw = ctx.state.get("kg_raw")
    pinned_ptrs: list[str] = []
    pinned_edges: list[str] = []
    if kg_raw is not None and getattr(kg_raw, "selected", None):
        pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
            user_id=ctx.state["user_id"],
            conversation_id=ctx.state["conversation_id"],
            turn_node_id=ctx.state["turn_node_id"],
            turn_index=ctx.state["turn_index"],
            self_span=ctx.state["self_span"],
            selected_knowledge=getattr(kg_raw, "selected"),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    outj = {"pinned_pointer_node_ids": list(pinned_ptrs), "pinned_edge_ids": list(pinned_edges)}
    ctx.state["kg_pin"] = outj
    return RunSuccess(conversation_node_id=None, outputs=[outj])


@default_resolver.register("answer")
def _answer(ctx: StepContext) -> RunResult:
    """Run answer-only agent, then link assistant turn into conversation chain."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("answer")

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    answer_only = deps.get("answer_only")
    if not callable(answer_only):
        raise RuntimeError("deps['answer_only'] must be callable")

    resp = answer_only(conversation_id=ctx.state["conversation_id"], prev_turn_meta_summary=prev_turn_meta_summary)
    ctx.state["answer_raw"] = resp

    response_node_id = getattr(resp, "response_node_id", None)
    if response_node_id:
        # Link assistant node to the user turn for conversation chain continuity.
        add_link = deps.get("add_link_to_new_turn")
        ce = deps["conversation_engine"]
        if callable(add_link):
            try:
                resp_node = ce.get_nodes([response_node_id])[0]
                user_turn_node = ce.get_nodes([ctx.state["turn_node_id"]])[0]
                add_link(resp_node, user_turn_node, ctx.state["conversation_id"], span=ctx.state["self_span"], prev_turn_meta_summary=prev_turn_meta_summary)
            except Exception:
                pass

        # Mirror legacy: advance distances after adding assistant turn, if available.
        try:
            txt = getattr(resp, "answer", None)
            if isinstance(txt, str):
                prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(txt)
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        except Exception:
            pass

    outj = {
        "response_node_id": response_node_id,
        "llm_decision_need_summary": bool(getattr(resp, "llm_decision_need_summary", False)),
    }
    ctx.state["answer"] = outj
    return RunSuccess(conversation_node_id=response_node_id, outputs=[outj])


@default_resolver.register("decide_summarize")
def _decide_summarize(ctx: StepContext) -> RunResult:
    """Decide whether to summarize, using the same policy as legacy."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("decide_summarize")

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    summary_char_threshold = int(deps.get("summary_char_threshold", 12000))
    ans = ctx.state.get("answer") or {}

    should = False
    try:
        if prev_turn_meta_summary.prev_node_distance_from_last_summary - 5 >= 0:
            should = True
        if prev_turn_meta_summary.prev_node_char_distance_from_last_summary > summary_char_threshold:
            should = True
        if bool(ans.get("llm_decision_need_summary", False)):
            should = True
    except Exception:
        pass

    ctx.state["summary"] = {"should_summarize": should, "did_summarize": False, "summary_node_id": None}
    ctx.state["prev_turn_meta_summary"] = {
        "prev_node_char_distance_from_last_summary": getattr(prev_turn_meta_summary, "prev_node_char_distance_from_last_summary", 0),
        "prev_node_distance_from_last_summary": getattr(prev_turn_meta_summary, "prev_node_distance_from_last_summary", 0),
    }
    return RunSuccess(conversation_node_id=None, outputs=[ctx.state["summary"]])


@default_resolver.register("summarize")
def _summarize(ctx: StepContext) -> RunResult:
    """Summarize last batch and reset distances (legacy behavior)."""
    deps = _deps(ctx)
    ctx.state.setdefault("op_log", []).append("summarize")

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    summarize_batch = deps.get("summarize_batch")
    if not callable(summarize_batch):
        raise RuntimeError("deps['summarize_batch'] must be callable")

    added_id = summarize_batch(
        ctx.state["conversation_id"],
        int(ctx.state["turn_index"]) + 1,
        prev_turn_meta_summary=prev_turn_meta_summary,
    )

    # Legacy resets after summarization.
    try:
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
        prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
    except Exception:
        pass

    ctx.state["summary"] = {
        "should_summarize": True,
        "did_summarize": True,
        "summary_node_id": added_id,
    }
    ctx.state["prev_turn_meta_summary"] = {
        "prev_node_char_distance_from_last_summary": 0,
        "prev_node_distance_from_last_summary": 0,
    }
    return RunSuccess(conversation_node_id=added_id, outputs=[ctx.state["summary"]])


@default_resolver.register("end")
def _end(ctx: StepContext) -> RunResult:
    ctx.state.setdefault("op_log", []).append("end")
    return RunSuccess(conversation_node_id=None, outputs=[{"done": True}])