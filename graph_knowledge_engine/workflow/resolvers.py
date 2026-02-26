from __future__ import annotations


from ..models import MetaFromLastSummary, KnowledgeRetrievalResult, MemoryRetrievalResult
from .runtime import StateUpdate
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
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Union

# Best-effort self-inspection for state schema inference
import ast
import inspect
if TYPE_CHECKING:
    from tool_runner import ToolRunner

Json = Any

# Import your real RunResult types from runtime/models
from graph_knowledge_engine.workflow.runtime import RunFailure, StepRunResult, RunSuccess, StepContext
from graph_knowledge_engine.models import ConversationEdge
from graph_knowledge_engine.conversation_orchestrator import get_id_for_conversation_turn_edge
RawStepFn = Callable[[StepContext], Union[Json, StepRunResult]]

@dataclass
class MappingStepResolver:
    handlers: Dict[str, RawStepFn]
    default: Optional[RawStepFn] = None

    def __init__(self, handlers: Optional[Mapping[str, RawStepFn]] = None, *, default: Optional[RawStepFn] = None) -> None:
        self.handlers = dict(handlers or {})
        self.default = default
        # Preferred merge mode per state key: 'u' overwrite, 'a' append, 'e' extend
        self._state_schema: dict[str, str] = {}

    def register(self, op: str) -> Callable[[RawStepFn], RawStepFn]:
        def _decorator(fn: RawStepFn) -> RawStepFn:
            self.handlers[op] = fn
            return fn
        return _decorator

    def resolve(self, op: str) -> Callable[[StepContext], StepRunResult]:
        raw = self.handlers.get(op) or self.default
        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        def _wrapped(ctx: StepContext) -> StepRunResult:
            try:
                out = raw(ctx)
                if isinstance(out, (RunSuccess, RunFailure)):
                    return out
                else:
                    raise TypeError("Resolver must return StepRunResult")
                # return out  # field name might be `data`/`payload` in your codebase
            except Exception as e:
                return RunFailure(conversation_node_id=ctx.state_view.get('workflow_node_id') , state_update = [('a', {'op_log': str(e)})], errors=[str(e)])  # match your RunFailure fields
        return _wrapped


    # ------------------------------------------------------------------
    # State schema for native updates + LangGraph conversion
    # ------------------------------------------------------------------

    def set_state_schema(self, schema: Mapping[str, str]) -> None:
        """Set preferred merge mode per state key.

        Values should be one of: 'u' (overwrite), 'a' (append), 'e' (extend).
        """
        self._state_schema = {str(k): str(v) for k, v in dict(schema).items()}

    def describe_state(self) -> dict[str, str]:
        """Return the state schema for native updates / LangGraph conversion."""
        return dict(self._state_schema)

    def infer_state_schema_best_effort(self) -> dict[str, str]:
        """Best-effort inference of keys touched by resolvers.

        We intentionally keep this conservative:
        - Any key assigned via state["k"] = ...  -> 'u'
        - Any key via state.setdefault("k", []).append(...) -> 'a'
        - Any key via state.setdefault("k", []).extend(...) -> 'e'

        If inference fails, returns the existing schema unchanged.
        """
        inferred: dict[str, str] = dict(self._state_schema)

        def _note(k: str, mode: str) -> None:
            if k and mode in ("u", "a", "e"):
                # if conflicts, prefer 'u' (safest) over list modes
                prev = inferred.get(k)
                if prev is None:
                    inferred[k] = mode
                elif prev != mode:
                    inferred[k] = "u"

        for op, fn in list(self.handlers.items()):
            try:
                src = inspect.getsource(fn)
            except Exception:
                continue
            try:
                tree = ast.parse(src)
            except Exception:
                continue

            # We look for patterns, not full correctness.
            for node in ast.walk(tree):
                # state["k"] = ...
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "state":
                            sl = tgt.slice
                            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                                _note(sl.value, "u")

                # state.setdefault("k", []).append/extend
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr not in ("append", "extend"):
                        continue
                    recv = node.func.value
                    if not (isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute)):
                        continue
                    if recv.func.attr != "setdefault":
                        continue
                    base = recv.func.value
                    if not (isinstance(base, ast.Name) and base.id == "state"):
                        continue
                    if not recv.args:
                        continue
                    k0 = recv.args[0]
                    if isinstance(k0, ast.Constant) and isinstance(k0.value, str):
                        _note(k0.value, "a" if attr == "append" else "e")

        self._state_schema = dict(inferred)
        return dict(self._state_schema)

    def __call__(self, op: str) -> Callable[[StepContext], StepRunResult]:
        return self.resolve(op)
    
    
default_resolver = MappingStepResolver()


def _deps(ctx: StepContext) -> Dict[str, Any]:
    deps = ctx.state_view.get("_deps")
    if not isinstance(deps, dict):
        raise RuntimeError("StepContext.state['_deps'] must be a dict of injected dependencies")
    return deps


@default_resolver.register("start")
def _start(ctx: StepContext) -> StepRunResult:
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("start")
        state["started"] = True
    result = RunSuccess(conversation_node_id=None, state_update=[('u',{"started": True})])
    return result


@default_resolver.register("context_snapshot")
def _context_snapshot(ctx: StepContext) -> StepRunResult:
    """Persist a ContextSnapshot node capturing the *actual* LLM prompt inputs.

    Expected dependencies (best-effort):
      - deps['conversation_engine'] : GraphKnowledgeEngine (conversation)
      - deps['llm']                : model (for model_name)

    Expected state (best-effort):
      - conversation_id
      - (optional) run_id / run_step_seq / attempt_seq

    Notes:
      - This op is intentionally lightweight. The *meaning* of the snapshot is
        defined by what you pass in `llm_input_payload` and `evidence_pack_digest`.
      - For full determinism, callers should provide stable run_id/step numbers.
    """
    deps = _deps(ctx)
    ce = deps["conversation_engine"]
    llm = deps.get("llm")
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or deps.get("run_id") or f"run_{conversation_id}")
    run_step_seq = int(sv.get("run_step_seq") or deps.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or deps.get("attempt_seq") or 0)
    stage = str(sv.get("stage") or deps.get("stage") or "answer")

    # Build the prompt context (debug/telemetry artifact).
    view = ce.get_conversation_view(conversation_id=conversation_id, purpose="answer")

    snap_id = ce.persist_context_snapshot(
        conversation_id=conversation_id,
        run_id=run_id,
        run_step_seq=run_step_seq,
        attempt_seq=attempt_seq,
        stage=stage,
        view=view,
        model_name=str(getattr(llm, "model_name", "") or ""),
        budget_tokens=int(getattr(view, "token_budget", 0) or 0),
        tail_turn_index=int(getattr(deps.get("prev_turn_meta_summary"), "tail_turn_index", 0) or 0),
        llm_input_payload={
            "system_prompt": ce.get_system_prompt(conversation_id),
            "user_text": sv.get("user_text"),
        },
        evidence_pack_digest=sv.get("evidence_pack_digest"),
    )

    return RunSuccess(conversation_node_id=snap_id, state_update=[('a', {"op_log": f"context_snapshot:{snap_id}"})])


@default_resolver.register("memory_retrieve")
def _memory_retrieve(ctx: StepContext) -> StepRunResult:
    """Retrieve candidate memories.

    Writes:
      - state['memory_raw'] : MemoryRetrievalResult (non-serializable, runtime-only)
      - state['memory']     : jsonable mirror
    """
    deps = _deps(ctx)
    nid_created = []
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("memory_retrieve")

    from ..memory_retriever import MemoryRetriever
    from ..workflow.serialize import to_jsonable

    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
    )
    tool_runner: ToolRunner | None = deps.get("tool_runner")
    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    state_view = ctx.state_view
    if tool_runner is None:
        # Fallback: run directly without tool recording.
        mem, call_node_id = mem_retriever.retrieve(
            user_id=state_view["user_id"],
            current_conversation_id=state_view["conversation_id"],
            query_embedding=state_view["embedding"],
            user_text=state_view["user_text"],
            context_text="",
            n_result = 12,
        )
    else:
        
        mem, call_node_id = tool_runner.run_tool(
            conversation_id=state_view["conversation_id"],
            user_id=state_view["user_id"],
            turn_node_id=state_view["turn_node_id"],
            turn_index=state_view["turn_index"],
            tool_name="memory_retrieve",
            args=[], #{"n_results": getattr(mem_retriever, "n_results", 12)},
            kwargs = dict(
                user_id=state_view["user_id"],
                current_conversation_id=state_view["conversation_id"],
                query_embedding=state_view["embedding"],
                user_text=state_view["user_text"],
                context_text="",
                n_result = 12,
            ),
            handler=mem_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    # ctx.state["memory_raw"] = mem
    memj = to_jsonable(mem)
    state_update: list[StateUpdate] = [('u', {'memory': memj})]
    # ctx.state["memory"] = memj
    result = RunSuccess(conversation_node_id=call_node_id, state_update=state_update)
    return result


@default_resolver.register("kg_retrieve")
def _kg_retrieve(ctx: StepContext) -> StepRunResult:
    """Retrieve KG facts/links based on query and memory seed ids."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("kg_retrieve")

    from ..knowledge_retriever import KnowledgeRetriever
    from ..workflow.serialize import to_jsonable

    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )
    state_view = ctx.state_view
    mem_raw = state_view.get("memory_raw")
    seed_ids = list(getattr(mem_raw, "seed_kg_node_ids", []) or []) if mem_raw is not None else []
    tool_runner: ToolRunner = deps.get("tool_runner")
    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    state = ctx.state_view
    if tool_runner is None:
        kg, call_node_id = kg_retriever.retrieve(
            user_text=state_view["user_text"],
            context_text="",
            query_embedding=state_view["embedding"],
            seed_kg_node_ids=seed_ids,
        )
    else:
        kw_args = {
            "max_retrieval_level": max_retrieval_level, "seed_kg_node_ids": seed_ids
        }
        kw_args.update(dict(
                user_text=state["user_text"],
                context_text="",
                query_embedding=state["embedding"],
                seed_kg_node_ids=seed_ids,
            ))
        kg, call_node_id = tool_runner.run_tool(
            conversation_id=state["conversation_id"],
            user_id=state["user_id"],
            turn_node_id=state["turn_node_id"],
            turn_index=state["turn_index"],
            tool_name="kg_retrieve",
            args={"max_retrieval_level": max_retrieval_level, "seed_kg_node_ids": seed_ids},
            kwargs = kw_args,
            handler= kg_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
    # ctx.state["kg_raw"] = kg
    kgj = to_jsonable(kg)
    # ctx.state["kg"] = kgj
    state_update = [('u', {'kg': kgj})]
    return RunSuccess(conversation_node_id=call_node_id, state_update=state_update)


@default_resolver.register("memory_pin")
def _memory_pin(ctx: StepContext) -> StepRunResult:
    """Pin selected memory into the conversation graph."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("memory_pin")

    from ..memory_retriever import MemoryRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    state = ctx.state_view
    mem_rehydrated = MemoryRetrievalResult(**state.get("memj"))
    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
    )

    out = None
    if (
        mem_rehydrated is not None
        and getattr(mem_rehydrated, "selected", None)
        and getattr(mem_rehydrated, "memory_context_text", None)
    ):
        out = mem_retriever.pin_selected(
            user_id=state["user_id"],
            current_conversation_id=state["conversation_id"],
            turn_node_id=state["turn_node_id"],
            mem_id=state["mem_id"],
            turn_index=state["turn_index"],
            self_span=state["self_span"],
            selected_memory=getattr(mem_rehydrated, "selected"),
            memory_context_text=getattr(mem_rehydrated, "memory_context_text"),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    # ctx.state["memory_pin_raw"] = out
    outj = {
        "memory_context_node_id": getattr(getattr(out, "memory_context_node", None), "id", None)
        if out
        else None,
        "pinned_edge_ids": [e.id for e in getattr(out, "pinned_edges", [])] if out else [],
    }
    # ctx.state["memory_pin"] = outj
    state_update = [('u', {'memory_pin': outj})]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("kg_pin")
def _kg_pin(ctx: StepContext) -> StepRunResult:
    """Pin selected KG nodes/edges (as pointers) into the conversation graph."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("kg_pin")
    state = ctx.state_view
    from ..knowledge_retriever import KnowledgeRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm=deps["llm"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )

    kg_rehydrated = KnowledgeRetrievalResult(**state.get("kg_pin"))
    pinned_ptrs: list[str] = []
    pinned_edges: list[str] = []
    if kg_rehydrated is not None and getattr(kg_rehydrated, "selected", None):
        pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
            turn_node_id=state["turn_node_id"],
            turn_index=state["turn_index"],
            self_span=state["self_span"],
            selected_knowledge=getattr(kg_rehydrated, "selected"),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    outj = {"pinned_pointer_node_ids": list(pinned_ptrs), "pinned_edge_ids": list(pinned_edges)}
    state_update = [('u', {'kg_pin': outj})]
    # ctx.state["kg_pin"] = outj
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("answer")
def _answer(ctx: StepContext) -> StepRunResult:
    """Run answer-only agent, then link assistant turn into conversation chain."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("answer")
    state = ctx.state_view
    prev_turn_meta_summary: MetaFromLastSummary = deps.get("prev_turn_meta_summary")
    answer_only = deps.get("answer_only")
    if not callable(answer_only):
        raise RuntimeError("deps['answer_only'] must be callable")
    from ..models import ConversationAIResponse
    res_raw = answer_only(conversation_id=state["conversation_id"], prev_turn_meta_summary=prev_turn_meta_summary)
    resp : ConversationAIResponse= ConversationAIResponse.model_validate(res_raw)
    # ctx.state["answer_raw"] = resp

    response_node_id = getattr(resp, "response_node_id", None)
    if response_node_id:
        # Link assistant node to the user turn for conversation chain continuity.
        add_link_to_new_turn = deps.get("add_link_to_new_turn")
        ce = deps["conversation_engine"]
        if callable(add_link_to_new_turn):
            try:
                resp_node = ce.get_nodes([response_node_id])[0]
                state = ctx.state_view
                user_turn_node = ce.get_nodes([state["turn_node_id"]])[0]
                seq_edge_id = get_id_for_conversation_turn_edge(ConversationEdge.id_kind, state['user_id'], 
                                                                state['conversation_id'], 
                                                                "next_turn", prev_turn_meta_summary.tail_turn_index,
                                                                [user_turn_node.id], [resp_node.id], 
                                                                [], [], 
                                                                "conversation_edge")
                state = ctx.state_view
                add_link_to_new_turn(seq_edge_id, resp_node, user_turn_node, state["conversation_id"], 
                                     span=state["self_span"], 
                                     prev_turn_meta_summary=prev_turn_meta_summary)
            except Exception as _e:
                pass

        # # Mirror legacy: advance distances after adding assistant turn, if available.
        # try:
        #     txt = getattr(resp, "answer", None)
        #     if isinstance(txt, str):
        #         prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(txt)
        #     prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        # except Exception:
        #     pass

    outj = {
        "response_node_id": response_node_id,
        "llm_decision_need_summary": bool(getattr(resp, "llm_decision_need_summary", False)),
    }
    # ctx.state["answer"] = outj
    state_update=[('u', {'answer': outj})]
    return RunSuccess(conversation_node_id=response_node_id, state_update=state_update)


@default_resolver.register("decide_summarize")
def _decide_summarize(ctx: StepContext) -> StepRunResult:
    """Decide whether to summarize, using the same policy as legacy."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("decide_summarize")
    state = ctx.state_view
    prev_turn_meta_summary: MetaFromLastSummary = deps.get("prev_turn_meta_summary")
    summary_char_threshold = int(deps.get("summary_char_threshold", 12000))
    ans = state.get("answer") or {}

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

    # state["summary"] = {"should_summarize": should, "did_summarize": False, "summary_node_id": None}
    # state["prev_turn_meta_summary"] = {
    #     "prev_node_char_distance_from_last_summary": getattr(prev_turn_meta_summary, "prev_node_char_distance_from_last_summary", 0),
    #     "prev_node_distance_from_last_summary": getattr(prev_turn_meta_summary, "prev_node_distance_from_last_summary", 0),
    # }
    summary = {
        "should_summarize": bool(should),
        "summary_char_threshold": int(summary_char_threshold),
        "summary_token_threshold": int(deps.get("summary_token_threshold")) if deps.get("summary_token_threshold") is not None else None,
        "summary_turn_threshold": int(deps.get("summary_turn_threshold", 5)),
    }
    state_update: list[StateUpdate] = [('u', {"summary": summary})]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("summarize")
def _summarize(ctx: StepContext) -> StepRunResult:
    """Summarize last batch and reset distances (legacy behavior)."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("summarize")
    state = ctx.state_view
    prev_turn_meta_summary: MetaFromLastSummary = deps.get("prev_turn_meta_summary")
    summarize_batch = deps.get("summarize_batch")
    if not callable(summarize_batch):
        raise RuntimeError("deps['summarize_batch'] must be callable")

    added_id = summarize_batch(
        state["conversation_id"],
        int(state["turn_index"]) + 1,
        prev_turn_meta_summary=prev_turn_meta_summary,
    )

    # Legacy resets after summarization.
    try:
        # idenpotent, so just reset for defensive, but must be exact for tail_turn_index and not increment here
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
        prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
    except Exception:
        pass

    # ctx.state["summary"] = {
    #     "should_summarize": True,
    #     "did_summarize": True,
    #     "summary_node_id": added_id,
    # }
    # ctx.state["prev_turn_meta_summary"] = {
    #     "prev_node_char_distance_from_last_summary": 0,
    #     "prev_node_distance_from_last_summary": 0,
    # }
    summary = {
        "should_summarize": True,
        "did_summarize": True,
        "summary_node_id": added_id,
    }
    mts = {
        "prev_node_char_distance_from_last_summary": 0,
        "prev_node_distance_from_last_summary": 0,
        "tail_turn_index": int(getattr(prev_turn_meta_summary, "tail_turn_index", 0)),
    }
    state_update: list[StateUpdate] = [('u', {"summary": summary}), ('u', {"prev_turn_meta_summary": mts})]
    result = RunSuccess(conversation_node_id=added_id, state_update=state_update)
    return result


@default_resolver.register("end")
def _end(ctx: StepContext) -> StepRunResult:
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("end")
    result = RunSuccess(conversation_node_id=None, state_update=[('u',{"done": True})])
    return result