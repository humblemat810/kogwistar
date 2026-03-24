from __future__ import annotations

from .models import (
    ConversationEdge,
    ConversationNode,
    KnowledgeRetrievalResult,
    MemoryRetrievalResult,
)

from .models import MetaFromLastSummary
from ..runtime.models import StateUpdate
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.runtime.models import StateUpdate
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

import os
import pathlib
import time

from ..utils.embedding_vectors import normalize_embedding_vector
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

# Best-effort self-inspection for state schema inference

Json = Any
if TYPE_CHECKING:
    from .tool_runner import ToolRunner
    from kogwistar.runtime.runtime import StepContext
    from kogwistar.runtime.runtime import StepRunResult

    RawStepFn = Callable[[StepContext], Union[Json, StepRunResult]]

from kogwistar.runtime.models import RunSuccess

# Import your real RunResult types from kogwistar.runtime/models


from kogwistar.engine_core.models import Span


# Agentic answering helper types
from .agentic_answering import (
    AgenticAnsweringAgent,
    snapshot_hash,
    AnswerWithCitations,
    AnswerEvaluation,
)

from kogwistar.runtime.resolvers import MappingStepResolver

default_resolver = MappingStepResolver()
conversation_default_resolver = default_resolver


def _deps(ctx: StepContext) -> Dict[str, Any]:
    deps = ctx.state_view.get("_deps")
    if not isinstance(deps, dict):
        raise RuntimeError(
            "StepContext.state['_deps'] must be a dict of injected dependencies"
        )
    return deps


def _chat_service(deps: Dict[str, Any]):
    from .service import ConversationService

    ce = deps["conversation_engine"]
    ke = deps.get("knowledge_engine") or deps.get("ref_knowledge_engine") or ce
    we = deps.get("workflow_engine") or getattr(ce, "workflow_engine", None)
    return ConversationService.from_engine(ce, knowledge_engine=ke, workflow_engine=we)


def _aa_agent(ctx: StepContext):
    deps = _deps(ctx)
    agent: AgenticAnsweringAgent | None = deps.get("agent")
    if agent is None:
        raise Exception("agent missing in dependency")
    if agent is None:
        raise RuntimeError("agentic answering ops require deps['agent']")
    return agent


@default_resolver.register("start")
def _start(ctx: StepContext) -> StepRunResult:
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("start")
        state["started"] = True
    result = RunSuccess(
        conversation_node_id=None,
        state_update=[("u", {"started": True}), ("u", {"turn_index": 0})],
    )
    return result


@default_resolver.register("noop")
def _noop(ctx: StepContext) -> StepRunResult:
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("noop")
        turn_index = state.get("turn_index")

    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)
    if type(turn_index) is int:
        pass
    else:
        turn_index = int(getattr(mts, "tail_turn_index", 0) or 0)
    turn_index += 1
    result = RunSuccess(
        conversation_node_id=None, state_update=[("u", {"turn_index": turn_index})]
    )
    return result


# ------------------------------------------------------------------
# Phase B.1 backbone ops (conversation chain mutations as primitives)
# ------------------------------------------------------------------


def _get_prev_turn_meta_summary_from_state_or_deps(ctx: "StepContext") -> Any:
    """Return a MetaFromLastSummary-like object.

    - Prefer deps['prev_turn_meta_summary'] (mutable, legacy object).
    - Else, synthesize a lightweight object from state['prev_turn_meta_summary'] dict.
    """
    deps = _deps(ctx)
    mts = deps.get("prev_turn_meta_summary")
    if mts is not None:
        return mts
    d = ctx.state_view.get("prev_turn_meta_summary") or {}

    # tiny duck-typed object
    class _MTS:
        __slots__ = (
            "prev_node_char_distance_from_last_summary",
            "prev_node_distance_from_last_summary",
            "tail_turn_index",
        )

        def __init__(self, dd: dict[str, Any]):
            self.prev_node_char_distance_from_last_summary = int(
                dd.get("prev_node_char_distance_from_last_summary", 0)
            )
            self.prev_node_distance_from_last_summary = int(
                dd.get("prev_node_distance_from_last_summary", 0)
            )
            self.tail_turn_index = int(dd.get("tail_turn_index", 0))

    return _MTS(dict(d))


@default_resolver.register("add_user_turn")
def _add_user_turn(ctx: "StepContext") -> "StepRunResult":
    """Create/persist the user turn node (workflow primitive).

    Expected state (best-effort):
      - user_id, conversation_id, role, user_text
      - turn_index (optional; defaults to mts.tail_turn_index)
      - turn_id (optional; used as doc_id; defaults to state['turn_node_id'] if set)

    Writes (via state_update):
      - turn_node_id, turn_index, self_span, embedding
      - prev_turn_meta_summary (json mirror if synthesized)
    """
    deps = _deps(ctx)
    ce = deps["conversation_engine"]
    sv = ctx.state_view

    user_id = str(sv["user_id"])
    conversation_id = str(sv["conversation_id"])
    role = str(sv.get("role") or "user")
    user_text = str(sv.get("user_text") or "")

    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)
    if type(sv.get("turn_index")) is int:
        turn_index = sv.get("turn_index")
    else:
        turn_index = int(getattr(mts, "tail_turn_index", 0) or 0) + 1
    # doc_id for the turn (keep external caller's turn_id if provided)
    turn_doc_id = str(
        sv.get("turn_id") or sv.get("turn_node_id") or f"turn_{turn_index}"
    )

    # deterministic node id (align with orchestrator)
    from kogwistar.conversation.conversation_orchestrator import (
        get_id_for_conversation_turn,
    )
    from kogwistar.engine_core.models import (
        Grounding,
        MentionVerification,
        Span,
    )

    # deterministic node id (align with orchestrator); prefer provided turn_node_id if present.
    expected_id = get_id_for_conversation_turn(
        ConversationNode.id_kind,
        user_id,
        conversation_id,
        user_text,
        str(turn_index),
        role,
        "conversation_turn",
        str(bool(sv.get("in_conv", True))),
    )
    provided_id = sv.get("turn_node_id")
    node_id = str(provided_id) if provided_id else str(expected_id)

    # embedding: prefer provided, else compute defensively if engine supports it
    embedding = sv.get("embedding")
    if embedding is None:
        emb_text0 = f"{role}: {user_text}"
        if hasattr(ce, "embed") and hasattr(ce.embed, "iterative_defensive_emb"):
            embedding = ce.embed.iterative_defensive_emb(emb_text0)
        elif hasattr(ce, "iterative_defensive_emb"):
            embedding = ce.iterative_defensive_emb(emb_text0)
        elif hasattr(ce, "_iterative_defensive_emb"):
            embedding = ce._iterative_defensive_emb(emb_text0)
    embedding = normalize_embedding_vector(embedding)
    span = Span(
        collection_page_url=f"conversation/{conversation_id}",
        document_page_url=f"conversation/{conversation_id}#{turn_doc_id}",
        doc_id=f"conv:{conversation_id}",
        insertion_method="conversation_turn",
        page_number=1,
        start_char=0,
        end_char=len(user_text),
        excerpt=user_text,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="verbatim input"
        ),
    )

    node = ConversationNode(
        user_id=user_id,
        id=node_id,
        label=f"Turn {turn_index} ({role})",
        type="entity",
        doc_id=turn_doc_id,
        summary=user_text,
        role=role,  # type: ignore
        turn_index=turn_index,
        conversation_id=conversation_id,
        mentions=[Grounding(spans=[span])],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": bool(sv.get("in_conv", True)),
            "in_ui_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
    )
    try:
        node.embedding = embedding
    except Exception:
        pass

    ce.write.add_node(node)

    # advance legacy distances (mirror orchestrator) if mts is mutable

    mts.tail_turn_index = turn_index
    mts.prev_node_char_distance_from_last_summary += len(user_text)
    mts.prev_node_distance_from_last_summary += 1

    with ctx.state_write as state:
        state.setdefault("op_log", []).append("add_user_turn")

    mts_json = None
    # if we synthesized mts (duck type), mirror it back as json so runtime can checkpoint it
    try:
        mts_json = {
            "prev_node_char_distance_from_last_summary": int(
                getattr(mts, "prev_node_char_distance_from_last_summary", 0)
            ),
            "prev_node_distance_from_last_summary": int(
                getattr(mts, "prev_node_distance_from_last_summary", 0)
            ),
            "tail_turn_index": int(getattr(mts, "tail_turn_index", 0)),
        }
    except Exception:
        mts_json = None

    upd = {
        "turn_node_id": node_id,
        "turn_index": turn_index,
        "self_span": span,
    }
    if embedding is not None:
        upd["embedding"] = embedding
    if mts_json is not None:
        upd["prev_turn_meta_summary"] = mts_json
    upd["turn_index"] = turn_index
    return RunSuccess(conversation_node_id=node_id, state_update=[("u", upd)])


@default_resolver.register("link_prev_turn")
def _link_prev_turn(ctx: "StepContext") -> "StepRunResult":
    """Link prev tail turn -> current turn via next_turn edge."""
    deps = _deps(ctx)
    ce = deps["conversation_engine"]
    add_link_to_new_turn = deps.get("add_link_to_new_turn")
    if not callable(add_link_to_new_turn):
        raise RuntimeError(
            "deps['add_link_to_new_turn'] must be callable for link_prev_turn"
        )

    sv = ctx.state_view
    prev_turn_id = sv.get("prev_turn_id") or sv.get("prev_node_id")
    if not prev_turn_id:
        # nothing to link
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"op_log": "link_prev_turn:skipped"})],
        )

    turn_node_id = str(sv["turn_node_id"])
    conversation_id = str(sv["conversation_id"])
    user_id = str(sv["user_id"])
    turn_index = int(sv.get("turn_index") or 0)
    span = sv.get("self_span")

    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)

    from .conversation_orchestrator import get_id_for_conversation_turn_edge
    from .models import ConversationEdge

    prev_node = ce.read.get_nodes([str(prev_turn_id)])[0]
    turn_node = ce.read.get_nodes([turn_node_id])[0]

    edge_id = get_id_for_conversation_turn_edge(
        ConversationEdge.id_kind,
        user_id,
        conversation_id,
        "next_turn",
        turn_index,
        [prev_node.id],
        [turn_node.id],
        [],
        [],
        "conversation_edge",
    )

    add_link_to_new_turn(
        edge_id,
        turn_node,
        prev_node,
        conversation_id,
        span=span,
        prev_turn_meta_summary=mts,
    )

    with ctx.state_write as state:
        state.setdefault("op_log", []).append("link_prev_turn")

    return RunSuccess(
        conversation_node_id=edge_id,
        state_update=[("u", {"prev_turn_edge_id": edge_id})],
    )


@default_resolver.register("link_assistant_turn")
def _link_assistant_turn(ctx: "StepContext") -> "StepRunResult":
    """Link user turn -> assistant response turn via next_turn edge."""
    deps = _deps(ctx)
    ce = deps["conversation_engine"]
    add_link_to_new_turn = deps.get("add_link_to_new_turn")
    if not callable(add_link_to_new_turn):
        raise RuntimeError(
            "deps['add_link_to_new_turn'] must be callable for link_assistant_turn"
        )

    sv = ctx.state_view
    ans = sv.get("answer") or {}
    response_node_id = None
    if isinstance(ans, dict):
        response_node_id = ans.get("response_node_id") or ans.get(
            "assistant_turn_node_id"
        )
    if not response_node_id:
        aa_res = sv.get("agentic_answering_result") or {}
        if isinstance(aa_res, dict):
            response_node_id = aa_res.get("assistant_turn_node_id") or aa_res.get(
                "response_node_id"
            )
    if not response_node_id:
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"op_log": "link_assistant_turn:skipped"})],
        )

    user_turn_id = str(sv["turn_node_id"])
    conversation_id = str(sv["conversation_id"])
    user_id = str(sv["user_id"])
    turn_index = int(sv.get("turn_index") or 0)
    span = sv.get("self_span")
    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)

    from .conversation_orchestrator import get_id_for_conversation_turn_edge

    user_turn = ce.read.get_nodes([user_turn_id])[0]
    resp_turn = ce.read.get_nodes([str(response_node_id)])[0]

    edge_id = get_id_for_conversation_turn_edge(
        ConversationEdge.id_kind,
        user_id,
        conversation_id,
        "next_turn",
        turn_index,
        [user_turn.id],
        [resp_turn.id],
        [],
        [],
        "conversation_edge",
    )

    add_link_to_new_turn(
        edge_id,
        resp_turn,
        user_turn,
        conversation_id,
        span=span,
        prev_turn_meta_summary=mts,
    )

    with ctx.state_write as state:
        state.setdefault("op_log", []).append("link_assistant_turn")

    return RunSuccess(
        conversation_node_id=edge_id,
        state_update=[
            ("u", {"assistant_turn_edge_id": edge_id, "turn_index": turn_index})
        ],
    )


@default_resolver.register("context_snapshot")
def _context_snapshot(ctx: StepContext) -> StepRunResult:
    """Persist a ContextSnapshot node capturing the *actual* LLM prompt inputs.

    Expected dependencies (best-effort):
      - deps['conversation_engine'] : GraphKnowledgeEngine (conversation)
      - deps['llm_tasks']           : task set (for provider label)

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
    svc = _chat_service(deps)
    llm_tasks = deps.get("llm_tasks")
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or deps.get("run_id") or f"run_{conversation_id}")
    run_step_seq = int(sv.get("run_step_seq") or deps.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or deps.get("attempt_seq") or 0)
    stage = str(sv.get("stage") or deps.get("stage") or "answer")

    # Build the prompt context (debug/telemetry artifact).
    view = svc.get_conversation_view(conversation_id=conversation_id, purpose="answer")
    snap_id = svc.persist_context_snapshot(
        conversation_id=conversation_id,
        run_id=run_id,
        run_step_seq=run_step_seq,
        attempt_seq=attempt_seq,
        stage=stage,
        view=view,
        model_name=str(
            getattr(
                getattr(llm_tasks, "provider_hints", None),
                "answer_with_citations_provider",
                "",
            )
            or ""
        ),
        budget_tokens=int(getattr(view, "token_budget", 0) or 0),
        tail_turn_index=int(
            getattr(deps.get("prev_turn_meta_summary"), "tail_turn_index", 0) or 0
        ),
        llm_input_payload={
            "system_prompt": svc.get_system_prompt(conversation_id),
            "user_text": sv.get("user_text"),
        },
        evidence_pack_digest=sv.get("evidence_pack_digest"),
    )

    return RunSuccess(
        conversation_node_id=snap_id,
        state_update=[("a", {"op_log": f"context_snapshot:{snap_id}"})],
    )


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

    from .memory_retriever import MemoryRetriever
    from ..runtime.serialize import to_jsonable

    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm_tasks=deps["llm_tasks"],
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
            n_results=12,
        )
    else:
        mem, call_node_id = tool_runner.run_tool(
            conversation_id=state_view["conversation_id"],
            user_id=state_view["user_id"],
            turn_node_id=state_view["turn_node_id"],
            turn_index=state_view["turn_index"],
            tool_name="memory_retrieve",
            args=[],  # {"n_results": getattr(mem_retriever, "n_results", 12)},
            kwargs=dict(
                user_id=state_view["user_id"],
                current_conversation_id=state_view["conversation_id"],
                query_embedding=state_view["embedding"],
                user_text=state_view["user_text"],
                context_text="",
                n_results=12,
            ),
            handler=mem_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    memj = to_jsonable(mem)
    state_update: list[StateUpdate] = [("u", {"memory": memj})]
    result = RunSuccess(conversation_node_id=call_node_id, state_update=state_update)
    return result


@default_resolver.register("kg_retrieve")
def _kg_retrieve(ctx: StepContext) -> StepRunResult:
    """Retrieve KG facts/links based on query and memory seed ids."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("kg_retrieve")

    from .knowledge_retriever import KnowledgeRetriever
    from ..runtime.serialize import to_jsonable

    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm_tasks=deps["llm_tasks"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )
    state_view = ctx.state_view
    mem_raw = state_view.get("memory_raw")
    seed_ids = (
        list(getattr(mem_raw, "seed_kg_node_ids", []) or [])
        if mem_raw is not None
        else []
    )
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
            "max_retrieval_level": max_retrieval_level,
            "seed_kg_node_ids": seed_ids,
        }
        kw_args.update(
            dict(
                user_text=state["user_text"],
                context_text="",
                query_embedding=state["embedding"],
                seed_kg_node_ids=seed_ids,
            )
        )
        kg, call_node_id = tool_runner.run_tool(
            conversation_id=state["conversation_id"],
            user_id=state["user_id"],
            turn_node_id=state["turn_node_id"],
            turn_index=state["turn_index"],
            tool_name="kg_retrieve",
            args={
                "max_retrieval_level": max_retrieval_level,
                "seed_kg_node_ids": seed_ids,
            },
            kwargs=kw_args,
            handler=kg_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
    # ctx.state["kg_raw"] = kg
    kgj = to_jsonable(kg)
    state_update = [("u", {"kg": kgj})]
    return RunSuccess(conversation_node_id=call_node_id, state_update=state_update)


@default_resolver.register("memory_pin")
def _memory_pin(ctx: StepContext) -> StepRunResult:
    """Pin selected memory into the conversation graph."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("memory_pin")

    from .memory_retriever import MemoryRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    state = ctx.state_view
    mem_rehydrated = MemoryRetrievalResult(**state.get("memory"))
    mem_retriever = MemoryRetriever(
        conversation_engine=deps["conversation_engine"],
        llm_tasks=deps["llm_tasks"],
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
        "memory_context_node_id": getattr(
            getattr(out, "memory_context_node", None), "id", None
        )
        if out
        else None,
        "pinned_edge_ids": [e.id for e in getattr(out, "pinned_edges", [])]
        if out
        else [],
    }
    # ctx.state["memory_pin"] = outj
    state_update = [("u", {"memory_pin": outj})]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("kg_pin")
def _kg_pin(ctx: StepContext) -> StepRunResult:
    """Pin selected KG nodes/edges (as pointers) into the conversation graph."""
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("kg_pin")
    state = ctx.state_view
    from .knowledge_retriever import KnowledgeRetriever

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    max_retrieval_level = int(deps.get("max_retrieval_level", 2))

    kg_retriever = KnowledgeRetriever(
        conversation_engine=deps["conversation_engine"],
        ref_knowledge_engine=deps["ref_knowledge_engine"],
        llm_tasks=deps["llm_tasks"],
        filtering_callback=deps["filtering_callback"],
        max_retrieval_level=max_retrieval_level,
    )

    kg_rehydrated = KnowledgeRetrievalResult(**state.get("kg"))
    pinned_ptrs: list[str] = []
    pinned_edges: list[str] = []
    if kg_rehydrated is not None and getattr(kg_rehydrated, "selected", None):
        from .models import FilteringResult

        # todo change model into pydantic if possible
        selected = FilteringResult(**kg_rehydrated.selected)
        pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
            turn_node_id=state["turn_node_id"],
            turn_index=state["turn_index"],
            self_span=state["self_span"],
            selected_knowledge=selected,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    outj = {
        "pinned_pointer_node_ids": list(pinned_ptrs),
        "pinned_edge_ids": list(pinned_edges),
    }
    state_update = [("u", {"kg_pin": outj})]
    # ctx.state["kg_pin"] = outj
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("answer", is_nested=True)
def _answer(ctx: StepContext) -> StepRunResult:
    """Run answering step (no chain linking here).

    Answering capability selection is explicit:
    - Use `deps['agent'].answer_workflow_v2(...)` when an agent is available.
    - Use `deps['answer_only'](...)` only when the caller explicitly forces that
      path, or when no agentic workflow capability is present.

    `answer_only` is not a hidden error fallback. It is the explicit local
    "use the current conversation view plus assembled KG evidence to answer
    now" path. Callers can use it when they want KG-backed answering without
    depending on the full workflow-driven assistant-turn materialization flow.

    Writes:
      state['answer'] = {'response_node_id': ..., 'llm_decision_need_summary': bool}
    """
    deps = _deps(ctx)
    cache_dir = getattr(ctx, "cache_dir", None)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("answer")
    state = ctx.state_view
    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)
    answer_only = deps.get("answer_only")
    force_answer_only = bool(deps.get("force_answer_only", False))

    response_node_id: str | None = None
    llm_decision_need_summary: bool = False
    assistant_text: str = ""

    # 1) Choose the answering capability explicitly.
    agent: AgenticAnsweringAgent | None = None if force_answer_only else deps.get("agent")
    agent_answer = getattr(agent, "answer_workflow_v2", None) if agent is not None else None

    if callable(agent_answer):
        workflow_engine = deps.get("agentic_workflow_engine") or deps.get(
            "workflow_engine"
        )
        out = agent_answer(
            conversation_id=state["conversation_id"],
            user_id=state.get("user_id"),
            prev_turn_meta_summary=mts,
            workflow_engine=workflow_engine,
            # reuse current runtime's emitter to avoid duplicate sqlite writers in nested runs
            events=ctx.events,
            trace=(ctx.events is None),
        )
        mts.tail_turn_index += 1
        if isinstance(out, dict):
            response_node_id = out.get("assistant_turn_node_id") or out.get(
                "response_node_id"
            )
            assistant_text = str(
                out.get("assistant_text") or out.get("text") or out.get("content") or ""
            )
            # agentic path doesn't currently expose llm_decision_need_summary; keep False unless present
            llm_decision_need_summary = bool(
                out.get("llm_decision_need_summary", False)
            )
    else:
        if not callable(answer_only):
            raise RuntimeError(
                "deps must provide either callable agent.answer_workflow_v2 or callable answer_only"
            )
        res_raw = answer_only(
            conversation_id=state["conversation_id"], prev_turn_meta_summary=mts,
            cache_dir=cache_dir
        )
        try:
            from .models import ConversationAIResponse

            resp = ConversationAIResponse.model_validate(res_raw)
            response_node_id = getattr(resp, "response_node_id", None)
            llm_decision_need_summary = bool(
                getattr(resp, "llm_decision_need_summary", False)
            )
            assistant_text = str(getattr(resp, "text", "") or "")
        except Exception:
            if isinstance(res_raw, dict):
                response_node_id = res_raw.get("response_node_id") or res_raw.get(
                    "assistant_turn_node_id"
                )
                llm_decision_need_summary = bool(
                    res_raw.get("llm_decision_need_summary", False)
                )
                assistant_text = str(
                    res_raw.get("text")
                    or res_raw.get("assistant_text")
                    or res_raw.get("content")
                    or ""
                )
            else:
                response_node_id = getattr(res_raw, "response_node_id", None) or getattr(
                    res_raw, "assistant_turn_node_id", None
                )
                llm_decision_need_summary = bool(
                    getattr(res_raw, "llm_decision_need_summary", False)
                )
                assistant_text = str(
                    getattr(res_raw, "text", None)
                    or getattr(res_raw, "assistant_text", None)
                    or getattr(res_raw, "content", None)
                    or ""
                )
                if response_node_id is None and not assistant_text.strip():
                    raise TypeError(
                        "answer_only returned an unsupported response without assistant text or response node id"
                    )

    # 3) Backward-compatible materialization:
    # if an explicit answer path returned text but not a persisted assistant
    # node, materialize one here so downstream workflow state still has a
    # concrete assistant turn to reference.
    if assistant_text:
        ce = deps.get("conversation_engine")
        if ce is not None:
            try:
                from .models import ConversationNode
                from .conversation_orchestrator import get_id_for_conversation_turn
                from kogwistar.engine_core.models import (
                    Grounding,
                    MentionVerification,
                    Span,
                )

                conversation_id = str(state["conversation_id"])
                user_id = str(state.get("user_id") or "")
                role = "assistant"
                in_conv = bool(state.get("in_conv", True))
                assistant_turn_index = int(state.get("turn_index") or 0) + 1

                if response_node_id is None:
                    response_node_id = get_id_for_conversation_turn(
                        ConversationNode.id_kind,
                        user_id,
                        conversation_id,
                        assistant_text,
                        str(assistant_turn_index),
                        role,
                        "conversation_turn",
                        str(in_conv),
                    )
                response_node_id = str(response_node_id)

                got = ce.backend.node_get(ids=[response_node_id], include=[])
                if not (got.get("ids") or []):
                    span = Span(
                        collection_page_url=f"conversation/{conversation_id}",
                        document_page_url=f"conversation/{conversation_id}#{response_node_id}",
                        doc_id=f"conv:{conversation_id}",
                        insertion_method="assistant_turn",
                        page_number=1,
                        start_char=0,
                        end_char=len(assistant_text),
                        excerpt=assistant_text,
                        context_before="",
                        context_after="",
                        chunk_id=None,
                        source_cluster_id=None,
                        verification=MentionVerification(
                            method="system",
                            is_verified=True,
                            score=1.0,
                            notes="resolver:answer fallback",
                        ),
                    )
                    node = ConversationNode(
                        id=response_node_id,
                        user_id=user_id,
                        label=f"Turn {assistant_turn_index} ({role})",
                        type="entity",
                        doc_id=response_node_id,
                        summary=assistant_text,
                        role=role,  # type: ignore[arg-type]
                        turn_index=assistant_turn_index,
                        conversation_id=conversation_id,
                        mentions=[Grounding(spans=[span])],
                        properties={},
                        metadata={
                            "entity_type": "conversation_turn",
                            "level_from_root": 0,
                            "in_conversation_chain": in_conv,
                            "in_ui_chain": True,
                        },
                        domain_id=None,
                        canonical_entity_id=None,
                    )

                    emb_text0 = f"{role}: {assistant_text}"
                    embedding = None
                    if hasattr(ce, "embed") and hasattr(
                        ce.embed, "iterative_defensive_emb"
                    ):
                        embedding = ce.embed.iterative_defensive_emb(emb_text0)
                    elif hasattr(ce, "iterative_defensive_emb"):
                        embedding = ce.iterative_defensive_emb(emb_text0)
                    elif hasattr(ce, "_iterative_defensive_emb"):
                        embedding = ce._iterative_defensive_emb(emb_text0)
                    if embedding is not None:
                        node.embedding = normalize_embedding_vector(
                            embedding, allow_none=False
                        )
                    ce.write.add_node(node)
            except Exception:
                # Keep previous behavior if fallback materialization fails for any reason.
                pass

    if (
        bool(deps.get("strict_answer_failure", False))
        and response_node_id is None
        and not str(assistant_text or "").strip()
    ):
        raise RuntimeError("Answer step produced no assistant response.")

    outj = {
        "response_node_id": response_node_id,
        "llm_decision_need_summary": bool(llm_decision_need_summary),
    }
    return RunSuccess(
        conversation_node_id=response_node_id, state_update=[("u", {"answer": outj})]
    )


@default_resolver.register("decide_summarize")
def _decide_summarize(ctx: StepContext) -> StepRunResult:
    """Decide whether to summarize.

    Phase D policy:
    - Prefer latest context_snapshot cost (if available) for char/token thresholds.
    - Fall back to legacy distance counters if no snapshot exists.
    - Always respect answer.llm_decision_need_summary if present.
    """
    deps = _deps(ctx)
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("decide_summarize")
    st = ctx.state_view
    mts = _get_prev_turn_meta_summary_from_state_or_deps(ctx)

    summary_char_threshold = int(deps.get("summary_char_threshold", 12000))
    summary_turn_threshold = int(deps.get("summary_turn_threshold", 5))
    summary_token_threshold = deps.get("summary_token_threshold")
    token_estimator = deps.get("token_estimator")  # optional callable
    stage = deps.get("summary_snapshot_stage") or "generate_answer_with_citations"

    ans = st.get("answer") or {}

    should = False

    # 1) Snapshot-first
    snap_cost = None
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    ce: GraphKnowledgeEngine = deps.get("conversation_engine")
    if ce is not None and hasattr(ce, "latest_context_snapshot_cost"):
        try:
            snap_cost = ce.latest_context_snapshot_cost(
                conversation_id=st["conversation_id"], stage=stage
            )
        except Exception:
            snap_cost = None

    if snap_cost is not None:
        try:
            if int(getattr(snap_cost, "char_count", 0)) > summary_char_threshold:
                should = True
            if summary_token_threshold is not None:
                tc = getattr(snap_cost, "token_count", None)
                if tc is not None and int(tc) > int(summary_token_threshold):
                    should = True
        except Exception:
            pass
    else:
        # 2) Legacy fallback
        try:
            if (
                int(getattr(mts, "prev_node_distance_from_last_summary", 0))
                - summary_turn_threshold
                >= 0
            ):
                should = True
            char_dist = int(
                getattr(mts, "prev_node_char_distance_from_last_summary", 0)
            )
            if char_dist > summary_char_threshold:
                should = True
            if summary_token_threshold is not None:
                # best-effort token estimate
                if callable(token_estimator):
                    tok = (
                        int(token_estimator("a" * min(4096, char_dist)))
                        if char_dist > 0
                        else 0
                    )
                    # scale linearly using proxy chars
                    proxy = max(1, min(4096, char_dist))
                    est = int(round(tok * (char_dist / proxy))) if proxy else tok
                else:
                    est = max(1, (char_dist + 3) // 4)
                if est > int(summary_token_threshold):
                    should = True
        except Exception:
            pass

    # 3) LLM decision flag
    if bool(ans.get("llm_decision_need_summary", False)):
        should = True

    summary = {
        "should_summarize": bool(should),
        "summary_char_threshold": int(summary_char_threshold),
        "summary_token_threshold": int(summary_token_threshold)
        if summary_token_threshold is not None
        else None,
        "summary_turn_threshold": int(summary_turn_threshold),
    }
    return RunSuccess(
        conversation_node_id=None, state_update=[("u", {"summary": summary})]
    )


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
    cache_dir = ctx.cache_dir
    added_id = summarize_batch(
        state["conversation_id"],
        int(state["turn_index"]) + 1,
        prev_turn_meta_summary=prev_turn_meta_summary,
        cache_dir=cache_dir
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
    prev_turn_meta_summary.tail_turn_index += 1
    mts = {
        "prev_node_char_distance_from_last_summary": 0,
        "prev_node_distance_from_last_summary": 0,
        "tail_turn_index": int(getattr(prev_turn_meta_summary, "tail_turn_index", 0)),
    }
    state_update: list[StateUpdate] = [
        ("u", {"summary": summary}),
        ("u", {"prev_turn_meta_summary": mts}),
    ]
    result = RunSuccess(conversation_node_id=added_id, state_update=state_update)
    return result


@default_resolver.register("end")
def _end(ctx: StepContext) -> StepRunResult:
    with ctx.state_write as state:
        state.setdefault("op_log", []).append("end")
    result = RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])
    return result


# =====================================================================
# Agentic Answering workflow ops (rewrite of AgenticAnsweringAgent.answer)
# =====================================================================


@default_resolver.register("aa_prepare")
def _aa_prepare(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    sv = ctx.state_view
    conversation_id = str(sv["conversation_id"])

    # Keep these aligned with agentic_answering.answer() defaults.
    run_id = str(sv.get("run_id") or f"run_{int(time.time() * 1000)}")

    # Create run anchor in canvas.
    run_node_id = agent._ensure_run_anchor(
        conversation_id=conversation_id, run_id=run_id
    )

    # Step identity counters.
    run_step_seq = int(sv.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or 0)
    iter_idx = int(sv.get("iter_idx") or 0)

    # Policy knobs that can escalate across iterations.
    max_candidates = int(sv.get("max_candidates") or agent.config.max_candidates)
    materialize_depth = str(
        sv.get("materialize_depth") or agent.config.materialize_depth
    )
    state_update: list[StateUpdate] = [
        (
            "u",
            {
                "run_id": run_id,
                "run_node_id": run_node_id,
                "run_step_seq": run_step_seq,
                "attempt_seq": attempt_seq,
                "iter_idx": iter_idx,
                "max_candidates": max_candidates,
                "materialize_depth": materialize_depth,
            },
        ),
        ("a", {"op_log": "aa_prepare"}),
    ]
    return RunSuccess(conversation_node_id=run_node_id, state_update=state_update)


@default_resolver.register("aa_get_view_and_question")
def _aa_get_view_and_question(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    sv = ctx.state_view
    conversation_id = str(sv["conversation_id"])
    user_id = sv.get("user_id")
    svc = _chat_service(deps)

    # Fetch conversation state.
    view = svc.get_conversation_view(
        conversation_id=conversation_id,
        user_id=user_id,
        purpose="answer",
        budget_tokens=6000,
    )
    messages = getattr(view, "messages", None)
    question = agent._get_last_user_text(messages)
    system_prompt = svc.get_system_prompt(conversation_id)
    if not question:
        from kogwistar.runtime.models import RunFailure

        return RunFailure(
            conversation_node_id=None,
            state_update=[
                ("a", {"op_log": "aa_get_view_and_question: no user message"})
            ],
            errors=["No user message found in conversation"],
        )

    # Store view runtime-only (may not be jsonable).
    with ctx.state_write as state:
        state.setdefault("_rt", {})["view"] = view

    state_update: list[StateUpdate] = [
        ("u", {"system_prompt": system_prompt, "question": question}),
        ("a", {"op_log": "aa_get_view_and_question"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_retrieve_candidates")
def _aa_retrieve_candidates(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    sv = ctx.state_view
    question = str(sv.get("question") or "")
    max_candidates = int(sv.get("max_candidates") or agent.config.max_candidates)

    candidates = agent._retrieve_candidates(question)[:max_candidates]
    state_update: list[StateUpdate] = [
        ("u", {"candidates": candidates}),
        ("a", {"op_log": f"aa_retrieve_candidates:{len(candidates)}"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_select_used_evidence")
def _aa_select_used_evidence(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    svc = _chat_service(deps)
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or "")
    run_step_seq = int(sv.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or 0)

    question = str(sv.get("question") or "")
    system_prompt = str(sv.get("system_prompt") or "")
    candidates = list(sv.get("candidates") or [])

    # Pull view for snapshots.
    view = (sv.get("_rt") or {}).get("view")
    if view is None:
        view = svc.get_conversation_view(
            conversation_id=conversation_id, purpose="answer"
        )
        with ctx.state_write as state:
            state.setdefault("_rt", {})["view"] = view

    if (agent.config.evidence_selector or "llm").lower() == "bm25":
        selection = agent._select_used_evidence_bm25(
            question=question, candidates=candidates
        )
        selection_dict = selection.model_dump()
    else:
        prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
        agent._persist_context_snapshot(
            conversation_id=conversation_id,
            run_id=run_id,
            run_step_seq=run_step_seq,
            attempt_seq=attempt_seq,
            stage="select_used_evidence",
            view=view,
            model_name=str(
                getattr(
                    getattr(deps.get("llm_tasks"), "provider_hints", None),
                    "filter_candidates_provider",
                    "",
                )
                or ""
            ),
            budget_tokens=int(getattr(view, "token_budget", 0) or 0),
            tail_turn_index=int(
                getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0
            ),
            extra_hash_payload={
                "question": question,
                "candidate_ids": [
                    c.get("id")
                    for c in (candidates or [])
                    if isinstance(c, dict) and c.get("id")
                ],
            },
            llm_input_payload={
                "system_prompt": system_prompt,
                "question": question,
                "candidate_ids": [
                    c.get("id")
                    for c in (candidates or [])
                    if isinstance(c, dict) and c.get("id")
                ],
            },
        )
        run_step_seq += 1
        selection = agent.select_used_evidence_cached(
            system_prompt=system_prompt,
            question=question,
            candidates=candidates,
            cache_dir=agent.cache_dir,
        )
        selection_dict = selection.model_dump()

    used_node_ids = list(
        (selection_dict.get("used_node_ids") or [])[: agent.config.max_used]
    )
    state_update: list[StateUpdate] = [
        (
            "u",
            {
                "selection": selection_dict,
                "used_node_ids": used_node_ids,
                "run_step_seq": run_step_seq,
            },
        ),
        ("a", {"op_log": f"aa_select_used_evidence:{len(used_node_ids)}"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_materialize_evidence_pack")
def _aa_materialize_evidence_pack(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    sv = ctx.state_view
    used_node_ids = list(sv.get("used_node_ids") or [])
    used_edge_ids = list(sv.get("used_edge_ids") or [])
    materialize_depth = str(
        sv.get("materialize_depth") or agent.config.materialize_depth
    )

    from ..utils.pydanic_model_consumer_wrapper import cache_pydantic_structured
    from joblib import Memory
    from .models import EvidencePackDigest

    mem = Memory(
        location=os.path.join(agent.cache_dir, "_materialize_evidence_pack")
    )
    cached_call = cache_pydantic_structured(
        fn=agent._materialize_evidence_pack,
        memory=mem,
        model=None,  # no llm
        ignore=["agent"],
    )
    evidence_pack = cached_call(
        agent=agent,
        node_ids=used_node_ids,
        edge_ids=used_edge_ids,
        depth=materialize_depth,
        max_chars_per_item=agent.config.max_chars_per_item,
        max_total_chars=agent.config.max_total_chars,
    )

    evidence_pack_hash = snapshot_hash(evidence_pack)
    evidence_digest = EvidencePackDigest(
        node_ids=list(used_node_ids),
        edge_ids=list(used_edge_ids),
        depth=str(materialize_depth),
        max_chars_per_item=int(agent.config.max_chars_per_item),
        max_total_chars=int(agent.config.max_total_chars),
        evidence_pack_hash=str(evidence_pack_hash),
    ).model_dump(mode="python")

    with ctx.state_write as state:
        state.setdefault("_rt", {})["evidence_pack"] = evidence_pack

    state_update: list[StateUpdate] = [
        ("u", {"evidence_pack_digest": evidence_digest}),
        ("a", {"op_log": "aa_materialize_evidence_pack"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_generate_answer_with_citations")
def _aa_generate_answer_with_citations(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    svc = _chat_service(deps)
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or "")
    run_step_seq = int(sv.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or 0)

    system_prompt = str(sv.get("system_prompt") or "")
    question = str(sv.get("question") or "")
    used_node_ids = list(sv.get("used_node_ids") or [])
    evidence_digest = sv.get("evidence_pack_digest")

    evidence_pack = (sv.get("_rt") or {}).get("evidence_pack")
    if evidence_pack is None:
        re = agent.rehydrate_evidence_pack_from_digest(digest=evidence_digest or {})
        evidence_pack = re.get("evidence_pack")
        with ctx.state_write as state:
            state.setdefault("_rt", {})["evidence_pack"] = evidence_pack

    # Pull view for snapshots.
    view = (sv.get("_rt") or {}).get("view")
    if view is None:
        view = svc.get_conversation_view(
            conversation_id=conversation_id, purpose="answer"
        )
        with ctx.state_write as state:
            state.setdefault("_rt", {})["view"] = view

    from ..utils.pydanic_model_consumer_wrapper import cache_pydantic_structured
    from joblib import Memory

    mem = Memory(
        location=os.path.join(agent.cache_dir, "_generate_answer_with_citations")
    )
    cached_call = cache_pydantic_structured(
        fn=agent._generate_answer_with_citations,
        memory=mem,
        model=AnswerWithCitations,
        ignore=["agent"],
    )

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    agent._persist_context_snapshot(
        conversation_id=conversation_id,
        run_id=run_id,
        run_step_seq=run_step_seq,
        attempt_seq=attempt_seq,
        stage="generate_answer_with_citations",
        view=view,
        model_name=str(
            getattr(
                getattr(deps.get("llm_tasks"), "provider_hints", None),
                "answer_with_citations_provider",
                "",
            )
            or ""
        ),
        budget_tokens=int(getattr(view, "token_budget", 0) or 0),
        tail_turn_index=int(getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0),
        extra_hash_payload={
            "question": question,
            "used_node_ids": used_node_ids,
            "evidence_pack_hash": snapshot_hash(evidence_pack),
        },
        llm_input_payload={
            "system_prompt": system_prompt,
            "question": question,
        },
        evidence_pack_digest=evidence_digest,
    )
    run_step_seq += 1

    ans_json = cached_call(
        agent=agent,
        system_prompt=system_prompt,
        question=question,
        evidence_pack=evidence_pack,
        used_node_ids=used_node_ids,
        out_model_schema=AnswerWithCitations.model_json_schema(),
        out_model=AnswerWithCitations,
    )
    ans = AnswerWithCitations.model_validate(ans_json).model_dump(mode="python")

    state_update: list[StateUpdate] = [
        ("u", {"answer": ans, "run_step_seq": run_step_seq}),
        ("a", {"op_log": "aa_generate_answer_with_citations"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_validate_or_repair_citations")
def _aa_validate_or_repair_citations(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    svc = _chat_service(deps)
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or "")
    run_step_seq = int(sv.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or 0)

    system_prompt = str(sv.get("system_prompt") or "")
    question = str(sv.get("question") or "")
    used_node_ids = list(sv.get("used_node_ids") or [])
    evidence_digest = sv.get("evidence_pack_digest")
    answer = sv.get("answer") or {}

    evidence_pack = (sv.get("_rt") or {}).get("evidence_pack")
    if evidence_pack is None:
        re = agent.rehydrate_evidence_pack_from_digest(digest=evidence_digest or {})
        evidence_pack = re.get("evidence_pack")
        with ctx.state_write as state:
            state.setdefault("_rt", {})["evidence_pack"] = evidence_pack

    view = (sv.get("_rt") or {}).get("view")
    if view is None:
        view = svc.get_conversation_view(
            conversation_id=conversation_id, purpose="answer"
        )
        with ctx.state_write as state:
            state.setdefault("_rt", {})["view"] = view

    from ..utils.pydanic_model_consumer_wrapper import cache_pydantic_structured
    from joblib import Memory

    mem = Memory(
        location=os.path.join(agent.cache_dir, "_generate_answer_with_citations")
    )
    cached_call = cache_pydantic_structured(
        fn=agent._validate_or_repair_citations,
        memory=mem,
        model=AnswerWithCitations,
        ignore=["agent", "answer_in_model"],
    )

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    agent._persist_context_snapshot(
        conversation_id=conversation_id,
        run_id=run_id,
        run_step_seq=run_step_seq,
        attempt_seq=attempt_seq,
        stage="validate_or_repair_citations",
        view=view,
        model_name=str(
            getattr(
                getattr(deps.get("llm_tasks"), "provider_hints", None),
                "repair_citations_provider",
                "",
            )
            or ""
        ),
        budget_tokens=int(getattr(view, "token_budget", 0) or 0),
        tail_turn_index=int(getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0),
        extra_hash_payload={
            "question": question,
            "used_node_ids": used_node_ids,
            "answer_hash": snapshot_hash(answer),
        },
        llm_input_payload={
            "system_prompt": system_prompt,
            "question": question,
            "answer_text": (answer.get("text") if isinstance(answer, dict) else None),
        },
        evidence_pack_digest=evidence_digest,
    )
    run_step_seq += 1

    repaired = cached_call(
        agent=agent,
        system_prompt=system_prompt,
        question=question,
        evidence_pack=evidence_pack,
        used_node_ids=used_node_ids,
        answer=answer,
        answer_in_model=AnswerWithCitations,
    )
    repaired = AnswerWithCitations.model_validate(repaired).model_dump(mode="python")

    state_update: list[StateUpdate] = [
        ("u", {"answer": repaired, "run_step_seq": run_step_seq}),
        ("a", {"op_log": "aa_validate_or_repair_citations"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_evaluate_answer")
def _aa_evaluate_answer(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    svc = _chat_service(deps)
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_id = str(sv.get("run_id") or "")
    run_step_seq = int(sv.get("run_step_seq") or 0)
    attempt_seq = int(sv.get("attempt_seq") or 0)

    system_prompt = str(sv.get("system_prompt") or "")
    question = str(sv.get("question") or "")
    used_node_ids = list(sv.get("used_node_ids") or [])
    answer = sv.get("answer") or {}
    evidence_digest = sv.get("evidence_pack_digest")

    evidence_pack = (sv.get("_rt") or {}).get("evidence_pack")
    if evidence_pack is None:
        re = agent.rehydrate_evidence_pack_from_digest(digest=evidence_digest or {})
        evidence_pack = re.get("evidence_pack")
        with ctx.state_write as state:
            state.setdefault("_rt", {})["evidence_pack"] = evidence_pack

    view = (sv.get("_rt") or {}).get("view")
    if view is None:
        view = svc.get_conversation_view(
            conversation_id=conversation_id, purpose="answer"
        )
        with ctx.state_write as state:
            state.setdefault("_rt", {})["view"] = view

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    agent._persist_context_snapshot(
        conversation_id=conversation_id,
        run_id=run_id,
        run_step_seq=run_step_seq,
        attempt_seq=attempt_seq,
        stage="evaluate_answer",
        view=view,
        model_name=str(
            getattr(
                getattr(deps.get("llm_tasks"), "provider_hints", None),
                "answer_with_citations_provider",
                "",
            )
            or ""
        ),
        budget_tokens=int(getattr(view, "token_budget", 0) or 0),
        tail_turn_index=int(getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0),
        extra_hash_payload={
            "question": question,
            "used_node_ids": used_node_ids,
            "answer_text": (answer.get("text") if isinstance(answer, dict) else ""),
        },
        llm_input_payload={
            "system_prompt": system_prompt,
            "question": question,
            "answer_text": (answer.get("text") if isinstance(answer, dict) else ""),
        },
        evidence_pack_digest=evidence_digest,
    )
    run_step_seq += 1

    eval_json = agent._evaluate_answer(
        agent=agent,
        system_prompt=system_prompt,
        question=question,
        answer_text=str(answer.get("text") if isinstance(answer, dict) else ""),
        used_node_ids=used_node_ids,
        evidence_pack=evidence_pack,
        out_model_schema=AnswerEvaluation.model_json_schema(),
        out_model=AnswerEvaluation,
    )
    evaluation = AnswerEvaluation.model_validate(eval_json).model_dump(mode="python")

    # Escalation knobs applied here (same as answer()).
    max_candidates = int(sv.get("max_candidates") or agent.config.max_candidates)
    materialize_depth = str(
        sv.get("materialize_depth") or agent.config.materialize_depth
    )
    if bool(evaluation.get("needs_more_info")) and not bool(
        evaluation.get("is_sufficient")
    ):
        materialize_depth = "deep"
        max_candidates = min(max_candidates * 2, 200)

    state_update: list[StateUpdate] = [
        (
            "u",
            {
                "evaluation": evaluation,
                "run_step_seq": run_step_seq,
                "max_candidates": max_candidates,
                "materialize_depth": materialize_depth,
            },
        ),
        ("a", {"op_log": "aa_evaluate_answer"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_project_pointers")
def _aa_project_pointers(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    sv = ctx.state_view
    conversation_id = str(sv["conversation_id"])
    run_node_id = str(sv.get("run_node_id") or "")
    used_node_ids = list(sv.get("used_node_ids") or [])
    deps = _deps(ctx)
    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    if prev_turn_meta_summary is None:
        raise Exception("prev_turn_meta_summary missing from dependency")
    projected: list[str] = []
    for kid in used_node_ids:
        pid = agent._project_kg_node(
            conversation_id=conversation_id,
            run_node_id=run_node_id,
            kg_node_id=kid,
            provenance_span=Span.from_dummy_for_conversation(),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
        projected.append(pid)

    state_update: list[StateUpdate] = [
        ("u", {"projected_pointer_ids": projected}),
        ("a", {"op_log": f"aa_project_pointers:{len(projected)}"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_maybe_iterate")
def _aa_maybe_iterate(ctx: StepContext) -> StepRunResult:
    sv = ctx.state_view
    evaluation = sv.get("evaluation") or {}
    iter_idx = int(sv.get("iter_idx") or 0)
    max_iter = int(sv.get("max_iter") or 1)

    needs_more = bool((evaluation or {}).get("needs_more_info"))
    sufficient = bool((evaluation or {}).get("is_sufficient"))

    should_iterate = bool(needs_more and not sufficient and (iter_idx + 1 < max_iter))
    next_iter = (iter_idx + 1) if should_iterate else iter_idx

    state_update: list[StateUpdate] = [
        ("u", {"should_iterate": should_iterate, "iter_idx": next_iter}),
        ("a", {"op_log": f"aa_maybe_iterate:{should_iterate}"}),
    ]
    return RunSuccess(conversation_node_id=None, state_update=state_update)


@default_resolver.register("aa_persist_response")
def _aa_persist_response(ctx: StepContext) -> StepRunResult:
    agent = _aa_agent(ctx)
    deps = _deps(ctx)
    sv = ctx.state_view

    conversation_id = str(sv["conversation_id"])
    run_node_id = str(sv.get("run_node_id") or "")
    used_node_ids = list(sv.get("used_node_ids") or [])
    projected_pointer_ids = list(sv.get("projected_pointer_ids") or [])
    answer = sv.get("answer") or {}
    evaluation = sv.get("evaluation") or {}

    prev_turn_meta_summary = deps.get("prev_turn_meta_summary")
    tail_turn_index = int(getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0)

    assistant_text = str(answer.get("text") or "")
    assistant_turn_node_id, assistant_turn_node = agent._add_assistant_turn(
        conversation_id=conversation_id,
        content=assistant_text,
        provenance_span=Span.from_dummy_for_conversation(),
        turn_index=tail_turn_index + 1,
        prev_turn_meta_summary=prev_turn_meta_summary,
    )
    agent._link_run_to_response(
        conversation_id=conversation_id,
        run_node_id=run_node_id,
        response_node_id=assistant_turn_node_id,
        used_node_ids=used_node_ids,
        provenance_span=Span.from_dummy_for_conversation(),
        prev_turn_meta_summary=prev_turn_meta_summary,
    )
    prev_turn_meta_summary.tail_turn_index += 1
    # surface prev_turn_meta_summary as jsonable for downstream callers.
    mts = {
        "prev_node_char_distance_from_last_summary": int(
            getattr(
                prev_turn_meta_summary, "prev_node_char_distance_from_last_summary", 0
            )
            or 0
        ),
        "prev_node_distance_from_last_summary": int(
            getattr(prev_turn_meta_summary, "prev_node_distance_from_last_summary", 0)
            or 0
        ),
        "tail_turn_index": int(
            getattr(prev_turn_meta_summary, "tail_turn_index", 0) or 0
        ),
    }
    result_payload = {
        "run_node_id": run_node_id,
        "assistant_turn_node_id": assistant_turn_node_id,
        "assistant_text": assistant_text,
        "used_node_ids": used_node_ids,
        "projected_pointer_ids": projected_pointer_ids,
        "claim_citations": answer,
        "evaluation": evaluation,
        "turn_node_dump": assistant_turn_node.model_dump()
        if hasattr(assistant_turn_node, "model_dump")
        else {},
    }

    state_update: list[StateUpdate] = [
        (
            "u",
            {"agentic_answering_result": result_payload, "prev_turn_meta_summary": mts},
        ),
        ("a", {"op_log": "aa_persist_response"}),
    ]
    return RunSuccess(
        conversation_node_id=assistant_turn_node_id, state_update=state_update
    )
