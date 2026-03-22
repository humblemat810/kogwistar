"""Conversation orchestrator (KGE-native, LangGraph-free).

This module serves as the **workflow entry point** for conversation interactions.
It owns *policy + control flow* for conversation turns, while delegating
all persistence/mutation to the conversation engine.

It is intentionally lightweight and uses your existing retrievers/agents.
"""

from __future__ import annotations


from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, cast

from kogwistar.llm_tasks import LLMTaskSet, SummarizeContextTaskRequest

from .models import ConversationEdge, MetaFromLastSummary
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.id_provider import stable_id
from .conversation_state_contracts import (
    WorkflowStateModel,
    PrevTurnMetaSummaryModel,
    ConversationWorkflowState,
)
from ..engine_core.models import Grounding, MentionVerification, Span, Role
from ..utils.embedding_vectors import normalize_embedding_vector
from .models import (
    ConversationAIResponse,
    ConversationNode,
    KnowledgeRetrievalResult,
    AddTurnResult,
)
from .tool_runner import ToolRunner
from .memory_retriever import MemoryRetriever, MemoryPinResult, MemoryRetrievalResult
from .knowledge_retriever import KnowledgeRetriever
from .policy import get_chat_tail, last_summary_of_node


@dataclass
class OrchestratorState:
    conversation_id: str
    user_id: str
    mem_id: str
    user_text: str
    role: str
    ref_knowledge_engine: Any

    # created turn
    current_turn_node_id: str | None = None
    turn_index: int | None = None
    self_span: Span | None = None
    embedding: List[float] | None = None

    # retrieval artifacts
    memory: MemoryRetrievalResult | None = None
    knowledge: KnowledgeRetrievalResult | None = None
    memory_pin: MemoryPinResult | None = None
    pinned_kg_pointer_node_ids: List[str] = field(default_factory=list)
    pinned_kg_edge_ids: List[str] = field(default_factory=list)

    # answer
    answer: ConversationAIResponse | None = None


@dataclass
class ExecClock:
    """Execution clock for append-only conversation runs.

    - run_id: identifies an execution attempt scope (defaults to conversation_id).
    - run_step_seq: coarse monotonic step counter within a run.
    - attempt_seq: retry counter for the same step (append-only retries).

    This stays in conversation_orchestrator.py (no workflow primitive dependency).
    """

    run_id: str
    run_step_seq: int
    attempt_seq: int = 0

    def bump_step(self) -> "ExecClock":
        return replace(self, run_step_seq=self.run_step_seq + 1, attempt_seq=0)

    def bump_attempt(self) -> "ExecClock":
        return replace(self, attempt_seq=self.attempt_seq + 1)


def _infer_prev_run_step_seq(
    conversation_engine: GraphKnowledgeEngine, conversation_id: str
) -> int:
    """Best-effort: derive last run_step_seq from conversation tail node metadata."""
    try:
        tail = _get_conversation_tail_compat(conversation_engine, conversation_id)
    except Exception:
        tail = None
    if tail is None:
        return 0
    try:
        return int((tail.metadata or {}).get("run_step_seq") or 0)
    except Exception:
        return 0


def _stamp_meta(
    meta: dict | None, *, run_id: str, run_step_seq: int, attempt_seq: int
) -> dict:
    """Stamp run fields into metadata without overwriting existing values."""
    out = dict(meta or {})
    out.setdefault("run_id", run_id)
    out.setdefault("run_step_seq", run_step_seq)
    out.setdefault("attempt_seq", attempt_seq)
    return out


def _estimate_tokens_from_chars(
    char_count: int, token_estimator: Callable[[str], int] | None = None
) -> int:
    """Estimate token count from character count.

    Default proxy is ~4 chars/token (deterministic + cheap).
    If token_estimator is provided, estimate on a bounded sample and scale linearly.
    """
    if char_count <= 0:
        return 0
    if token_estimator is None:
        return max(1, (char_count + 3) // 4)

    sample_n = min(4096, char_count)
    sample = "a" * sample_n
    try:
        sample_tokens = int(token_estimator(sample))
    except Exception:
        return max(1, (char_count + 3) // 4)
    if sample_tokens <= 0:
        return max(1, (char_count + 3) // 4)
    return max(1, int(round(sample_tokens * (char_count / sample_n))))


def _get_conversation_tail_compat(conversation_engine: Any, conversation_id: str):
    try:
        return get_chat_tail(conversation_engine, conversation_id=conversation_id)
    except Exception:
        return None


def _iterative_emb_compat(engine: Any, text: str):
    embed_ns = getattr(engine, "embed", None)
    if embed_ns is not None and hasattr(embed_ns, "iterative_defensive_emb"):
        return normalize_embedding_vector(
            embed_ns.iterative_defensive_emb(text), allow_none=False
        )
    if hasattr(engine, "iterative_defensive_emb"):
        return normalize_embedding_vector(
            engine.iterative_defensive_emb(text), allow_none=False
        )
    if hasattr(engine, "_iterative_defensive_emb"):
        return normalize_embedding_vector(
            engine._iterative_defensive_emb(text), allow_none=False
        )
    raise AttributeError("conversation engine has no defensive embedding API")


from .models import RetrievalResult


def get_id_for_conversation_turn(
    id_kind, user_id, conversation_id, content, new_index, role, entity_type, in_conv
):
    return str(
        stable_id(
            id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            role,
            entity_type,
            str(in_conv),
        )
    )


def get_id_for_conversation_turn_edge(
    id_kind,
    user_id,
    conversation_id,
    content,
    new_index,
    source_ids,
    target_ids,
    source_edge_ids,
    target_edge_ids,
    entity_type,
):
    return str(
        stable_id(
            id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            str(source_ids),
            str(target_ids),
            str(source_edge_ids),
            str(target_edge_ids),
            entity_type,
        )
    )


# ---------------------------------------------------------------------------
# Conversation-specific step resolver
# ---------------------------------------------------------------------------
from .resolvers import MappingStepResolver, default_resolver


class ConversationStepResolver(MappingStepResolver):
    """Conversation-specific resolver wrapper.

    Step implementations live in `workflow/resolvers.py` (default_resolver).
    Conversation orchestration should only "declare/select" ops via this subclass.
    """

    def __init__(self) -> None:
        super().__init__(
            handlers=dict(default_resolver.handlers), default=default_resolver.default
        )


class ConversationOrchestrator:
    """KGE-native orchestrator and workflow entry point.

    This class acts as the central coordinator for the conversation workflow.
    It orchestrates the flow of adding turns (`add_conversation_turn_workflow_v2`),
    retrieving context (memory/knowledge), and generating responses.

    Simple orchestrator that bind one knowledge base and one conversation.

    Use workflow primitive if each step may bind to a different knowledge base,
    or use a knowledge router that has same api but delegate to get knowledge from network of
    knowledge bases

    Service-first boundary:
    - ConversationService is the public entry point
    - orchestration steps live here
    - tool-call/tool-result events are recorded in the conversation graph
    """

    def create_conversation(self, *args, **kwargs):
        from .service import ConversationService

        svc = ConversationService.from_engine(
            self.conversation_engine,
            knowledge_engine=self.ref_knowledge_engine,
            workflow_engine=self.workflow_engine,
        )
        return svc.create_conversation(*args, **kwargs)

    # ----------------------------
    # Workflow-v2: design + step resolver
    # ----------------------------

    def _ensure_add_turn_workflow_design(
        self, workflow_id: str, conversation_id: str
    ) -> None:
        """Ensure a workflow design exists in workflow_engine for legacy add_turn behavior.

        This writes *design nodes/edges* into workflow_engine (kg_graph_type="workflow").
        The design mirrors ConversationOrchestrator.add_conversation_turn:
          - memory_retrieve
          - kg_retrieve
          - pin memory (conditional)
          - pin kg (conditional)
          - answer
          - decide summarize
          - summarize (conditional)
          - end

        If the design is already present, this is a no-op.
        """
        if self.workflow_engine is None:
            raise RuntimeError("workflow_engine must be provided for workflow_v2")

        from .designer import ConversationWorkflowDesigner

        designer = ConversationWorkflowDesigner(
            workflow_engine=self.workflow_engine,
            predicate_registry={
                # Dummy predicates to satisfy design-time validation; runtime supplies real logic.
                "always": lambda st, r: True,
                "in_conv": lambda st, r: bool(st.get("in_conv", True)),
                "should_pin_memory": lambda st, r: True,
                "should_pin_kg": lambda st, r: True,
                "should_summarize": lambda st, r: True,
            },
        )

        # We do not validate here because orchestrator owns the predicate registry.
        # If a stale/broken design exists, runtime validation will raise with the real registry.
        try:
            designer.ensure_add_turn_flow(
                workflow_id=workflow_id, mode="full", include_context_snapshot=True
            )
        except Exception as _e:
            raise
            # If predicate_registry is required by validate() inside ensure_add_turn_flow(),
            # we still want to create the design. The designer will re-validate when runtime runs.
            designer.ensure_add_turn_flow(
                workflow_id=workflow_id, mode="full", include_context_snapshot=True
            )

    def add_turn_only_workflow(
        self,
        *,
        run_id: str,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: Role,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        workflow_id: str,
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        summary_token_threshold: int | None = None,
        summary_turn_threshold=5,
        token_estimator: Callable[[str], int] | None = None,
        in_conv: bool = True,
        prev_turn_meta_summary: MetaFromLastSummary | None = None,
        add_turn_only=None,
        max_workers: int = 4,
        strict_answer_failure: bool = False,
        force_answer_only: bool | None = None,
    ) -> AddTurnResult:

        prev_node = (
            _get_conversation_tail_compat(self.conversation_engine, conversation_id)
            if in_conv
            else None
        )
        prev_turn_meta_summary, new_index = (
            self.ensure_prev_turn_meta_summary_new_index(
                prev_node, prev_turn_meta_summary
            )
        )

        from .designer import ConversationWorkflowDesigner

        predicate_registry = {"always": lambda st, r: True}
        designer = ConversationWorkflowDesigner(
            workflow_engine=self.workflow_engine, predicate_registry=predicate_registry
        )
        # Use a deterministic backbone workflow_id derived from caller workflow_id.
        backbone_wid = f"{workflow_id}::backbone"
        designer.ensure_backbone(workflow_id=backbone_wid)

        # Create the user turn node + next_turn link (same as v1) before running the skeleton workflow.
        prev_node = (
            _get_conversation_tail_compat(self.conversation_engine, conversation_id)
            if in_conv
            else None
        )
        if prev_node is not None:
            new_index = (
                (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0
            )
        else:
            new_index = 0

        turn_node_id = get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            str(role),
            "conversation_turn",
            in_conv=True,
        )
        self_span = Span.from_dummy_for_conversation()
        self_span.doc_id = turn_node_id
        self_span.page_number = 1
        embedding = _iterative_emb_compat(self.conversation_engine, content)

        # Phase F: do not persist the user turn node or chain edge here.

        # The backbone workflow (mode=backbone) will run add_user_turn.

        # Run minimal workflow skeleton (for trace/state plumbing) - no retrieval/answer steps.
        from ..runtime.runtime import WorkflowRuntime

        runtime = WorkflowRuntime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            step_resolver=ConversationStepResolver(),
            predicate_registry=predicate_registry,
            checkpoint_every_n_steps=1,
            max_workers=max_workers,
        )

        init_state: ConversationWorkflowState = cast(
            ConversationWorkflowState,
            WorkflowStateModel(
                conversation_id=conversation_id,
                user_id=user_id,
                turn_node_id=turn_node_id,
                turn_index=new_index,
                mem_id=mem_id,
                self_span=self_span,
                role=str(role),
                user_text=content,
                embedding=embedding,
                prev_turn_meta_summary=PrevTurnMetaSummaryModel(
                    prev_node_char_distance_from_last_summary=prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    prev_node_distance_from_last_summary=prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    tail_turn_index=prev_turn_meta_summary.tail_turn_index,
                ),
                _deps={
                    "conversation_engine": self.conversation_engine,
                    "ref_knowledge_engine": self.ref_knowledge_engine,
                    "llm_tasks": self.llm_tasks,
                    "filtering_callback": filtering_callback,
                },
            ).model_dump(),
        )
        prev_turn_meta_summary.tail_turn_index += 1
        runtime.run(
            workflow_id=backbone_wid,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=init_state,
            run_id=f"add_turn_backbone|{turn_node_id}",
        )

        return AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=None,
            turn_index=new_index,
            relevant_kg_node_ids=[],
            relevant_kg_edge_ids=[],
            pinned_kg_pointer_node_ids=[],
            pinned_kg_edge_ids=[],
            memory_context_node_id=None,
            memory_context_edge_ids=[],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    def ensure_prev_turn_meta_summary_new_index(
        self, prev_node, prev_turn_meta_summary
    ):
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary(0, 0)
        if prev_node is not None:
            new_index = (
                (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0
            )

            node_last_char_dist = prev_node.metadata.get(
                "char_distance_from_last_summary"
            )
            if node_last_char_dist is not None:
                node_last_char_dist += len(prev_node.summary)
            else:
                node_last_char_dist = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = (
                node_last_char_dist
            )

            node_last_turn_dist = prev_node.metadata.get(
                "turn_distance_from_last_summary"
            )
            if node_last_turn_dist is not None:
                node_last_turn_dist += 1
            else:
                node_last_turn_dist = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = (
                node_last_turn_dist
            )

            node_tail_turn_index = prev_node.metadata.get("tail_turn_index")
            if node_tail_turn_index is not None:
                node_tail_turn_index += 1
            else:
                node_tail_turn_index = -1  # start_node
            prev_turn_meta_summary.tail_turn_index = node_tail_turn_index
        else:
            new_index = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = -1
            prev_turn_meta_summary.tail_turn_index = -1  # start node
        return prev_turn_meta_summary, new_index

    def add_conversation_turn_workflow_v2(
        self,
        *,
        run_id: str,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: Role,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        workflow_id: str,
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        summary_token_threshold: int | None = None,
        summary_turn_threshold=5,
        token_estimator: Callable[[str], int] | None = None,
        in_conv: bool = True,
        prev_turn_meta_summary: MetaFromLastSummary | None = None,
        add_turn_only=None,
        max_workers: int = 4,
        strict_answer_failure: bool = False,
        force_answer_only: bool | None = None,
        cache_dir = None,
    ) -> AddTurnResult:
        """
        V2 workflow-driven orchestration.

        Design engine: workflow_engine (kg_graph_type="workflow")
        Trace/checkpoints: conversation_engine (kg_graph_type="conversation")
        """
        if self.workflow_engine is None:
            raise RuntimeError("workflow_engine must be provided for workflow_v2")

        # designer and conversation have their own separate clock
        designer_clock = ExecClock(run_id=conversation_id, run_step_seq=0)

        # ----------------------------
        # 0) Compute prior-summary distances (mirror legacy)
        # ----------------------------

        # ----------------------------
        # 0b) add_turn_only: only create the user turn backbone and run a minimal workflow skeleton.
        # ----------------------------

        prev_node = (
            _get_conversation_tail_compat(self.conversation_engine, conversation_id)
            if in_conv
            else None
        )
        prev_turn_meta_summary, new_index = (
            self.ensure_prev_turn_meta_summary_new_index(
                prev_node, prev_turn_meta_summary
            )
        )
        if add_turn_only:
            return self.add_turn_only_workflow(
                user_id=user_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                mem_id=mem_id,
                role=role,
                run_id=run_id,
                workflow_id=workflow_id,
                content=content,
                filtering_callback=filtering_callback,
                max_retrieval_level=max_retrieval_level,
                summary_char_threshold=summary_char_threshold,
                summary_token_threshold=summary_token_threshold,
                summary_turn_threshold=summary_turn_threshold,
                token_estimator=token_estimator,
                in_conv=in_conv,
                prev_turn_meta_summary=prev_turn_meta_summary,
                add_turn_only=add_turn_only,
            )

            # ----------------------------
        # 1) Prepare the user turn identifiers (NO persistence here)
        # ----------------------------
        # Phase F: orchestrator does not mutate the conversation graph.
        # Node/edge creation happens inside workflow ops: add_user_turn + link_prev_turn.
        emb_text0 = f"{role}: {content}"
        embedding = _iterative_emb_compat(self.conversation_engine, emb_text0)
        if embedding is None:
            raise RuntimeError("uncalculatable embeddings")

        turn_node_id = get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            str(role),
            "conversation_turn",
            str(bool(in_conv)),
        )

        self_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="conversation_turn",
            page_number=1,
            start_char=0,
            end_char=len(content),
            excerpt=content,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="human", is_verified=True, score=1.0, notes="verbatim input"
            ),
        )
        # ----------------------------
        # 2) Ensure the workflow design exists
        # ----------------------------
        self._ensure_add_turn_workflow_design(workflow_id, conversation_id)

        # ----------------------------
        # 3) Predicates (dynamic routing)
        # ----------------------------
        def always(workflow_info, st, r):
            return True

        def should_pin_memory(workflow_info, st, r):
            return bool((st.get("memory") or {}).get("selected")) and bool(
                (st.get("memory") or {}).get("memory_context_text")
            )

        def should_pin_kg(workflow_info, st, r):
            return bool((st.get("kg") or {}).get("selected"))

        def should_summarize(workflow_info, st, r):
            return bool(st.get("summary", {}).get("should_summarize", False))

        predicate_registry = {
            "always": always,
            "should_pin_memory": should_pin_memory,
            "should_pin_kg": should_pin_kg,
            "should_summarize": should_summarize,
        }

        # ----------------------------
        # 4) Step resolver
        # ----------------------------
        # Use the package-default resolver registry (resolvers.py).
        # Step implementations pull dependencies from ctx.state["_deps"].
        resolve_step = ConversationStepResolver()

        # ----------------------------
        # 5) Run with WorkflowRuntime (real persisted checkpoints)
        # ----------------------------
        from ..runtime.runtime import WorkflowRuntime

        runtime = WorkflowRuntime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            step_resolver=resolve_step,
            predicate_registry=predicate_registry,
            checkpoint_every_n_steps=1,
            max_workers=1,
        )
        uses_custom_answer_only = force_answer_only
        if uses_custom_answer_only is None:
            bound_answer_only = getattr(self.answer_only, "__func__", None)
            uses_custom_answer_only = (
                bound_answer_only is None
                or bound_answer_only is not ConversationOrchestrator.answer_only
            )
        def cacheable_summarize_batch(conversation_id, current_index, *, prev_turn_meta_summary, cache_dir):
            return self._summarize_conversation_batch(
                    conversation_id,
                    current_index,
                    prev_turn_meta_summary=prev_turn_meta_summary,
                    cache_dir = cache_dir
                )
        def cacheable_answer_only(*, conversation_id, prev_turn_meta_summary, cache_dir):
            return self.answer_only(
                    conversation_id=conversation_id,
                    prev_turn_meta_summary=prev_turn_meta_summary,
                    cache_dir=cache_dir
                )
        deps = {
            "conversation_engine": self.conversation_engine,
            "ref_knowledge_engine": self.ref_knowledge_engine,
            "llm_tasks": self.llm_tasks,
            "filtering_callback": filtering_callback,
            "tool_runner": self.tool_runner,
            "max_retrieval_level": max_retrieval_level,
            "summary_char_threshold": summary_char_threshold,
            "prev_turn_meta_summary": prev_turn_meta_summary,
            "agent": self.agentic_answering_agent,
            "workflow_engine": self.workflow_engine,
            "agentic_workflow_engine": self.workflow_engine,
            "summary_token_threshold": summary_token_threshold,
            "summary_turn_threshold": summary_turn_threshold,
            "token_estimator": token_estimator,
            "strict_answer_failure": bool(strict_answer_failure),
            "force_answer_only": bool(uses_custom_answer_only),
            "answer_only": cacheable_answer_only,#lambda *, conversation_id, prev_turn_meta_summary, cache_dir: (
                # self.answer_only(
                #     conversation_id=conversation_id,
                #     prev_turn_meta_summary=prev_turn_meta_summary,
                #     cache_dir=cache_dir
                # )
            # ),
            "summarize_batch":  cacheable_summarize_batch,#lambda conversation_id, current_index, *, prev_turn_meta_summary, cache_dir: (
            #     self._summarize_conversation_batch(
            #         conversation_id,
            #         current_index,
            #         prev_turn_meta_summary=prev_turn_meta_summary,
            #         cache_dir = cache_dir
            #     )
            # ),
            "add_link_to_new_turn": self.add_link_to_new_turn,
        }
        init_state: ConversationWorkflowState = cast(
            ConversationWorkflowState,
            WorkflowStateModel(
                conversation_id=conversation_id,
                user_id=user_id,
                turn_node_id=turn_node_id,
                turn_index=new_index,
                mem_id=mem_id,
                self_span=self_span,
                role=str(role),
                user_text=content,
                embedding=embedding,
                prev_turn_meta_summary=PrevTurnMetaSummaryModel(
                    prev_node_char_distance_from_last_summary=prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    prev_node_distance_from_last_summary=prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    tail_turn_index=prev_turn_meta_summary.tail_turn_index,
                ),
            ).model_dump(),
        )
        # Dependency injection for workflow step resolvers (workflow/resolvers.py)
        init_state["_deps"] = deps
        init_state["in_conv"] = bool(in_conv)
        if prev_node is not None:
            init_state["prev_turn_id"] = str(prev_node.id)

        run_result = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=init_state,
            run_id=f"add_turn|{turn_node_id}",
            cache_dir = cache_dir,
        )
        final_state, run_id = run_result.final_state, run_result.run_id
        if str(getattr(run_result, "status", "")) != "succeeded":
            raise RuntimeError(
                f"Workflow add-turn run failed with status={getattr(run_result, 'status', '')!r}"
            )
        # ----------------------------
        # 6) Map final state to legacy AddTurnResult
        # ----------------------------
        mem = final_state.get("memory") or {}
        kg = final_state.get("kg") or {}
        mem_pin = final_state.get("memory_pin") or {}
        kg_pin = final_state.get("kg_pin") or {}
        answer = final_state.get("answer") or {}

        # update prev_turn_meta_summary in-place so callers can chain it
        mts = final_state.get("prev_turn_meta_summary") or {}
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = int(
            mts.get(
                "prev_node_char_distance_from_last_summary",
                prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
            )
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary = int(
            mts.get(
                "prev_node_distance_from_last_summary",
                prev_turn_meta_summary.prev_node_distance_from_last_summary,
            )
        )
        prev_turn_meta_summary.tail_turn_index = int(
            mts.get("tail_turn_index", prev_turn_meta_summary.tail_turn_index)
        )

        result = AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=answer.get("response_node_id"),
            turn_index=new_index,
            relevant_kg_node_ids=list((kg.get("selected") or {}).get("node_ids") or []),
            relevant_kg_edge_ids=list((kg.get("selected") or {}).get("edge_ids") or []),
            pinned_kg_pointer_node_ids=list(
                kg_pin.get("pinned_pointer_node_ids") or []
            ),
            pinned_kg_edge_ids=list(kg_pin.get("pinned_edge_ids") or []),
            memory_context_node_id=mem_pin.get("memory_context_node_id"),
            memory_context_edge_ids=list(mem_pin.get("pinned_edge_ids") or []),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
        if strict_answer_failure and not result.response_turn_node_id:
            raise RuntimeError(
                "Workflow add-turn completed without materializing an assistant response node."
            )
        return result

    def __init__(
        self,
        *,
        conversation_engine: GraphKnowledgeEngine,
        ref_knowledge_engine: GraphKnowledgeEngine,
        workflow_engine: GraphKnowledgeEngine | None = None,
        tool_call_id_factory: Callable | None = None,
        llm_tasks: LLMTaskSet | None = None,
        agent_cache_dir: str | None = None,
    ) -> None:
        tool_call_id_factory = (
            tool_call_id_factory or conversation_engine.tool_call_id_factory
        )
        if tool_call_id_factory is None:
            raise Exception("missing tool_call_id_factory")
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.ref_knowledge_engine: GraphKnowledgeEngine = ref_knowledge_engine
        self.workflow_engine: GraphKnowledgeEngine | None = workflow_engine
        self.llm_tasks: LLMTaskSet = llm_tasks or conversation_engine.llm_tasks
        self.tool_runner = ToolRunner(
            conversation_engine=conversation_engine,
            tool_call_id_factory=tool_call_id_factory,
        )

        # Agentic answering agent (used by workflow op 'answer' when provided).
        try:
            from .conversation.agentic_answering import AgenticAnsweringAgent

            self.agentic_answering_agent = AgenticAnsweringAgent(
                conversation_engine=self.conversation_engine,
                knowledge_engine=self.ref_knowledge_engine,
                llm_tasks=self.llm_tasks,
                cache_dir=agent_cache_dir,
            )
        except Exception:
            self.agentic_answering_agent = None

        self.tail_search_includes = ["conversation_turn", "conversation_summary"]

    def add_link_to_new_turn(
        self,
        edge_id,
        turn_node,
        prev_node,
        conversation_id,
        span,
        prev_turn_meta_summary: MetaFromLastSummary,
        causal_type: str | None = "chain",
        clock: ExecClock | None = None,
    ):

        # eid = stable_id("edge")

        meta = {
            "relation": "next_turn",
            "target_id": turn_node.id,
            "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
            "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
            "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
            **({"causal_type": causal_type} if causal_type else {}),
        }
        if clock is not None:
            meta = _stamp_meta(
                meta,
                run_id=clock.run_id,
                run_step_seq=clock.run_step_seq,
                attempt_seq=clock.attempt_seq,
            )

        seq_edge = ConversationEdge(
            id=edge_id,
            source_ids=[prev_node.id],
            target_ids=[turn_node.id],
            relation="next_turn",
            label="next_turn",
            type="relationship",
            summary="Sequential flow",
            doc_id=f"conv:{conversation_id}",
            mentions=[Grounding(spans=[span])],
            domain_id=None,
            canonical_entity_id=None,
            properties={"entity_type": "conversation_edge"},
            embedding=None,
            metadata=meta,
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.write.add_edge(seq_edge)
        return seq_edge

    # ----------------------------
    # Public entry points
    # ----------------------------
    def add_conversation_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: Role,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        summary_token_threshold: int | None = None,
        summary_turn_threshold=5,
        token_estimator: Callable[[str], int] | None = None,
        in_conv: bool = True,
        prev_turn_meta_summary: MetaFromLastSummary | None = None,
        add_turn_only=None,
    ) -> AddTurnResult:
        """Ingest a user/assistant turn and run retrieval+answering policy."""

        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        if self.ref_knowledge_engine is None:
            raise Exception("GraphKnowledgeEngine must be specified")

        st = OrchestratorState(
            conversation_id=conversation_id,
            user_id=user_id,
            mem_id=mem_id,
            user_text=content,
            role=role,
            ref_knowledge_engine=self.ref_knowledge_engine,
        )
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary(0, 0)
        _prev_step = _infer_prev_run_step_seq(self.conversation_engine, conversation_id)
        clock = ExecClock(run_id=conversation_id, run_step_seq=_prev_step).bump_step()
        prev_node = _get_conversation_tail_compat(
            self.conversation_engine, conversation_id
        )
        if prev_node is not None:
            new_index = (
                (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0
            )
            # prev_turn_meta_summary.prev_node_char_distance_from_last_summary = prev_node.metadata.get("char_distance_from_last_summary") or 0
            node_last_char_dist = prev_node.metadata.get(
                "char_distance_from_last_summary"
            )
            if node_last_char_dist is not None:
                node_last_char_dist += len(prev_node.summary)
            else:
                node_last_char_dist = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = (
                node_last_char_dist
            )
            # ======================
            node_last_turn_dist = prev_node.metadata.get(
                "turn_distance_from_last_summary"
            )
            if node_last_turn_dist is not None:
                node_last_turn_dist += 1
            else:
                node_last_turn_dist = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = node_last_turn_dist  # prev_node.metadata.get("turn_distance_from_last_summary") or 0
            # ======================
            node_tail_turn_index = prev_node.metadata.get("tail_turn_index")
            if node_tail_turn_index is not None:
                node_tail_turn_index += 1
            else:
                node_tail_turn_index = -1  # start node
            prev_turn_meta_summary.tail_turn_index = node_tail_turn_index
        else:
            new_index = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = -1
            prev_turn_meta_summary.tail_turn_index = -1  # start node

        turn_node_id = turn_id or get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            role,
            "conversation_turn",
            str(in_conv),
        )
        self_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="conversation_turn",
            page_number=1,
            start_char=0,
            end_char=len(content),
            excerpt=content,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system",
                is_verified=True,
                score=1.0,
                notes=f"verbatim {role} input",
            ),
        )

        turn_node = ConversationNode(
            user_id=user_id,
            id=turn_node_id,
            label=f"Turn {new_index} ({role})",
            type="entity",
            doc_id=turn_node_id,
            summary=content,
            role=role,  # type: ignore
            turn_index=new_index,
            conversation_id=conversation_id,
            mentions=[Grounding(spans=[self_span])],
            properties={},
            metadata=_stamp_meta(
                {
                    "entity_type": "conversation_turn",
                    "turn_index": new_index,
                    "level_from_root": 0,
                    "in_conversation_chain": in_conv,
                    "in_ui_chain": True,
                },
                run_id=clock.run_id,
                run_step_seq=clock.run_step_seq,
                attempt_seq=clock.attempt_seq,
            ),
            domain_id=None,
            canonical_entity_id=None,
        )
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        prev_turn_meta_summary.tail_turn_index = new_index
        new_index += 1
        emb_text0 = f"{role}: {content}"
        embedding = _iterative_emb_compat(self.conversation_engine, emb_text0)
        if embedding is None:
            raise Exception("uncalculatable embeddings")
        turn_node.embedding = embedding

        self.conversation_engine.write.add_node(turn_node, None)
        if prev_node:
            seq_edge_id = get_id_for_conversation_turn_edge(
                ConversationEdge.id_kind,
                user_id,
                conversation_id,
                "next_turn",
                new_index,
                [prev_node.id],
                [turn_node.id],
                [],
                [],
                "conversation_edge",
            )
            seq_edge = self.add_link_to_new_turn(
                seq_edge_id,
                turn_node,
                prev_node,
                conversation_id,
                span=self_span,
                prev_turn_meta_summary=prev_turn_meta_summary,
                causal_type="chain",
                clock=clock,
            )
        prev_node = turn_node
        st.current_turn_node_id = turn_node_id
        st.turn_index = new_index
        st.self_span = self_span
        st.embedding = embedding
        add_turn_result = AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=None,  # response_turn_node_id,
            turn_index=new_index,
            relevant_kg_node_ids=[],
            relevant_kg_edge_ids=[],
            pinned_kg_pointer_node_ids=[],
            pinned_kg_edge_ids=[],
            memory_context_node_id=None,
            memory_context_edge_ids=[],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

        if role in ["assistent", "system"]:
            if add_turn_only is None:
                add_turn_only = True
        else:
            if (
                add_turn_only is None
            ):  # default human turn will trigger machine response
                add_turn_only = False

        if not add_turn_only:
            add_turn_result = self.gen_machine_response_turns(
                user_id,
                conversation_id,
                turn_node_id,
                new_index,
                embedding,
                content,
                filtering_callback,
                max_retrieval_level,
                st,
                mem_id,
                self_span,
                prev_node,
                summary_char_threshold,
                prev_turn_meta_summary=prev_turn_meta_summary,
                summary_token_threshold=summary_token_threshold,
                summary_turn_threshold=summary_turn_threshold,
                token_estimator=token_estimator,
            )
        return add_turn_result

    def join_tool_node_to_turn(
        self, conversation_id, node_id, turn_node_id, prev_turn_meta_summary
    ):
        eid = str(stable_id("tool_call_entry", conversation_id, node_id, turn_node_id))
        edge = ConversationEdge(
            id=eid,
            source_ids=[node_id],
            target_ids=[turn_node_id],
            relation="tool_call_entry_point",
            label=f"tool_call_entry_point:{node_id}:{turn_node_id}",
            type="relationship",
            summary="tool_call_entry_point:{node_id}:{turn_node_id}",
            doc_id=f"conv:{conversation_id}",
            domain_id=None,
            canonical_entity_id=None,
            properties=None,
            embedding=None,
            mentions=[
                Grounding(
                    spans=[
                        Span(
                            collection_page_url=f"conversation/{conversation_id}",
                            document_page_url=f"conversation/{conversation_id}",
                            doc_id=f"conv:{conversation_id}",
                            chunk_id=None,
                            source_cluster_id=None,
                            insertion_method="tool_call",
                            page_number=1,
                            start_char=0,
                            end_char=1,
                            excerpt="tool_call entry point",
                            context_before="",
                            context_after="",
                            verification=MentionVerification(
                                method="system",
                                is_verified=True,
                                score=1.0,
                                notes="link",
                            ),
                        )
                    ]
                )
            ],
            metadata={
                "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
            },
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.write.add_edge(edge)

    def gen_machine_response_turns(
        self,
        user_id,
        conversation_id,
        turn_node_id,
        new_index,  # user visible new turn index
        embedding,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        max_retrieval_level: int,
        st: OrchestratorState,
        mem_id: str,
        self_span: Span,
        prev_node: ConversationNode | None,
        summary_char_threshold: int,
        prev_turn_meta_summary: MetaFromLastSummary,
        summary_turn_threshold: int = 5,
        summary_token_threshold: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
        cache_dir = None,
    ):
        """Run retrieval, pinning, answer generation, and optional summarization for one turn.

        The user turn is assumed to already exist. This method orchestrates
        memory retrieval, KG retrieval, pinning side effects, answer
        persistence, next-turn edge creation, and summary triggering in one
        stateful pass. prev_turn_meta_summary remains the sequencing source of
        truth for chain edges and summary thresholds.
        """
        # 3) Retrieval + pinning (recorded as tool events)
        mem_retriever = MemoryRetriever(
            conversation_engine=self.conversation_engine,
            llm_tasks=self.llm_tasks,
            filtering_callback=filtering_callback,
        )

        kg_retriever = KnowledgeRetriever(
            conversation_engine=self.conversation_engine,
            ref_knowledge_engine=self.ref_knowledge_engine,
            llm_tasks=self.llm_tasks,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
        )
        mem_args = dict(
            user_id=user_id,
            current_conversation_id=conversation_id,
            query_embedding=embedding,
            user_text=content,
            context_text="",
            n_results=12,
        )
        # memory retrieve tool
        mem: MemoryRetrievalResult
        mem, call_node = self.tool_runner.run_tool(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=new_index,
            tool_name="memory_retrieve",
            args=[],
            kwargs=mem_args,
            handler=mem_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
            prev_node=prev_node,
            orchestrator=self,
        )
        # self.join_tool_node_to_turn(conversation_id, call_node, turn_node_id, prev_turn_meta_summary)
        st.memory = mem
        kg_args = dict(
            user_text=content,
            context_text="",
            query_embedding=embedding,
            seed_kg_node_ids=list(getattr(mem, "seed_kg_node_ids", []) or []),
        )
        # KG retrieve tool
        kg: KnowledgeRetrievalResult
        kg, kg_call_node = self.tool_runner.run_tool(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=new_index,
            tool_name="kg_retrieve",
            args=[],
            kwargs=kg_args,
            # {
            #     # "max_retrieval_level": max_retrieval_level,
            #     "seed_kg_node_ids": list(getattr(mem, "seed_kg_node_ids", []) or []),
            # }
            handler=kg_retriever.retrieve,
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary,
            orchestrator=self,
        )
        # self.join_tool_node_to_turn(conversation_id, kg_call_node, turn_node_id, prev_turn_meta_summary)
        if type(kg) is dict:
            kg = KnowledgeRetrievalResult(kg)
        st.knowledge = kg

        # pin memory (not a tool call; it's a graph mutation derived from the tool outputs)
        memory_pin: Optional[MemoryPinResult] = None
        if mem.selected and mem.memory_context_text:
            memory_pin = mem_retriever.pin_selected(
                user_id=user_id,
                current_conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                mem_id=mem_id,
                turn_index=new_index,
                self_span=self_span,
                selected_memory=mem.selected,
                memory_context_text=mem.memory_context_text,
                prev_turn_meta_summary=prev_turn_meta_summary,
            )
        pinned_ptrs = []
        pinned_edges = []
        st.memory_pin = memory_pin
        if kg.selected:
            pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
                user_id=user_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                turn_index=new_index,
                self_span=self_span,
                selected_knowledge=kg.selected,
                prev_turn_meta_summary=prev_turn_meta_summary,
            )
            st.pinned_kg_pointer_node_ids = pinned_ptrs
            st.pinned_kg_edge_ids = pinned_edges

        # 4) Answer (answer-only facade); response persistence happens inside the agent today.
        response = self.answer_only(
            conversation_id=conversation_id,
            prev_turn_meta_summary=prev_turn_meta_summary,
            cache_dir=cache_dir
        )
        st.answer = response
        response_turn_node_id = None

        if response.response_node_id is not None:
            response_turn_node_id = response.response_node_id
            turn_node: ConversationNode = self.conversation_engine.read.get_nodes(
                [response.response_node_id]
            )[0]

            if prev_node:
                seq_edge_id = get_id_for_conversation_turn_edge(
                    ConversationEdge.id_kind,
                    user_id,
                    conversation_id,
                    "next_turn",
                    turn_node.turn_index,
                    [prev_node.id],
                    [turn_node.id],
                    [],
                    [],
                    "conversation_edge",
                )
                self.add_link_to_new_turn(
                    seq_edge_id,
                    turn_node,
                    prev_node,
                    conversation_id,
                    span=self_span,
                    prev_turn_meta_summary=prev_turn_meta_summary,
                )
                prev_node = turn_node

            else:
                raise Exception("An AI turn must at least have a predecessor turn.")
        add_turn_result = AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=response_turn_node_id,
            turn_index=new_index,
            relevant_kg_node_ids=[
                i for i in (kg.selected.node_ids if kg.selected else [])
            ],
            relevant_kg_edge_ids=[
                i for i in (kg.selected.edge_ids if kg.selected else [])
            ],
            pinned_kg_pointer_node_ids=pinned_ptrs,
            pinned_kg_edge_ids=pinned_edges,
            memory_context_node_id=memory_pin.memory_context_node.id
            if memory_pin
            else None,
            memory_context_edge_ids=[i.id for i in memory_pin.pinned_edges]
            if memory_pin
            else [],
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

        # new_index = prev_turn_meta_summary.tail_turn_index
        # 5) Summarization trigger policy remains here; implementation stays in engine.

        if new_index > 0 and (
            prev_turn_meta_summary.prev_node_distance_from_last_summary
            - summary_turn_threshold
            >= 0
            or prev_turn_meta_summary.prev_node_char_distance_from_last_summary
            > summary_char_threshold
            or (
                summary_token_threshold is not None
                and _estimate_tokens_from_chars(
                    prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    token_estimator,
                )
                > summary_token_threshold
            )
            or (
                response and bool(getattr(response, "llm_decision_need_summary", False))
            )
        ):
            added_id = self._summarize_conversation_batch(
                conversation_id,
                new_index,
                prev_turn_meta_summary=prev_turn_meta_summary,
            )
            # prev_turn_meta_summary.tail_turn_index = new_index
            # summary added, user visible in chain node + 1
            new_index += 1
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
            add_turn_result = replace(add_turn_result, turn_index=new_index)
        return add_turn_result

    def get_chain_nodes(self, user_id, conversation_id):
        return self.conversation_engine.read.get_nodes(
            where={
                "$and": [{"in_ui_chain": True}] + [{"user_id": user_id}]
                if user_id
                else [] + [{"conversation_id": conversation_id}]
                if conversation_id
                else []
            }
        )

    def get_chain_length(self, user_id, conversation_id):
        return len(self.get_chain_nodes(user_id, conversation_id))

    # @conversation_only
    def _summarize_conversation_batch(
        self,
        conversation_id: str,
        current_index: int,
        batch_size: int = 5,
        in_conv=True,
        user_id: str = None,
        prev_turn_meta_summary: MetaFromLastSummary = None,
        cache_dir = None
    ):
        """Summarize a recent conversation window into a summary turn plus provenance edges.

        The batch may prepend the prior summary node, persists a context snapshot
        before the LLM summarize call, and then writes a summary turn plus
        summarization links back into the conversation graph. current_index tracks
        visible turn order, and prev_turn_meta_summary is mutated to reset summary
        distance counters after the batch is materialized.
        """
        # in_conversation  = False if side car
        if not in_conv:
            current_index -= 1
        if in_conv:
            prev_node = _get_conversation_tail_compat(
                self.conversation_engine, conversation_id
            )
        else:
            prev_node = None
        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        start_index = max(0, current_index - batch_size + 1)

        # Fetch the nodes
        # In a real impl, better querying needed.
        # We assume we can get them by index logic or linear scan
        # For prototype, scan and filter
        all_nodes = self.conversation_engine.read.get_nodes(
            where={"conversation_id": conversation_id}, node_type=ConversationNode
        )
        batch_ids = []
        batch_text = []
        provenance_spans = []
        if all_nodes is None:
            raise Exception("Unreacheable")

        nodes: list[ConversationNode] = all_nodes

        last_summary_node: list[ConversationNode] = last_summary_of_node(
            self.conversation_engine, prev_node
        )
        if last_summary_node:
            batch_ids.append(last_summary_node[-1].safe_get_id())
            batch_text.append(last_summary_node[-1].summary)
        for n in nodes:
            turn_index = getattr(
                n, "turn_index", None
            )  # non chain, snapshots/ subsidiary do not have turn index
            if (
                turn_index is not None
                and start_index <= int(turn_index) <= current_index
            ):
                # Check type
                # d = json.loads(doc)
                # d.update(meta)
                # n = ConversationNode.model_validate(d)
                if n.metadata.get("entity_type") == "conversation_turn":
                    batch_ids.append(n.id)
                    batch_text.append(f"{n.role}: {n.summary}")
                    # Collect provenance: The turn itself is the source
                    if n.mentions:
                        provenance_spans.extend(n.mentions[0].spans)
                elif n.type == "reference_pointer":
                    batch_ids.append(n.id)
                    batch_text.append(f"{n.role}: {n.summary}")
                    # Include referenced knowledge node in provenance too (as requested by user)
                    if n.mentions:
                        provenance_spans.extend(n.mentions[0].spans)

        if not batch_ids:
            return

        full_text = "\n".join(batch_text)

        # LLM Summarize
        if cache_dir:
            import joblib
            memory = joblib.Memory(cache_dir)
        else:
            memory = self.conversation_engine.memory  # .memory.cache
        @memory.cache
        def get_summary(full_text):
            summary_res = self.llm_tasks.summarize_context(
                SummarizeContextTaskRequest(full_text=str(full_text or ""))
            )
            return summary_res.text

        # Phase 2B: persist context snapshot BEFORE the LLM call (summary stage).
        from types import SimpleNamespace

        char_count = len(str(full_text or ""))
        view = SimpleNamespace(
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation segment into a concise memory.",
                },
                {"role": "user", "content": str(full_text or "")},
            ],
            items=[SimpleNamespace(node_id=nid) for nid in (batch_ids or [])],
            tokens_used=_estimate_tokens_from_chars(char_count),
        )
        clock = ExecClock(
            run_id=f"summary|{conversation_id}",
            run_step_seq=int(current_index),
            attempt_seq=0,
        )
        from .service import ConversationService

        svc = ConversationService.from_engine(
            self.conversation_engine,
            knowledge_engine=self.ref_knowledge_engine,
            workflow_engine=self.workflow_engine,
        )
        svc.persist_context_snapshot(
            conversation_id=conversation_id,
            run_id=clock.run_id,
            run_step_seq=clock.run_step_seq,
            attempt_seq=clock.attempt_seq,
            stage="summary",
            view=view,
            model_name=str(
                getattr(self.llm_tasks.provider_hints, "summarize_context_provider", "")
            ),
            budget_tokens=0,
            tail_turn_index=int(
                prev_turn_meta_summary.tail_turn_index if prev_turn_meta_summary else 0
            ),
        )

        summary_content = get_summary(full_text)
        # Dedupe spans for provenance
        unique_spans = []
        seen = set()
        for s in provenance_spans:
            # key based on loc
            k = (s.doc_id, s.start_char, s.end_char)
            if k not in seen:
                seen.add(k)
                unique_spans.append(s)
        import json

        # Create Summary Node
        new_index = prev_turn_meta_summary.tail_turn_index + 1
        content = str(summary_content)
        summary_turn_id = get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conversation_id,
            content,
            str(new_index),
            "system",
            "conversation_summary",
            str(in_conv),
        )
        summary_node = ConversationNode(
            id=summary_turn_id,
            label=f"Summary {start_index}-{current_index}",
            type="entity",
            summary=content,
            role="system",  # type: ignore
            conversation_id=conversation_id,
            turn_index=new_index,  # Anchored at end of batch
            # Provenance: We link back to the source turns and knowledge refs
            mentions=[Grounding(spans=unique_spans)] if unique_spans else [],
            properties={"content": json.dumps(summary_content)},
            metadata={
                "level_from_root": 1,
                "entity_type": "conversation_summary",
                "turn_index": new_index,
                "in_conversation_chain": in_conv,  # Summary is higher level?
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        prev_turn_meta_summary.tail_turn_index = new_index
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
        prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        # Persist

        self.conversation_engine.write.add_node(summary_node)
        new_index += 1

        # Edges: Summary -> Turns
        eid = f"summary|{summary_node.id}|batchhash|{stable_id(batch_ids)}"
        sum_edge = ConversationEdge(
            id=eid,
            source_ids=[summary_node.id],
            target_ids=batch_ids,
            relation="summarizes",
            label="summarizes",
            type="relationship",
            summary="Memory summarization",
            doc_id=f"conv:{conversation_id}",
            domain_id=None,
            canonical_entity_id=None,
            properties=None,
            embedding=None,
            mentions=[
                Grounding(
                    spans=[
                        Span(
                            collection_page_url=f"conversation/{conversation_id}",
                            document_page_url=f"conversation/{conversation_id}",
                            doc_id=f"conv:{conversation_id}",
                            chunk_id=None,
                            source_cluster_id=None,
                            insertion_method="summarization_link",
                            page_number=1,
                            start_char=0,
                            end_char=1,
                            excerpt="summary link",
                            context_before="",
                            context_after="",
                            verification=MentionVerification(
                                method="system",
                                is_verified=True,
                                score=1.0,
                                notes="link",
                            ),
                        )
                    ]
                )
            ],
            metadata={
                "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
            },
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.write.add_edge(sum_edge)

        if in_conv:
            if prev_node is None:
                raise Exception("unreacheable")
            # span = Span(
            #         collection_page_url=f"conversation/{conversation_id}",
            #         document_page_url=f"conversation/{conversation_id}",
            #         doc_id=f"conv:{conversation_id}",
            #         chunk_id = None,
            #         source_cluster_id = None,
            #         insertion_method="summarization_link",
            #         page_number=1, start_char=0, end_char=1,
            #         excerpt="summary link", context_before="", context_after="",
            #         verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="link")
            #     )
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{summary_node.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="summary_turn",
                page_number=1,
                start_char=0,
                end_char=len(summary_node.summary),
                excerpt=summary_node.summary,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="human",
                    is_verified=True,
                    score=1.0,
                    notes="system_summary_turn",
                ),
            )
            eid = get_id_for_conversation_turn_edge(
                ConversationEdge.id_kind,
                user_id,
                conversation_id,
                "next_turn",
                new_index,
                [prev_node.id],
                [summary_node.id],
                [],
                [],
                "conversation_edge",
            )
            self.add_link_to_new_turn(
                eid,
                summary_node,
                prev_node,
                conversation_id,
                self_span,
                prev_turn_meta_summary,
            )

            # add next turn relationship

        return summary_node.id

    # @property
    # def ui_turn_id(self):
    #     nodes=  self.conversation_engine.get_nodes(where= {"in_conversation_chain": True})
    #     return len(nodes)
    def answer_only(
        self,
        *,
        conversation_id: str,
        model_names: Optional[list[str]] = None,
        prev_turn_meta_summary: MetaFromLastSummary,
        cache_dir = None,
    ) -> ConversationAIResponse:
        """Generate an answer from the assembled conversation/KG context.

        This path answers against the current conversation view and pinned KG
        evidence without requiring the full workflow-driven assistant-turn
        materialization path. It is the explicit non-agentic/local-development
        route for "use the KG and current context to answer now".

        In other words, `answer_only(...)` is the direct "answer from the
        current conversation state plus retrieved knowledge" primitive. It is
        useful when callers want KG-backed answering without explicitly driving
        the conversation graph response workflow.
        """

        from .agentic_answering import AgenticAnsweringAgent, AgentConfig

        model_names = model_names or [
            str(
                getattr(
                    self.llm_tasks.provider_hints,
                    "answer_with_citations_provider",
                    "default",
                )
            )
        ]

        agent = AgenticAnsweringAgent(
            conversation_engine=self.conversation_engine,
            knowledge_engine=self.ref_knowledge_engine,
            llm_tasks=self.llm_tasks,
            config=AgentConfig(),
            cache_dir = cache_dir
        )
        out = agent.answer(
            conversation_id=conversation_id,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

        if isinstance(out, ConversationAIResponse):
            return out
        if isinstance(out, dict):
            return ConversationAIResponse(
                text=str(out.get("assistant_text") or out.get("text") or ""),
                llm_decision_need_summary=bool(
                    out.get("llm_decision_need_summary", False)
                ),
                used_kg_node_ids=list(
                    out.get("used_kg_node_ids") or out.get("used_node_ids") or []
                ),
                projected_conversation_node_ids=list(
                    out.get("projected_node_ids")
                    or out.get("projected_pointer_ids")
                    or []
                ),
                meta={"raw": out, "model_name": model_names[0]},
                response_node_id=out.get("assistant_turn_node_id"),
            )
        return ConversationAIResponse(text=str(out))
