"""Conversation service facade following service-pattern composition."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

from pydantic import BaseModel

from kogwistar.conversation.conversation_context import (
    ContextItem,
    ContextRenderer,
    ContextSources,
    ConversationContextView,
    DroppedItem,
    PromptContext,
    apply_ordering,
)
from kogwistar.conversation.conversation_orchestrator import (
    ConversationOrchestrator,
)
from kogwistar.conversation.models import (
    AddTurnResult,
    ContextSnapshotMetadata,
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
    RetrievalResult,
)
from kogwistar.conversation.policy import (
    get_chat_tail,
    get_last_seq_node,
    install_engine_hooks,
    last_summary_of_node,
    normalize_edge_metadata,
    validate_edge_add,
)
from kogwistar.llm_tasks import LLMTaskSet
from kogwistar.engine_core.models import (
    ContextCost,
    Grounding,
    MentionVerification,
    Span,
)
from kogwistar.id_provider import stable_id
from kogwistar.runtime import WorkflowRuntime

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine


class _ApproxTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class ConversationService:
    """High-level conversation behavior service."""

    def __init__(
        self,
        *,
        conversation_engine: "GraphKnowledgeEngine",
        knowledge_engine: "GraphKnowledgeEngine",
        workflow_engine: Optional["GraphKnowledgeEngine"] = None,
        llm_tasks: LLMTaskSet | None = None,
        runtime_cls: type[WorkflowRuntime] = WorkflowRuntime,
    ) -> None:
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine
        self.workflow_engine = workflow_engine
        self.llm_tasks = llm_tasks or conversation_engine.llm_tasks
        self.runtime_cls = runtime_cls
        install_engine_hooks(conversation_engine)

        self.orchestrator = ConversationOrchestrator(
            conversation_engine=conversation_engine,
            ref_knowledge_engine=knowledge_engine,
            workflow_engine=workflow_engine,
            llm_tasks=self.llm_tasks,
            tool_call_id_factory=stable_id,
        )

    @classmethod
    def from_engine(
        cls,
        conversation_engine: "GraphKnowledgeEngine",
        *,
        knowledge_engine: "GraphKnowledgeEngine | None" = None,
        workflow_engine: "GraphKnowledgeEngine | None" = None,
        llm_tasks: LLMTaskSet | None = None,
    ) -> "ConversationService":
        cache = getattr(conversation_engine, "_conversation_service_cache", None)
        if cache is None:
            cache = {}
            conversation_engine._conversation_service_cache = cache

        ke = knowledge_engine or conversation_engine
        we = workflow_engine
        tasks = llm_tasks or conversation_engine.llm_tasks
        key = (id(ke), id(we), id(tasks))
        svc = cache.get(key)
        if svc is not None:
            return svc

        svc = cls(
            conversation_engine=conversation_engine,
            knowledge_engine=ke,
            workflow_engine=we,
            llm_tasks=tasks,
        )
        cache[key] = svc
        return svc

    @classmethod
    def orchestrator_for_engine(
        cls,
        conversation_engine: "GraphKnowledgeEngine",
        *,
        ref_knowledge_engine: "GraphKnowledgeEngine",
    ) -> ConversationOrchestrator:
        svc = cls.from_engine(
            conversation_engine,
            knowledge_engine=ref_knowledge_engine,
            workflow_engine=getattr(conversation_engine, "workflow_engine", None),
        )
        return svc.orchestrator

    def max_node_seq_present(self, conversation_id):
        return self.conversation_engine.meta_sqlite.current_user_seq(conversation_id)

    def persist_workflow_cancel_request(
        self,
        *,
        conversation_id: str,
        run_id: str,
        workflow_id: str = "",
        requested_by: str = "user",
        reason: str = "cancel_requested",
    ) -> str:
        eng = self.conversation_engine
        node_id = f"wf_cancel_req|{run_id}"
        got = eng.backend.node_get(ids=[node_id], include=[])
        if got.get("ids"):
            return node_id

        node = ConversationNode(
            id=node_id,
            label="Workflow cancel request",
            type="entity",
            summary=f"workflow cancel requested for run_id={run_id}",
            conversation_id=conversation_id,
            role="system",
            turn_index=None,
            level_from_root=0,
            properties={"entity_type": "workflow_cancel_request"},
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "entity_type": "workflow_cancel_request",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "requested_by": requested_by,
                "reason": reason,
                "level_from_root": 0,
                "in_conversation_chain": False,
                "in_ui_chain": False,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        eng.write.add_node(node)

        run_node_id = f"wf_run|{run_id}"
        run_got = eng.backend.node_get(ids=[run_node_id], include=[])
        if run_got.get("ids"):
            content = f"{run_node_id} cancel_requested {node_id}"
            span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{node_id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="workflow_cancel_request",
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
                    notes="cancel request edge",
                ),
            )
            edge_id = str(
                stable_id("workflow.edge", "cancel_request", run_node_id, node_id)
            )
            existing_edge = eng.backend.edge_get(ids=[edge_id], include=[])
            if not existing_edge.get("ids"):
                edge = ConversationEdge(
                    id=edge_id,
                    source_ids=[run_node_id],
                    target_ids=[node_id],
                    relation="wf_cancel_request",
                    label="wf_cancel_request",
                    type="relationship",
                    summary="workflow cancel request",
                    doc_id=f"wf_cancel_request|{run_id}",
                    mentions=[Grounding(spans=[span])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "conversation_edge"},
                    embedding=None,
                    metadata={
                        "entity_type": "conversation_edge",
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "causal_type": "reference",
                    },
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                eng.write.add_edge(edge)
        return node_id

    def persist_workflow_cancelled_event(
        self,
        *,
        conversation_id: str,
        run_id: str,
        workflow_id: str = "",
        accepted_step_seq: int = -1,
        cancel_request_node_id: str | None = None,
        cancel_request_seq: int | None = None,
        accepted_watermark: int | None = None,
        last_processed_node_id: str | None = None,
    ) -> str:
        eng = self.conversation_engine
        node_id = f"wf_cancelled|{run_id}"
        got = eng.backend.node_get(ids=[node_id], include=[])
        if got.get("ids"):
            return node_id

        node = ConversationNode(
            id=node_id,
            label="Workflow cancelled",
            type="entity",
            summary=f"workflow cancelled run_id={run_id}",
            conversation_id=conversation_id,
            role="system",
            turn_index=None,
            level_from_root=0,
            properties={"entity_type": "workflow_cancelled"},
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "entity_type": "workflow_cancelled",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "accepted_step_seq": int(accepted_step_seq),
                "cancel_request_node_id": cancel_request_node_id,
                "cancel_request_seq": cancel_request_seq,
                "accepted_watermark": accepted_watermark,
                "last_processed_node_id": (
                    str(last_processed_node_id) if last_processed_node_id else None
                ),
                "level_from_root": 0,
                "in_conversation_chain": False,
                "in_ui_chain": False,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        eng.write.add_node(node)

        run_node_id = f"wf_run|{run_id}"
        run_got = eng.backend.node_get(ids=[run_node_id], include=[])
        if run_got.get("ids"):
            edge_id = str(stable_id("workflow.edge", "cancelled", run_node_id, node_id))
            existing_edge = eng.backend.edge_get(ids=[edge_id], include=[])
            if not existing_edge.get("ids"):
                edge = ConversationEdge(
                    id=edge_id,
                    source_ids=[run_node_id],
                    target_ids=[node_id],
                    relation="wf_cancelled",
                    label="wf_cancelled",
                    type="relationship",
                    summary="workflow cancelled",
                    doc_id=f"wf_cancelled|{run_id}",
                    mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "conversation_edge"},
                    embedding=None,
                    metadata={
                        "entity_type": "conversation_edge",
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "causal_type": "reference",
                    },
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                eng.write.add_edge(edge)

        if cancel_request_node_id:
            edge_id = str(
                stable_id(
                    "workflow.edge",
                    "cancel_reconciled",
                    cancel_request_node_id,
                    node_id,
                )
            )
            existing_edge = eng.backend.edge_get(ids=[edge_id], include=[])
            if not existing_edge.get("ids"):
                edge = ConversationEdge(
                    id=edge_id,
                    source_ids=[cancel_request_node_id],
                    target_ids=[node_id],
                    relation="wf_cancel_reconciled",
                    label="wf_cancel_reconciled",
                    type="relationship",
                    summary="cancel request reconciled",
                    doc_id=f"wf_cancelled|{run_id}",
                    mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "conversation_edge"},
                    embedding=None,
                    metadata={
                        "entity_type": "conversation_edge",
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "causal_type": "reference",
                    },
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                eng.write.add_edge(edge)

        if last_processed_node_id:
            edge_id = str(
                stable_id(
                    "workflow.edge",
                    "cancelled_at",
                    node_id,
                    str(last_processed_node_id),
                )
            )
            existing_edge = eng.backend.edge_get(ids=[edge_id], include=[])
            if not existing_edge.get("ids"):
                edge = ConversationEdge(
                    id=edge_id,
                    source_ids=[node_id],
                    target_ids=[str(last_processed_node_id)],
                    relation="wf_cancelled_at",
                    label="wf_cancelled_at",
                    type="relationship",
                    summary="workflow cancelled at node",
                    doc_id=f"wf_cancelled|{run_id}",
                    mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "conversation_edge"},
                    embedding=None,
                    metadata={
                        "entity_type": "conversation_edge",
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "causal_type": "reference",
                    },
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                eng.write.add_edge(edge)
        return node_id

    def get_last_seq_node(self, conversation_id, buffer=5):
        _ = buffer
        return self._get_last_seq_node(conversation_id)

    def _normalize_conversation_edge_metadata(self, edge: ConversationEdge) -> None:
        normalize_edge_metadata(edge)

    def _validate_conversation_edge_add(self, edge: ConversationEdge) -> None:
        validate_edge_add(self.conversation_engine, edge)

    def _create_conversation_primitive(
        self,
        user_id,
        conv_id=None,
        node_id: str | None | uuid.UUID = None,
    ) -> tuple[str, str]:
        from kogwistar.conversation.conversation_orchestrator import (
            get_id_for_conversation_turn,
        )

        eng = self.conversation_engine
        if eng.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        new_index = -1
        conv_id = conv_id or str(uuid.uuid4())
        node_id = node_id or get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conv_id,
            "Start of conversation",
            str(new_index),
            "system",
            "conversation_summary",
            in_conv=True,
        )
        start_node = ConversationNode(
            id=str(node_id),
            user_id=user_id,
            label="conversation start",
            type="entity",
            summary="Start of conversation",
            role="system",
            turn_index=-1,
            conversation_id=conv_id,
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"status": "active"},
            metadata={
                "level_from_root": 0,
                "entity_type": "conversation_start",
                "turn_index": -1,
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            doc_id=None,
            embedding=None,
        )
        dummy_span = Span.from_dummy_for_conversation()
        start_node.mentions = [Grounding(spans=[dummy_span])]
        writer = getattr(eng, "write", None)
        add_node = getattr(writer, "add_node", None) if writer is not None else None
        if callable(add_node):
            add_node(start_node)
        else:
            try:
                eng.write.add_node(start_node, None)
            except TypeError:
                eng.write.add_node(start_node)
        return conv_id, str(node_id)

    def create_conversation(
        self, user_id, conv_id=None, node_id: str | None | uuid.UUID = None
    ) -> tuple[str, str]:
        conv_out, node_out = self._create_conversation_primitive(
            user_id, conv_id, node_id
        )
        return str(conv_out), str(node_out)

    def _get_last_seq_node(self, conversation_id, min_seq=None):
        return get_last_seq_node(
            self.conversation_engine, conversation_id, min_seq=min_seq
        )

    def _get_conversation_tail(
        self,
        conversation_id: str,
        min_turn_index: int | None = None,
        tail_search_includes: list[str] = [
            "conversation_start",
            "conversation_turn",
            "conversation_summary",
            "assistant_turn",
        ],
    ) -> Optional[ConversationNode]:
        return get_chat_tail(
            self.conversation_engine,
            conversation_id=conversation_id,
            min_turn_index=min_turn_index,
            tail_search_includes=tail_search_includes,
        )

    def last_summary_of_node(self, node: ConversationNode):
        return last_summary_of_node(self.conversation_engine, node)

    def get_conversation_tail(
        self,
        conversation_id: str,
        min_turn_index: int | None = None,
        tail_search_includes: list[str] | None = None,
    ) -> Optional[ConversationNode]:
        return self._get_conversation_tail(
            conversation_id=conversation_id,
            min_turn_index=min_turn_index,
            tail_search_includes=tail_search_includes
            or [
                "conversation_start",
                "conversation_turn",
                "conversation_summary",
                "assistant_turn",
            ],
        )

    def add_turn(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn(*args, **kwargs)

    def add_conversation_turn(
        self,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: str,
        content: str,
        ref_knowledge_engine: "GraphKnowledgeEngine",
        filtering_callback: Callable[
            ..., tuple[FilteringResult | RetrievalResult, str]
        ],
        max_retrieval_level: int = 2,
        summary_char_threshold=12000,
        prev_turn_meta_summary: MetaFromLastSummary = MetaFromLastSummary(0, 0),
        add_turn_only=None,
    ) -> AddTurnResult:
        if ref_knowledge_engine is not self.knowledge_engine:
            self.knowledge_engine = ref_knowledge_engine
            self.orchestrator = ConversationOrchestrator(
                conversation_engine=self.conversation_engine,
                ref_knowledge_engine=ref_knowledge_engine,
                workflow_engine=self.workflow_engine,
                llm_tasks=self.llm_tasks,
            )
        return self.orchestrator.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            mem_id=mem_id,
            role=role,
            content=content,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
            summary_char_threshold=summary_char_threshold,
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=add_turn_only,
        )

    def add_turn_workflow_v2(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn_workflow_v2(*args, **kwargs)

    def answer_only(self, *args, **kwargs):
        return self.orchestrator.answer_only(*args, **kwargs)

    def get_conversation(self, conversation_id):
        _ = conversation_id
        return None

    def get_system_prompt(self, conversation_id: str) -> str:
        _ = conversation_id
        return "You are a helpful assistant. Answer the user using the conversation and any provided evidence."

    def get_response_model(self, conversation_id) -> Type[BaseModel]:
        _ = conversation_id
        return ConversationAIResponse

    def get_conversation_view(
        self,
        *,
        conversation_id: str,
        user_id: str | None = None,
        purpose: str = "answer",
        budget_tokens: int = 6000,
        tail_turns: int = 8,
        include_summaries: bool = True,
        include_memory_context: bool = True,
        include_pinned_kg_refs: bool = True,
        ordering_strategy: str | None = None,
    ):
        """Assemble a token-budgeted prompt view from conversation context sources.

        The view is built from summaries, memory context, pinned KG references, and
        tail turns, then packed under an approximate token budget with optional
        compression for items that advertise max_tokens. ordering_strategy can change
        packing order, but system-prompt injection and final tail-turn ordering stay
        deterministic.
        """
        _ = user_id
        tokenizer = _ApproxTokenizer()
        eng = self.conversation_engine

        sources = ContextSources(
            conversation_engine=eng,
            tail_turns=tail_turns,
            include_summaries=include_summaries,
            include_memory_context=include_memory_context,
            include_pinned_kg_refs=include_pinned_kg_refs,
        )
        items: list[ContextItem] = sources.gather(
            conversation_id=conversation_id, purpose=purpose
        )

        sys = self.get_system_prompt(conversation_id)
        items.insert(
            0,
            ContextItem(
                role="system",
                kind="system_prompt",
                text=str(sys or ""),
                node_id=None,
                priority=0,
                pinned=True,
                max_tokens=900,
                source="system",
            ),
        )

        priced: list[ContextItem] = []
        for it in items:
            cost = tokenizer.count_tokens(it.text or "")
            priced.append(ContextItem(**{**it.__dict__, "token_cost": cost}))

        if ordering_strategy is None or ordering_strategy == "default":
            pinned_non_turn = [i for i in priced if i.pinned and i.kind != "tail_turn"]
            tail_turn_items = [i for i in priced if i.kind == "tail_turn"]

            pinned_non_turn.sort(key=lambda x: x.priority)
            tail_turn_items.sort(key=lambda x: x.priority)

            kept: list[ContextItem] = []
            dropped: list[DroppedItem] = []
            used = 0

            def _try_add(it: ContextItem) -> bool:
                nonlocal used
                if used + it.token_cost <= budget_tokens:
                    kept.append(it)
                    used += it.token_cost
                    return True

                if it.max_tokens is not None and it.max_tokens < it.token_cost:
                    new_text = it.text[: max(1, it.max_tokens * 4)]
                    new_cost = tokenizer.count_tokens(new_text)
                    if used + new_cost <= budget_tokens:
                        kept.append(
                            ContextItem(
                                **{
                                    **it.__dict__,
                                    "text": new_text,
                                    "token_cost": new_cost,
                                }
                            )
                        )
                        used += new_cost
                        dropped.append(
                            DroppedItem(
                                kind=it.kind,
                                node_id=it.node_id,
                                reason="compressed",
                                token_cost=it.token_cost,
                            )
                        )
                        return True

                dropped.append(
                    DroppedItem(
                        kind=it.kind,
                        node_id=it.node_id,
                        reason="over_budget",
                        token_cost=it.token_cost,
                    )
                )
                return False

            for it in pinned_non_turn:
                _try_add(it)

            for it in tail_turn_items:
                _try_add(it)

            non_turn_kept = [i for i in kept if i.kind != "tail_turn"]
            turn_kept = [i for i in kept if i.kind == "tail_turn"]
            turn_kept.sort(key=lambda x: int((x.extra or {}).get("turn_index", 10**9)))
            kept = non_turn_kept + turn_kept
        else:
            iter_items = apply_ordering(
                items=list(priced), ordering=ordering_strategy, phase="pre_pack"
            )

            kept = []
            dropped = []
            used = 0

            def _try_add(it: ContextItem) -> bool:
                nonlocal used
                if used + it.token_cost <= budget_tokens:
                    kept.append(it)
                    used += it.token_cost
                    return True

                if it.max_tokens is not None and it.max_tokens < it.token_cost:
                    new_text = it.text[: max(1, it.max_tokens * 4)]
                    new_cost = tokenizer.count_tokens(new_text)
                    if used + new_cost <= budget_tokens:
                        kept.append(
                            ContextItem(
                                **{
                                    **it.__dict__,
                                    "text": new_text,
                                    "token_cost": new_cost,
                                }
                            )
                        )
                        used += new_cost
                        dropped.append(
                            DroppedItem(
                                kind=it.kind,
                                node_id=it.node_id,
                                reason="compressed",
                                token_cost=it.token_cost,
                            )
                        )
                        return True

                if it.kind == "system_prompt":
                    raise ValueError("System prompt alone exceeds budget")
                dropped.append(
                    DroppedItem(
                        kind=it.kind,
                        node_id=it.node_id,
                        reason="over_budget",
                        token_cost=it.token_cost,
                    )
                )
                return False

            for it in iter_items:
                _try_add(it)

            kept = apply_ordering(
                items=list(kept), ordering=ordering_strategy, phase="post_pack"
            )

        non_turn_kept = [i for i in kept if i.kind != "tail_turn"]
        turn_kept = [i for i in kept if i.kind == "tail_turn"]
        turn_kept.sort(key=lambda x: int((x.extra or {}).get("turn_index", 10**9)))
        kept = non_turn_kept + turn_kept

        renderer = ContextRenderer()
        messages = renderer.render(kept, purpose=purpose)

        included_node_ids = tuple(sorted({i.node_id for i in kept if i.node_id}))
        included_edge_ids = tuple(sorted({e for i in kept for e in (i.edge_ids or ())}))
        included_pointer_ids = tuple(
            sorted({p for i in kept for p in (i.pointer_ids or ()) if p})
        )

        head_summary_ids = tuple(
            i.node_id for i in kept if i.kind == "head_summary" and i.node_id
        )
        tail_turn_ids_out = tuple(
            i.node_id for i in kept if i.kind == "tail_turn" and i.node_id
        )
        active_memory_context_ids = tuple(
            i.node_id for i in kept if i.kind == "memory_context" and i.node_id
        )
        pinned_kg_ref_ids = tuple(
            i.node_id for i in kept if i.kind == "pinned_kg_ref" and i.node_id
        )

        return ConversationContextView(
            conversation_id=conversation_id,
            purpose=purpose,
            messages=tuple(messages),
            token_budget=budget_tokens,
            tokens_used=used,
            items=tuple(kept),
            dropped=tuple(dropped),
            included_node_ids=included_node_ids,
            included_edge_ids=included_edge_ids,
            included_pointer_ids=included_pointer_ids,
            head_summary_ids=head_summary_ids,
            tail_turn_ids=tail_turn_ids_out,
            active_memory_context_ids=active_memory_context_ids,
            pinned_kg_ref_ids=pinned_kg_ref_ids,
        )

    def make_conversation_span(self, conversation_id):
        return Span.from_dummy_for_conversation(conversation_id)

    def persist_context_snapshot(
        self,
        *,
        conversation_id: str,
        run_id: str,
        run_step_seq: int,
        attempt_seq: int = 0,
        stage: str,
        view: PromptContext,
        model_name: str = "",
        budget_tokens: int = 0,
        tail_turn_index: int = 0,
        extra_hash_payload=None,
        llm_input_payload: dict[str, Any] | None = None,
        evidence_pack_digest: dict[str, Any] | None = None,
    ) -> str:
        """Persist a stable snapshot node plus depends_on edges for one model step.

        The snapshot id is content-addressed by conversation, run, stage, step, and
        attempt, so repeated calls for the same rendered context reuse the same node.
        Stored metadata captures the prompt hash, sequencing fields, token cost, and
        every referenced node id so later audits can reconstruct what the model saw.
        """
        eng = self.conversation_engine

        def _stable_json(obj: Any) -> str:
            return json.dumps(
                obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )

        def _snapshot_hash(payload: Any) -> str:
            h = hashlib.sha256()
            h.update(_stable_json(payload).encode("utf-8"))
            return h.hexdigest()

        msgs = list(getattr(view, "messages", None) or [])
        norm_msgs: list[dict[str, str]] = []
        for m in msgs:
            role = getattr(m, "role", None) or (
                m.get("role") if isinstance(m, dict) else None
            )
            content = getattr(m, "content", None) or (
                m.get("content") if isinstance(m, dict) else None
            )
            norm_msgs.append({"role": str(role or ""), "content": str(content or "")})
        rendered_hash = _snapshot_hash(
            {
                "messages": norm_msgs,
                "llm_input_payload": llm_input_payload,
                "evidence_pack_digest": evidence_pack_digest,
                "extra_hash_payload": extra_hash_payload,
            }
        )

        used_node_ids: list[str] = []
        for it in view.items:
            nid = getattr(it, "node_id", None)
            if nid:
                used_node_ids.append(str(nid))

        char_count = sum(len(m.get("content", "")) for m in norm_msgs)
        token_count = getattr(getattr(view, "cost", None), "token_count", None)
        if token_count is None:
            token_count = getattr(view, "tokens_used", None)
        cost = ContextCost(
            char_count=int(char_count),
            token_count=(None if token_count is None else int(token_count)),
        )

        meta_model = ContextSnapshotMetadata(
            run_id=run_id,
            run_step_seq=int(run_step_seq),
            attempt_seq=int(attempt_seq),
            stage=str(stage),
            model_name=str(model_name or ""),
            budget_tokens=int(budget_tokens or 0),
            tail_turn_index=int(tail_turn_index or 0),
            used_node_ids=list(used_node_ids),
            rendered_context_hash=str(rendered_hash),
            cost=cost,
        )

        sid = str(
            stable_id(
                "conversation.context_snapshot",
                conversation_id,
                run_id,
                stage,
                str(int(run_step_seq)),
                str(int(attempt_seq)),
            )
        )

        existing = eng.backend.node_get(ids=[sid], include=[])
        if not existing.get("ids"):
            node = ConversationNode(
                id=sid,
                label="Context Snapshot",
                type="entity",
                summary="",
                conversation_id=conversation_id,
                role="system",  # type: ignore
                turn_index=None,
                level_from_root=0,
                properties={
                    "entity_type": "context_snapshot",
                    "prompt_messages": json.dumps(norm_msgs),
                    "llm_input_payload": json.dumps(llm_input_payload or {}),
                    "evidence_pack_digest": json.dumps(evidence_pack_digest or {}),
                },
                mentions=[
                    Grounding(spans=[self.make_conversation_span(conversation_id)])
                ],
                metadata={
                    "entity_type": "context_snapshot",
                    "level_from_root": 0,
                    "in_conversation_chain": False,
                    "in_ui_chain": False,
                    **meta_model.to_chroma_metadata(),
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            eng.write.add_node(node)

        scope = f"conv:{conversation_id}"
        for ordinal, nid in enumerate(used_node_ids):
            eid = str(
                stable_id(
                    "conversation.edge", scope, "depends_on", sid, nid, str(ordinal)
                )
            )
            ex = eng.backend.edge_get(ids=[eid], include=[])
            if ex.get("ids"):
                continue
            doc_id = scope
            sp = Span.from_dummy_for_conversation(doc_id=doc_id)
            edge = ConversationEdge(
                id=eid,
                source_ids=[sid],
                target_ids=[nid],
                relation="depends_on",
                label=f"depends_on:{sid}->{nid}",
                type="relationship",
                summary=f"Context snapshot {sid} depends on node {nid}",
                doc_id=doc_id,
                mentions=[Grounding(spans=[sp])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge", "ordinal": ordinal},
                embedding=None,
                metadata={
                    "entity_type": "conversation_edge",
                    "ordinal": ordinal,
                    "run_id": run_id,
                    "run_step_seq": int(run_step_seq),
                    "attempt_seq": int(attempt_seq),
                    "tail_turn_index": int(tail_turn_index or 0),
                },
                source_edge_ids=[],
                target_edge_ids=[],
            )
            eng.write.add_edge(edge)

        return sid

    def latest_context_snapshot_node(
        self,
        *,
        conversation_id: str,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> ConversationNode | None:
        where: dict[str, str] = {
            "entity_type": "context_snapshot",
        }
        if run_id is not None:
            where["run_id"] = run_id
        if stage is not None:
            where["stage"] = stage
        snaps = self.conversation_engine.read.get_nodes(
            where=where,
            node_type=ConversationNode,
            limit=10_000,
        )
        snaps = [
            n
            for n in snaps
            if str(getattr(n, "conversation_id", "") or "") == conversation_id
        ]
        if not snaps:
            return None

        def _k(n: ConversationNode):
            try:
                return int((n.metadata or {}).get("run_step_seq", 0))
            except Exception:
                return 0

        return sorted(snaps, key=_k)[-1]

    def get_context_snapshot_payload(
        self,
        *,
        snapshot_node_id: str,
    ) -> dict[str, Any]:
        got = self.conversation_engine.backend.node_get(
            ids=[snapshot_node_id], include=["documents", "metadatas"]
        )
        ids = got.get("ids") or []
        if not ids:
            raise KeyError(f"context snapshot node not found: {snapshot_node_id!r}")
        doc = (got.get("documents") or [None])[0]
        if isinstance(doc, str):
            try:
                payload = json.loads(doc)
                if isinstance(payload, dict):
                    props = payload.get("properties") or {}
                    return {
                        "properties": props,
                        "metadata": (payload.get("metadata") or {}),
                    }
            except Exception:
                pass
        return {
            "properties": {},
            "metadata": (got.get("metadatas") or [{}])[0] or {},
        }

    def latest_context_snapshot_cost(
        self,
        *,
        conversation_id: str,
        stage: str | None = None,
    ) -> ContextCost | None:
        n = self.latest_context_snapshot_node(
            conversation_id=conversation_id, stage=stage
        )
        if n is None:
            return None
        meta = getattr(n, "metadata", {}) or {}
        try:
            cs = ContextSnapshotMetadata.from_chroma_metadata(meta)
            return cs.cost
        except Exception:
            return None

    def get_ai_conversation_response(
        self, conversation_id, ref_knowledge_engine=None, model_names=None
    ) -> ConversationAIResponse:
        if (
            ref_knowledge_engine is not None
            and ref_knowledge_engine is not self.knowledge_engine
        ):
            self.knowledge_engine = ref_knowledge_engine
            self.orchestrator = ConversationOrchestrator(
                conversation_engine=self.conversation_engine,
                ref_knowledge_engine=ref_knowledge_engine,
                workflow_engine=self.workflow_engine,
                llm_tasks=self.llm_tasks,
            )
        return self.answer_only(
            conversation_id=conversation_id, model_names=model_names
        )
