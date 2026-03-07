from __future__ import annotations

from typing import Any, Callable

from ...id_provider import stable_id
from .base import NamespaceProxy


class ConversationSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Service/bootstrap helpers
    def get_conversation_service(self, *args, **kwargs):
        return self._e._get_conversation_service(*args, **kwargs)

    def get_orchestrator(self, *args, **kwargs):
        return self._e._get_orchestrator(*args, **kwargs)

    # Conversation primitive helpers
    def normalize_conversation_edge_metadata(self, *args, **kwargs):
        return self._e._normalize_conversation_edge_metadata(*args, **kwargs)

    def validate_conversation_edge_add(self, *args, **kwargs):
        return self._e._validate_conversation_edge_add(*args, **kwargs)

    def create_conversation_primitive(self, *args, **kwargs):
        return self._e._create_conversation_primitive(*args, **kwargs)

    def get_last_seq_node_internal(self, *args, **kwargs):
        return self._e._get_last_seq_node(*args, **kwargs)

    def get_conversation_tail(self, *args, **kwargs):
        return self._e._get_conversation_tail(*args, **kwargs)

    def where_and(self, *args, **kwargs):
        return self._e._where_and(*args, **kwargs)

    def edge_endpoints_exists(self, *args, **kwargs):
        return self._e._edge_endpoints_exists(*args, **kwargs)

    def edge_endpoints_first_edge_id(self, *args, **kwargs):
        return self._e._edge_endpoints_first_edge_id(*args, **kwargs)

    def conversation_doc_id_for_edge(self, *args, **kwargs):
        return self._e._conversation_doc_id_for_edge(*args, **kwargs)

    def is_duplicate_next_turn_noop(self, *args, **kwargs):
        return self._e._is_duplicate_next_turn_noop(*args, **kwargs)

    # Public conversation API surfaced from engine methods
    def max_node_seq_present(self, conversation_id):
        return self.get_conversation_service().max_node_seq_present(conversation_id)

    def get_last_seq_node(self, conversation_id, buffer=5):
        return self.get_conversation_service().get_last_seq_node(conversation_id, buffer=buffer)

    def create_conversation(self, *args, **kwargs):
        out = self.get_conversation_service().create_conversation(*args, **kwargs)
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("ConversationService.create_conversation must return (conversation_id, node_id)")
        conv_out, node_out = out
        return str(conv_out), str(node_out)

    def add_conversation_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: str,
        content: str,
        ref_knowledge_engine: Any,
        filtering_callback: Callable[..., Any],
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        prev_turn_meta_summary: Any = None,
        add_turn_only: bool | None = None,
    ):
        svc = self.get_conversation_service(knowledge_engine=ref_knowledge_engine)
        return svc.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            ref_knowledge_engine=svc.knowledge_engine,
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

    def respond_to_utterance(
        self,
        *,
        user_id: str,
        conversation_id: str,
        content: str,
        ref_knowledge_engine: Any,
        role: str = "user",
        turn_id: str | None = None,
        mem_id: str | None = None,
        filtering_callback: Callable[..., Any] | None = None,
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        prev_turn_meta_summary: Any = None,
        add_turn_only: bool | None = None,
    ):
        if mem_id is None:
            mem_id = str(stable_id("memory_context", user_id, conversation_id))
        if filtering_callback is None:
            # Avoid top-level circular import; resolve default callback lazily.
            from ..engine import candiate_filtering_callback

            filtering_callback = candiate_filtering_callback
        return self.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=(turn_id or ""),
            mem_id=mem_id,
            role=role,
            content=content,
            ref_knowledge_engine=ref_knowledge_engine,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
            summary_char_threshold=summary_char_threshold,
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=add_turn_only,
        )

    def get_conversation(self, conversation_id):
        return self.get_conversation_service().get_conversation(conversation_id)

    def get_system_prompt(self, conversation_id: str) -> str:
        return self.get_conversation_service().get_system_prompt(conversation_id)

    def get_response_model(self, conversation_id):
        return self.get_conversation_service().get_response_model(conversation_id)

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
        return self.get_conversation_service().get_conversation_view(
            conversation_id=conversation_id,
            user_id=user_id,
            purpose=purpose,
            budget_tokens=budget_tokens,
            tail_turns=tail_turns,
            include_summaries=include_summaries,
            include_memory_context=include_memory_context,
            include_pinned_kg_refs=include_pinned_kg_refs,
            ordering_strategy=ordering_strategy,
        )

    def make_conversation_span(self, conversation_id):
        return self.get_conversation_service().make_conversation_span(conversation_id)

    def persist_context_snapshot(
        self,
        *,
        conversation_id: str,
        run_id: str,
        run_step_seq: int,
        attempt_seq: int = 0,
        stage: str,
        view,
        model_name: str = "",
        budget_tokens: int = 0,
        tail_turn_index: int = 0,
        extra_hash_payload=None,
        llm_input_payload: dict[str, Any] | None = None,
        evidence_pack_digest: dict[str, Any] | None = None,
    ) -> str:
        return self.get_conversation_service().persist_context_snapshot(
            conversation_id=conversation_id,
            run_id=run_id,
            run_step_seq=run_step_seq,
            attempt_seq=attempt_seq,
            stage=stage,
            view=view,
            model_name=model_name,
            budget_tokens=budget_tokens,
            tail_turn_index=tail_turn_index,
            extra_hash_payload=extra_hash_payload,
            llm_input_payload=llm_input_payload,
            evidence_pack_digest=evidence_pack_digest,
        )

    def latest_context_snapshot_node(
        self,
        *,
        conversation_id: str,
        run_id: str | None = None,
        stage: str | None = None,
    ):
        return self.get_conversation_service().latest_context_snapshot_node(
            conversation_id=conversation_id,
            run_id=run_id,
            stage=stage,
        )

    def get_context_snapshot_payload(
        self,
        *,
        snapshot_node_id: str,
    ) -> dict[str, Any]:
        return self.get_conversation_service().get_context_snapshot_payload(
            snapshot_node_id=snapshot_node_id
        )

    def latest_context_snapshot_cost(
        self,
        *,
        conversation_id: str,
        stage: str | None = None,
    ):
        return self.get_conversation_service().latest_context_snapshot_cost(
            conversation_id=conversation_id,
            stage=stage,
        )

    def get_ai_conversation_response(
        self,
        conversation_id,
        ref_knowledge_engine,
        model_names=None,
    ):
        svc = self.get_conversation_service(knowledge_engine=ref_knowledge_engine)
        return svc.get_ai_conversation_response(
            conversation_id=conversation_id,
            ref_knowledge_engine=ref_knowledge_engine,
            model_names=model_names,
        )
