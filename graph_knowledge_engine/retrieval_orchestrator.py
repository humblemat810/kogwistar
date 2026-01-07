from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from .memory_retriever import MemoryRetriever, MemoryRetrievalResult, MemoryPinResult
from .knowledge_retriever_v2 import KnowledgeRetriever, KnowledgeRetrievalResult
from .models import Span


@dataclass
class RetrievalOutcome:
    memory: MemoryRetrievalResult
    knowledge: KnowledgeRetrievalResult
    memory_pin: Optional[MemoryPinResult]
    pinned_kg_pointer_node_ids: List[str]
    pinned_kg_edge_ids: List[str]


class RetrievalOrchestrator:
    """Coordinates memory + KG retrieval and pinning for a single turn.

    Naming (do not mix):
    - conversation_engine: canvas engine instance (self in add_conversation_turn)
    - ref_knowledge_engine: knowledge graph engine instance
    """

    def __init__(
        self,
        *,
        conversation_engine,
        ref_knowledge_engine,
        llm: BaseChatModel,
        memory_filtering_callback: Callable[..., Tuple[List[str], str]],
        knowledge_filtering_callback: Callable[..., Tuple[List[str], str]],
        max_retrieval_level: int = 2,
    ) -> None:
        self.conversation_engine = conversation_engine
        self.ref_knowledge_engine = ref_knowledge_engine
        self.llm = llm

        self.memory_retriever = MemoryRetriever(
            conversation_engine=conversation_engine,
            llm=llm,
            filtering_callback=memory_filtering_callback,
        )
        self.knowledge_retriever = KnowledgeRetriever(
            conversation_engine=conversation_engine,
            ref_knowledge_engine=ref_knowledge_engine,
            llm=llm,
            filtering_callback=knowledge_filtering_callback,
            max_retrieval_level=max_retrieval_level,
        )

    def run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_text: str,
        query_embedding: List[float],
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        prev_char_distance_from_last_summary: int,
        prev_turn_distance_from_last_summary: int,
    ) -> RetrievalOutcome:
        # 1) memory retrieve (cross-conversation, keyed by user_id)
        mem = self.memory_retriever.retrieve(
            user_id=user_id,
            current_conversation_id=conversation_id,
            query_embedding=query_embedding,
            user_text=user_text,
        )

        # 2) pin memory context into current conversation (one compact node)
        mem_pin = self.memory_retriever.pin_selected(
            user_id=user_id,
            current_conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            self_span=self_span,
            selected_source_node_ids=mem.selected_node_ids,
            memory_context_text=mem.memory_context_text,
        )

        # 3) KG retrieve (shallow + seeded deep)
        kg = self.knowledge_retriever.retrieve(
            user_text=user_text,
            query_embedding=query_embedding,
            seed_kg_node_ids=mem.seed_kg_node_ids,
        )

        # 4) pin selected KG refs
        pinned_ptrs, pinned_edges, _, _ = self.knowledge_retriever.pin_selected(
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            self_span=self_span,
            selected_kg_node_ids=kg.selected_kg_node_ids,
            prev_char_distance_from_last_summary=prev_char_distance_from_last_summary,
            prev_turn_distance_from_last_summary=prev_turn_distance_from_last_summary,
        )

        return RetrievalOutcome(
            memory=mem,
            knowledge=kg,
            memory_pin=mem_pin,
            pinned_kg_pointer_node_ids=pinned_ptrs,
            pinned_kg_edge_ids=pinned_edges,
        )
