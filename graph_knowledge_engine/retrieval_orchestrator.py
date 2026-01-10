from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from .memory_retriever import MemoryRetriever, MemoryRetrievalResult, MemoryPinResult
from .knowledge_retriever import KnowledgeRetriever
from .models import KnowledgeRetrievalResult, Span


@dataclass
class RetrievalOutcome:
    memory: MemoryRetrievalResult
    knowledge: KnowledgeRetrievalResult
    memory_pin: Optional[MemoryPinResult]
    pinned_kg_pointer_node_ids: List[str]
    pinned_kg_edge_ids: List[str]

    @property
    def memory_context_node_id(self):
        return self.memory_pin.memory_context_node_id if self.memory_pin else None

    @property
    def memory_context_edge_ids(self):
        return self.memory_pin.pinned_edge_ids if self.memory_pin else []


class RetrievalOrchestrator:
    """
    Coordinates memory + KG retrieval for a single turn.

    Names (do not mix):
    - conversation_engine: the conversation canvas engine instance (current engine / self in add_conversation_turn)
    - ref_knowledge_engine: the knowledge graph engine instance used for KG retrieval
    
    serve multiple user_id, an abstration across engine.
    
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
        self.max_retrieval_level = max_retrieval_level

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
        mem_id: str,
        user_text: str,
        query_embedding: List[float],
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        prev_char_distance_from_last_summary: int,
        prev_turn_distance_from_last_summary: int,
    ) -> RetrievalOutcome:
        # 1) memory retrieve across this user_id
        mem : MemoryRetrievalResult = self.memory_retriever.retrieve(
            user_id=user_id,
            current_conversation_id=conversation_id,
            query_embedding=query_embedding,
            user_text=user_text,
            context_text = "" # for inserting research progress so far if iterative agent later
        )
         # 2) KG retrieval seeded by memory-derived KG ids (from selected pointers)
        kg : KnowledgeRetrievalResult = self.knowledge_retriever.retrieve(
            user_text=user_text,
            context_text = "", # for inserting research progress so far if iterative agent later
            query_embedding=query_embedding,
            seed_kg_node_ids=mem.seed_kg_node_ids,
        )
        
        # 3) pin memory_context into current canvas (if any selection)
        memory_pin: Optional[MemoryPinResult] = None
        if mem.selected and mem.memory_context_text:
            memory_pin = self.memory_retriever.pin_selected(                
                user_id=user_id,
                current_conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                mem_id=mem_id,
                turn_index=turn_index,
                self_span=self_span,
                selected_memory=mem.selected,
                memory_context_text=mem.memory_context_text,
            )

       


        # 4) pin selected KG refs
        pinned_ptrs, pinned_edges = self.knowledge_retriever.pin_selected(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            self_span=self_span,
            selected_knowledge=kg.selected,
            selected_knowledge_nodes=kg.get_filtered_candidate()
        )

        return RetrievalOutcome(
            memory=mem,
            knowledge=kg,
            memory_pin=memory_pin,
            pinned_kg_pointer_node_ids=pinned_ptrs,
            pinned_kg_edge_ids=pinned_edges,
        )
