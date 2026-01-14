"""Conversation orchestrator (KGE-native, LangGraph-free).

This module owns *policy + control flow* for conversation turns, while delegating
all persistence/mutation to the conversation engine.

It is intentionally lightweight and uses your existing retrievers/agents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from server_mcp import GraphKnowledgeEngine

from .models import (
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    Grounding,
    KnowledgeRetrievalResult,
    MentionVerification,
    Span,
)
from .tool_runner import ToolRunner
from .memory_retriever import MemoryRetriever, MemoryPinResult, MemoryRetrievalResult
from .knowledge_retriever import KnowledgeRetriever


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
from graph_knowledge_engine.models import MetaFromLastSummary, RetrievalResult
class ConversationOrchestrator:
    """KGE-native orchestrator.

    Minimal, surgical refactor:
    - engine.add_conversation_turn(...) stays stable
    - orchestration steps live here
    - tool-call/tool-result events are recorded in the conversation graph
    """

    def __init__(
        self,
        *,
        conversation_engine: Any,
        ref_knowledge_engine: Any,
        tool_call_id_factory = uuid.uuid4,
        llm: BaseChatModel,
    ) -> None:
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.ref_knowledge_engine: GraphKnowledgeEngine = ref_knowledge_engine
        self.llm = llm
        self.tool_runner = ToolRunner(conversation_engine=conversation_engine,
                                      tool_call_id_factory=tool_call_id_factory)

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
        role: str,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
    ) -> Dict[str, Any]:
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
        prev_turn_meta_summary = MetaFromLastSummary(0,0)
        prev_node = self.conversation_engine._get_conversation_tail(conversation_id)
        if prev_node is not None:
            new_index = (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = prev_node.metadata.get("char_distance_from_last_summary") or 0
            prev_turn_meta_summary.prev_node_turn_distance_from_last_summary = prev_node.metadata.get("turn_distance_from_last_summary") or 0
        else:
            new_index = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_turn_distance_from_last_summary = -1
        
        # 1) Append the conversation turn node
        turn_node_id = turn_id or str(uuid.uuid4())
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
                method="human",
                is_verified=True,
                score=1.0,
                notes="verbatim user/system input",
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
            metadata={
                "entity_type": "conversation_turn",
                "level_from_root": 0,
                "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_turn_distance_from_last_summary,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
        prev_turn_meta_summary.prev_node_turn_distance_from_last_summary += 1
        new_index += 1
        emb_text0 = f"{role}: {content}"
        embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
        if embedding is None:
            raise Exception("uncalculatable embeddings")
        turn_node.embedding = embedding
        self.conversation_engine.add_node(turn_node, None)

        st.current_turn_node_id = turn_node_id
        st.turn_index = new_index
        st.self_span = self_span
        st.embedding = embedding

        # 2) Link sequentially
        if prev_node:
            seq_edge = ConversationEdge(
                id=str(uuid.uuid4()),
                source_ids=[prev_node.id],
                target_ids=[turn_node_id],
                relation="next_turn",
                label="next_turn",
                type="relationship",
                summary="Sequential flow",
                doc_id=f"conv:{conversation_id}",
                mentions=[Grounding(spans=[self_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity-type": "conversation_edge"},
                embedding=None,
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(seq_edge)
            prev_node = turn_node
        # 3) Retrieval + pinning (recorded as tool events)
        mem_retriever = MemoryRetriever(
            conversation_engine=self.conversation_engine,
            llm=self.llm,
            filtering_callback=filtering_callback,
        )
        kg_retriever = KnowledgeRetriever(
            conversation_engine=self.conversation_engine,
            ref_knowledge_engine=self.ref_knowledge_engine,
            llm=self.llm,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
        )

        # memory retrieve tool
        mem: MemoryRetrievalResult = self.tool_runner.run_tool(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=new_index,
            tool_name="memory_retrieve",
            args={
                "n_results": mem_retriever.n_results,
            },
            handler=lambda: mem_retriever.retrieve(
                user_id=user_id,
                current_conversation_id=conversation_id,
                query_embedding=embedding,
                user_text=content,
                context_text="",
            ),
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary
        )
        st.memory = mem

        # KG retrieve tool
        kg: KnowledgeRetrievalResult = self.tool_runner.run_tool(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=new_index,
            tool_name="kg_retrieve",
            args={
                "max_retrieval_level": max_retrieval_level,
                "seed_kg_node_ids": list(getattr(mem, "seed_kg_node_ids", []) or []),
            },
            handler=lambda: kg_retriever.retrieve(
                user_text=content,
                context_text="",
                query_embedding=embedding,
                seed_kg_node_ids=list(getattr(mem, "seed_kg_node_ids", []) or []),
            ),
            render_result=lambda r: getattr(r, "reasoning", "")[:800],
            prev_turn_meta_summary=prev_turn_meta_summary
        )
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
            )
            st.pinned_kg_pointer_node_ids = pinned_ptrs
            st.pinned_kg_edge_ids = pinned_edges

        # 4) Answer (answer-only facade); response persistence happens inside the agent today.
        response = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
        st.answer = response
        if response.response_node_id is not None:
            turn_node = self.conversation_engine.get_nodes([response.response_node_id])[0]
            if prev_node:
                seq_edge = ConversationEdge(
                    id=str(uuid.uuid4()),
                    source_ids=[prev_node.id],
                    target_ids=[turn_node.id],
                    relation="next_turn",
                    label="next_turn",
                    type="relationship",
                    summary="Sequential flow",
                    doc_id=f"conv:{conversation_id}",
                    mentions=[Grounding(spans=[self_span])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity-type": "conversation_edge"},
                    embedding=None,
                    metadata={},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                self.conversation_engine.add_edge(seq_edge)
                prev_node = turn_node
            else:
                raise Exception("An AI turn must at least have a predecessor turn.")
        
        # 5) Summarization trigger policy remains here; implementation stays in engine.
        
        if new_index > 0 and (
            prev_turn_meta_summary.prev_node_turn_distance_from_last_summary % 5 == 0
            or prev_turn_meta_summary.prev_node_char_distance_from_last_summary > summary_char_threshold
            or bool(getattr(response, "llm_decision_need_summary", False))
        ):
            self.conversation_engine._summarize_conversation_batch(conversation_id, new_index)

        return {
            "turn_node_id": turn_node_id,
            "turn_index": new_index,
            "relevant_kg_node_ids": [i for i in (kg.selected.node_ids if kg.selected else [])],
            "relevant_kg_edge_ids": [i for i in (kg.selected.edge_ids if kg.selected else [])],
            "pinned_kg_pointer_node_ids": pinned_ptrs,
            "pinned_kg_edge_ids": pinned_edges,
            "memory_context_node_id": memory_pin.memory_context_node if memory_pin else None,
            "memory_context_edge_ids": memory_pin.pinned_edges if memory_pin else [],
        }

    def answer_only(self, *, conversation_id: str, model_names: Optional[list[str]] = None, prev_turn_meta_summary: MetaFromLastSummary) -> ConversationAIResponse:
        """Generate an answer for an existing conversation (no new user turn ingestion)."""

        from .agentic_answering import AgenticAnsweringAgent, AgentConfig

        model_names = model_names or [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ]

        last_err: Exception | None = None
        for model_name in model_names:
            try:
                llm = self.conversation_engine.get_llm(model_name) or self.llm
                agent = AgenticAnsweringAgent(
                    conversation_engine=self.conversation_engine,
                    knowledge_engine=self.ref_knowledge_engine,
                    llm=llm,
                    config=AgentConfig(),
                )
                out = agent.answer(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)

                if isinstance(out, ConversationAIResponse):
                    return out
                if isinstance(out, dict):
                    return ConversationAIResponse(
                        text=str(out.get("assistant_text") or out.get("text") or ""),
                        llm_decision_need_summary=bool(out.get("llm_decision_need_summary", False)),
                        used_kg_node_ids=list(out.get("used_kg_node_ids") or out.get("used_node_ids") or []),
                        projected_conversation_node_ids=list(out.get("projected_node_ids") or out.get("projected_pointer_ids") or []),
                        meta={"raw": out, "model_name": model_name},
                        response_node_id = out.get("assistant_turn_node_id")
                    )
                return ConversationAIResponse(text=str(out))
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"tried all models; last error: {last_err}")
