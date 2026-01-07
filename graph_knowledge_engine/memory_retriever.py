from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from .models import ConversationNode, ConversationEdge, Grounding, Span
from .engine import GraphKnowledgeEngine

@dataclass
class MemoryRetrievalResult:
    # Cross-conversation memory candidates (by user_id)
    candidate_node_ids: List[str]
    selected_node_ids: List[str]
    reasoning: str

    # Derived artifacts
    memory_context_text: str
    seed_kg_node_ids: List[str]


@dataclass
class MemoryPinResult:
    memory_context_node_id: str
    pinned_edge_ids: List[str]


class MemoryRetriever:
    """Long-memory retriever over the conversation store, keyed by user_id.

    This retriever supports:
    - selecting prior summary/turn nodes across conversations for the same user
    - synthesizing a compact `memory_context` node for the *current* turn
    - extracting KG seeds from selected nodes when they contain pinned KG pointers
      (type == 'reference_pointer' and properties['refers_to_id'] exists)

    Naming conventions (do not mix):
    - conversation_engine: the conversation canvas engine (current engine instance)
    - user_id: identity used for cross-conversation retrieval
    """

    def __init__(
        self,
        *,
        conversation_engine,
        llm: BaseChatModel,
        filtering_callback: Callable[..., Tuple[List[str], str]],
        summarize_callback: Optional[Callable[..., str]] = None,
        n_results: int = 12,
        prefer_types: Optional[List[str]] = None,
    ) -> None:
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.llm = llm
        self.filtering_callback = filtering_callback
        self.summarize_callback = summarize_callback
        self.n_results = n_results
        self.prefer_types = prefer_types or ["conversation_summary", "conversation_turn", "reference_pointer"]

    def retrieve(
        self,
        *,
        user_id: str,
        current_conversation_id: str,
        query_embedding: List[float],
        user_text: str,
        context_text: str,
    ) -> MemoryRetrievalResult:
        # Broad memory retrieval across same user
        where = {"user_id": user_id}
        # _nodes = self.conversation_engine.query_nodes(query_embeddings = query_embedding)
        rows = self.conversation_engine.node_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.n_results,
            where=where,
            include=["metadatas", "documents", "embeddings"],
        )
        # _nodes = self.conversation_engine.nodes_from_query_result(rows, node_type = ConversationNode)
        candidate_ids: List[str] = (rows.get("ids") or [[]])[0] or []
        candidate_docs: List[str] = (rows.get("documents") or [[]])[0] or []
        candidate_metas = (rows.get("metadatas") or [[]])[0] or []

        # Optional type preference reorder (soft heuristic, no filtering)
        if candidate_ids and candidate_metas:
            zipped = list(zip(candidate_ids, candidate_docs, candidate_metas))
            def _rank(m):
                t = (m or {}).get("type") or (m or {}).get("entity_type") or ""
                return 0 if t in self.prefer_types else 1
            zipped.sort(key=lambda x: _rank(x[2]))
            candidate_ids = [z[0] for z in zipped]
            candidate_docs = [z[1] for z in zipped]
            candidate_metas = [z[2] for z in zipped]
        import json
        selected_ids: List[str] = []
        reasoning = ""
        if candidate_ids:
            # Keep prompt compact: show meta summaries only
            cand_list_str = "\n".join(
                [
                    f"- ID: {rid} | conversation_id: {meta.get('conversation_id')} | type: {meta.get('type')} | entity_type: {meta.get('entity_type')} | summary: {meta.get('summary')}"
                    for rid, meta in zip(candidate_ids, candidate_metas)
                ]
            )
            selected_ids, reasoning = self.filtering_callback(self.llm, user_text, cand_list_str, candidate_ids, context_text)
        
        # Extract KG seeds from selected nodes when they are reference pointers
        seed_kg_ids: List[str] = []
        if selected_ids:
            selected_rows = self.conversation_engine.node_collection.get(ids=selected_ids, include=["metadatas","documents"])
            # nodes = self.conversation_engine.nodes_from_query_result(rows, node_type = ConversationNode)
            docs = selected_rows.get("documents") or []
            metadatas =selected_rows.get("metadatas") or []
            for doc, meta in zip(docs, metadatas) :
                try:
                    djson = json.loads(doc)
                    djson.update({'metadata':meta})
                    n = ConversationNode.model_validate(djson)
                except Exception:
                    continue
                if n.type != "reference_pointer":
                    continue
                rid = (n.properties or {}).get("refers_to_id")
                if isinstance(rid, str) and rid:
                    seed_kg_ids.append(rid)

        # De-dupe seeds in order
        seen = set()
        dedup_seeds: List[str] = []
        for x in seed_kg_ids:
            if x not in seen:
                seen.add(x)
                dedup_seeds.append(x)

        # Build memory context text (LLM summarize or fallback join)
        memory_context_text = ""
        if selected_ids:
            if self.summarize_callback is not None:
                memory_context_text = self.summarize_callback(self.llm, user_text, selected_ids)
            else:
                # Fallback: concatenate short snippets from metadata/documents
                # Keep it bounded: use first 5 items
                parts = []
                for rid, meta in list(zip(selected_ids, candidate_metas))[:5]:
                    parts.append(f"[{rid}] {meta.get('summary') or ''}")
                memory_context_text = "\n".join(parts).strip()

        return MemoryRetrievalResult(
            candidate_node_ids=candidate_ids,
            selected_node_ids=selected_ids,
            reasoning=reasoning,
            memory_context_text=memory_context_text,
            seed_kg_node_ids=dedup_seeds,
        )

    def pin_selected(
        self,
        *,
        user_id: str,
        current_conversation_id: str,
        mem_id: str | None = None, # for both node and edge
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        selected_source_node_ids: List[str],
        memory_context_text: str,
    ) -> Optional[MemoryPinResult]:
        """Materialize a `memory_context` node into the current conversation canvas.

        Edges:
        - Turn --has_memory_context--> MemoryContext
        - MemoryContext --summarizes--> SourceNode (by id)

        If nothing selected, returns None.
        """
        if not selected_source_node_ids or not memory_context_text:
            return None

        mem_node_id = mem_id or str(uuid.uuid4())
        mem_node = ConversationNode(
            id=mem_node_id,
            label=f"Memory context (turn {turn_index})",
            type="entity",
            doc_id=mem_node_id,
            summary=memory_context_text,
            role="system",  # type: ignore
            turn_index=turn_index,
            conversation_id=current_conversation_id,
            user_id=user_id,
            mentions=[Grounding(spans=[self_span])],
            properties={
                "user_id": user_id,
                "source_node_ids": list(selected_source_node_ids),
            },
            metadata={
                "entity_type": "memory_context",
                "type": "entity",
                "level_from_root": 0, 
                "char_distance_from_last_summary": 0,  # memory itself is a summary of other nodes
                "turn_distance_from_last_summary": 0,                
                "user_id": user_id,
                "conversation_id": current_conversation_id,
                "turn_index": turn_index,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        self.conversation_engine.add_node(mem_node)

        edge_ids: List[str] = []

        # Turn -> MemoryContext
        e1_id = mem_id or str(uuid.uuid4())
        e1 = ConversationEdge(
            id=e1_id,
            source_ids=[turn_node_id],
            target_ids=[mem_node_id],
            relation="has_memory_context",
            label="has_memory_context",
            type="relationship",
            summary="This turn uses long-memory context",
            doc_id=f"conv:{current_conversation_id}",
            mentions=[Grounding(spans=[self_span])],
            domain_id=None,
            canonical_entity_id=None,
            properties={"entity-type": "conversation_edge"},
            embedding=None,
            metadata={},
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.add_edge(e1)
        edge_ids.append(e1_id)

        # MemoryContext -> Sources
        for sid in selected_source_node_ids:
            e_id = f"{mem_id}::{sid}" if mem_id else str(uuid.uuid4())
            e = ConversationEdge(
                id=e_id,
                source_ids=[mem_node_id],
                target_ids=[sid],
                relation="summarizes",
                label="summarizes",
                type="relationship",
                summary="Memory context summarizes prior conversation artifact",
                doc_id=f"conv:{current_conversation_id}",
                mentions=[Grounding(spans=[self_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity-type": "conversation_edge"},
                embedding=None,
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(e)
            edge_ids.append(e_id)

        return MemoryPinResult(memory_context_node_id=mem_node_id, pinned_edge_ids=edge_ids)
