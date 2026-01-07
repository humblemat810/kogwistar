from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from .models import ConversationNode, ConversationEdge, Grounding, Span


@dataclass
class KnowledgeRetrievalResult:
    candidate_kg_node_ids: List[str]
    selected_kg_node_ids: List[str]
    reasoning: str


class KnowledgeRetriever:
    """
    KG retrieval with:
    - shallow vector retrieval
    - deep seeded semantic expansion (seed summaries condition the query)
    - pinning selected KG refs into the conversation canvas as pointer nodes

    Names (do not mix):
    - conversation_engine: conversation canvas engine (current engine instance)
    - ref_knowledge_engine: KG engine instance
    """

    def __init__(
        self,
        *,
        conversation_engine,
        ref_knowledge_engine,
        llm: BaseChatModel,
        filtering_callback: Callable[..., Tuple[List[str], str]],
        max_retrieval_level: int = 2,
        shallow_n_results: int = 20,
        deep_per_seed_results: int = 10,
        deep_seed_limit: int = 5,
    ) -> None:
        self.conversation_engine = conversation_engine
        self.ref_knowledge_engine = ref_knowledge_engine
        self.llm = llm
        self.filtering_callback = filtering_callback
        self.max_retrieval_level = max_retrieval_level
        self.shallow_n_results = shallow_n_results
        self.deep_per_seed_results = deep_per_seed_results
        self.deep_seed_limit = deep_seed_limit

    def _shallow_query(self, *, query_embedding: List[float]) -> List[str]:
        rows = self.ref_knowledge_engine.node_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.shallow_n_results,
            where={"level_from_root": {"$lte": self.max_retrieval_level}},
            include=["metadatas"],
        )
        return (rows.get("ids") or [[]])[0] or []

    def _deep_seeded_semantic(self, *, user_text: str, seed_kg_node_ids: List[str]) -> List[str]:
        seed_ids = seed_kg_node_ids[: self.deep_seed_limit]
        out: List[str] = []
        for sid in seed_ids:
            got = self.ref_knowledge_engine.node_collection.get(ids=[sid], include=["metadatas"])
            if not got.get("ids"):
                continue
            meta = (got.get("metadatas") or [{}])[0] or {}
            seed_summary = str(meta.get("summary") or meta.get("label") or "")
            q_text = f"{user_text}\n\nRelated memory / prior context:\n{seed_summary}"
            q_emb = self.ref_knowledge_engine.iterative_defensive_emb(q_text)
            rows = self.ref_knowledge_engine.node_collection.query(
                query_embeddings=[q_emb],
                n_results=self.deep_per_seed_results,
                where={"level_from_root": {"$lte": self.max_retrieval_level}},
                include=["metadatas"],
            )
            ids = (rows.get("ids") or [[]])[0] or []
            out.extend(ids)

        seen = set()
        dedup: List[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                dedup.append(x)
        return dedup

    def retrieve(
        self,
        *,
        user_text: str,
        context_text: str,
        query_embedding: List[float],
        seed_kg_node_ids: Optional[List[str]] = None,
    ) -> KnowledgeRetrievalResult:
        shallow_ids = self._shallow_query(query_embedding=query_embedding)
        deep_ids: List[str] = []
        if seed_kg_node_ids:
            deep_ids = self._deep_seeded_semantic(user_text=user_text, seed_kg_node_ids=seed_kg_node_ids)

        # merge
        seen = set()
        candidate_ids: List[str] = []
        for x in shallow_ids + deep_ids:
            if x not in seen:
                seen.add(x)
                candidate_ids.append(x)

        selected_ids: List[str] = []
        reasoning = ""
        if candidate_ids:
            rows = self.ref_knowledge_engine.node_collection.get(ids=candidate_ids, include=["metadatas"])
            metas = rows.get("metadatas") or []
            cand_list_str = "\n".join(
                [f"- ID: {rid} | Label: {meta.get('label')} | Summary: {meta.get('summary')}" for rid, meta in zip(candidate_ids, metas)]
            )
            selected_ids, reasoning = self.filtering_callback(self.llm, user_text, cand_list_str, candidate_ids, context_text)

        return KnowledgeRetrievalResult(candidate_kg_node_ids=candidate_ids, selected_kg_node_ids=selected_ids, reasoning=reasoning)

    def pin_selected(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        selected_kg_node_ids: List[str],
    ) -> tuple[List[str], List[str]]:
        pinned_pointer_node_ids: List[str] = []
        pinned_edge_ids: List[str] = []

        for kg_id in selected_kg_node_ids:
            kg_got = self.ref_knowledge_engine.node_collection.get(ids=[kg_id], include=["metadatas"])
            if not kg_got.get("ids"):
                continue
            kg_meta = (kg_got.get("metadatas") or [{}])[0] or {}
            summary = str(kg_meta.get("summary", ""))

            ptr_id = str(uuid.uuid4())
            ptr_node = ConversationNode(
                id=ptr_id,
                label=f"Ref: {kg_meta.get('label')}",
                type="reference_pointer",
                doc_id=ptr_id,
                summary=summary,
                role="system",  # type: ignore
                turn_index=turn_index,
                conversation_id=conversation_id,
                user_id=user_id,
                mentions=[Grounding(spans=[self_span])],
                properties={
                    "refers_to_collection": "nodes",
                    "refers_to_id": kg_id,
                    "entity_type": "knowledge_reference",
                },
                metadata={
                    "entity_type": "knowledge_reference",
                    "level_from_root": 0,
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            self.conversation_engine.add_node(ptr_node)
            pinned_pointer_node_ids.append(ptr_id)

            edge_id = str(uuid.uuid4())
            edge = ConversationEdge(
                id=edge_id,
                source_ids=[turn_node_id],
                target_ids=[ptr_id],
                relation="references",
                label="references",
                type="relationship",
                summary="Turn references KG node",
                doc_id=f"conv:{conversation_id}",
                mentions=[Grounding(spans=[self_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge"},
                embedding=None,
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)
            pinned_edge_ids.append(edge_id)

        return pinned_pointer_node_ids, pinned_edge_ids
