from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from graph_knowledge_engine.engine import GraphKnowledgeEngine

from .models import ConversationNode, ConversationEdge, Edge, Grounding, Node, Span


@dataclass
class KnowledgeRetrievalResult:
    candidate: RetrievalResult
    selected: RetrievalResult | None
    reasoning: str

@dataclass
class RetrievalResult:
    nodes: List[Node]
    edges: List[Edge]

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
        filtering_callback: Callable[..., tuple[RetrievalResult, str]] ,
        max_retrieval_level: int = 2,
        shallow_n_results: int = 20,
        deep_per_seed_results: int = 10,
        deep_seed_limit: int = 5,
    ) -> None:
        self.conversation_engine : GraphKnowledgeEngine= conversation_engine
        self.ref_knowledge_engine: GraphKnowledgeEngine = ref_knowledge_engine
        self.llm = llm
        self.filtering_callback = filtering_callback
        self.max_retrieval_level = max_retrieval_level
        self.shallow_n_results = shallow_n_results
        self.deep_per_seed_results = deep_per_seed_results
        self.deep_seed_limit = deep_seed_limit

    def _shallow_query(self, *, query_embedding: List[float]) -> RetrievalResult:
        nodes = self.ref_knowledge_engine.query_nodes(
            query_embeddings=[query_embedding],
            n_results=self.shallow_n_results,
            where={"level_from_root": {"$lte": self.max_retrieval_level}},
            include=["metadatas", "documents", "embeddings"],
        )
        edges = self.ref_knowledge_engine.query_edges(
            query_embeddings=[query_embedding],
            n_results=self.shallow_n_results,
            where={"level_from_root": {"$lte": self.max_retrieval_level}},
            include=["metadatas", "documents", "embeddings"],
        )
        return RetrievalResult(nodes, edges)
        # nodes = self.ref_knowledge_engine.query_nodes(query_embeddings=[query_embedding],
        #     n_results=self.shallow_n_results,
        #     where={"level_from_root": {"$lte": self.max_retrieval_level}},
        #     include=["metadatas", "documents", "embeddings"],)
        
        # return (rows.get("ids") or [[]])[0] or []

    def _deep_seeded_semantic(self, *, user_text: str, seed_kg_node_ids: List[str]) -> RetrievalResult:
        seed_ids = seed_kg_node_ids[: self.deep_seed_limit]
        # out: List[str] = []
        # for sid in seed_ids:
        layers = self.ref_knowledge_engine.query.k_hop(seed_ids)
        nodes = []
        edges = []
        for l in layers:
            nodes.extend(list(l['nodes']))
            edges.extend(list(l['edges']))
        return RetrievalResult(nodes = self.ref_knowledge_engine
                                  .nodes_from_single_or_id_query_result(self.ref_knowledge_engine.nodes_by_ids(nodes)),
                                  edges = self.ref_knowledge_engine
                                  .edges_from_single_or_id_query_result(self.ref_knowledge_engine.edges_by_ids(edges)))
            # got = self.ref_knowledge_engine.node_collection.get(ids=[sid], include=["metadatas"])
            # if not got.get("ids"):
            #     continue
            # meta = (got.get("metadatas") or [{}])[0] or {}
            # seed_summary = str(meta.get("summary") or meta.get("label") or "")
            # q_text = f"{user_text}\n\nRelated memory / prior context:\n{seed_summary}"
            # q_emb = self.ref_knowledge_engine.iterative_defensive_emb(q_text)
            # rows = self.ref_knowledge_engine.node_collection.query(
            #     query_embeddings=[q_emb],
            #     n_results=self.deep_per_seed_results,
            #     where={"level_from_root": {"$lte": self.max_retrieval_level}},
            #     include=["metadatas"],
            # )
            # ids = (rows.get("ids") or [[]])[0] or []
            # out.extend(ids)

        # seen = set()
        # dedup: List[str] = []
        # for x in out:
        #     if x not in seen:
        #         seen.add(x)
        #         dedup.append(x)
        # return dedup

    def retrieve(
        self,
        *,
        user_text: str,
        context_text: str,
        query_embedding: List[float],
        seed_kg_node_ids: Optional[List[str]] = None,
    ) -> KnowledgeRetrievalResult:
        shallow_results = self._shallow_query(query_embedding=query_embedding)
        # deep_ids: List[str] = []
        if seed_kg_node_ids:
            deep_results = self._deep_seeded_semantic(user_text=user_text, seed_kg_node_ids=seed_kg_node_ids)
        else:
            deep_results = RetrievalResult(nodes = [], edges = [])
        # merge
        candidates = RetrievalResult(nodes = list(set(shallow_results.nodes + deep_results.nodes)), 
                                        edges = list(set(shallow_results.edges + deep_results.edges)))
        

        reasoning = ""
        if candidates.edges or candidates.nodes:
            # rows = self.ref_knowledge_engine.node_collection.get(ids=candidate_ids, include=["metadatas"])
            # metas = rows.get("metadatas") or []
            
            cand_node_list_str = "\n".join(
                [f"-Node ID: {node.id} | Label: {node.metadata.get('label')} | Summary: {node.metadata.get('summary')}" for node in candidates.nodes]
            )
            cand_edge_list_str = "\n".join(
                [f"-Edge ID: {edge.id} | Label: {edge.metadata.get('label')} | Summary: {edge.metadata.get('summary')}" for edge in candidates.edges]
            )
            selected, reasoning = self.filtering_callback(self.llm, user_text, cand_node_list_str, cand_edge_list_str, candidates, context_text)
            return KnowledgeRetrievalResult(candidate=candidates, selected=selected, reasoning=reasoning)
        else:
            return KnowledgeRetrievalResult(candidate=RetrievalResult(nodes = [], edges = []), 
                                            selected=RetrievalResult(nodes = [], edges = []),  
                                            reasoning=reasoning)
        

    def pin_selected(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        selected_knowledge: RetrievalResult,
    ) -> tuple[List[str], List[str]]:
        pinned_pointer_node_ids: List[str] = []
        pinned_edge_ids: List[str] = []
        ref_kg_engine: GraphKnowledgeEngine
        ref_kg_engine = self.ref_knowledge_engine
        kgs= selected_knowledge
        for kg in kgs.nodes:
            # kg_got = self.ref_knowledge_engine.node_collection.get(ids=[kg_id], include=["documents", "embeddings", "metadatas"])
            
            # if not kg_got.get("ids"):
            #     continue
            
            
            # kg = ref_kg_engine.nodes_from_single_or_id_query_result( kg_got)
            
            kg_meta = kg.metadata
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
                    "refers_to_id": kg.id,
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

        for kg in kgs.nodes:
            # kg_got = self.ref_knowledge_engine.node_collection.get(ids=[kg_id], include=["documents", "embeddings", "metadatas"])
            
            # if not kg_got.get("ids"):
            #     continue
            
            
            # kg = ref_kg_engine.nodes_from_single_or_id_query_result( kg_got)
            
            kg_meta = kg.metadata
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
                    "refers_to_collection": "edges",
                    "refers_to_id": kg.id,
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
                summary="Turn references KG edge",
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
