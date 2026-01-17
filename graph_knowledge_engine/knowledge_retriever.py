from __future__ import annotations

import uuid
from typing import Callable, List, Optional, Tuple, cast

from langchain_core.language_models import BaseChatModel

from graph_knowledge_engine.agentic_answering import snapshot_hash
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import KnowledgeRetrievalResult, RetrievalResult

from .models import ConversationNode, ConversationEdge, FilteringResult, Grounding, MetaFromLastSummary, Span


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
        filtering_callback: Callable[..., tuple[FilteringResult, str]] ,
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
        )[0]
        edges = self.ref_knowledge_engine.query_edges(
            query_embeddings=[query_embedding],
            n_results=self.shallow_n_results,
            where={"level_from_root": {"$lte": self.max_retrieval_level}},
            include=["metadatas", "documents", "embeddings"],
        )[0]
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
        
        candidates = RetrievalResult(   nodes = list({n.id: n for n in shallow_results.nodes + deep_results.nodes}.values()), 
                                        edges = list({n.id: n for n in shallow_results.edges + deep_results.edges}.values()))
        

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
            selected, reasoning = self.filtering_callback(self.llm, user_text, cand_node_list_str, cand_edge_list_str, 
                                                          [i.id for i in candidates.nodes],
                                                          [i.id for i in candidates.edges], context_text)
            return KnowledgeRetrievalResult(candidate=candidates, selected=selected, reasoning=reasoning)
        else:
            return KnowledgeRetrievalResult(candidate=RetrievalResult(nodes = [], edges = []), 
                                            selected=FilteringResult(node_ids = [], edge_ids = []),  
                                            reasoning=reasoning)
        

    def pin_selected(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        selected_knowledge: Optional[FilteringResult],
        selected_knowledge_nodes: Optional[RetrievalResult] = None,
        prev_turn_meta_summary: MetaFromLastSummary| None = None
    ) -> tuple[List[str], List[str]]:
        pinned_pointer_node_ids: List[str] = []
        pinned_edge_ids: List[str] = []
        
        # ref_kg_engine: GraphKnowledgeEngine
        # ref_kg_engine = self.ref_knowledge_engine
        # kgs= selected_knowledge
        if prev_turn_meta_summary is None:
            raise Exception("prev_turn_meta_summary cannot be None") 
        if self.ref_knowledge_engine.kg_graph_type == "knowledge":
            pass
        else:
            raise(Exception("ref_knowledge_engine used conversation instead of knowledge kg_graph_type"))
        if selected_knowledge_nodes:
            nodes = selected_knowledge_nodes.nodes
            edges = selected_knowledge_nodes.edges
        else:
            if selected_knowledge:
                # do not read deprecated deleted nodes, it is ok if llm decided both node edges are empty list []
                if selected_knowledge.node_ids == []:
                    nodes = []
                else:
                    nodes = self.ref_knowledge_engine.get_nodes(selected_knowledge.node_ids, resolve_mode="redirect")
                if selected_knowledge.edge_ids == []:
                    edges = []
                else:
                    edges = self.ref_knowledge_engine.get_edges(selected_knowledge.edge_ids, resolve_mode="redirect")
            else:
                raise Exception("selected_knowledge and selected_knowledge_nodes cannot be both None")
        for kg  in nodes:
            # kg_got = self.ref_knowledge_engine.node_collection.get(ids=[kg_node_id], include=["documents", "embeddings", "metadatas"])
            
            # if not kg_got.get("ids"):
            #     continue
            
            
            # kg = ref_kg_engine.nodes_from_single_or_id_query_result( kg_got)
            
            kg_meta = kg.metadata
            summary = str(kg_meta.get("summary", ""))

            # ptr_id = str(uuid.uuid4())
            meta = kg_meta
            snap = {
                "entity_id": kg.id,
                "label": meta.get("label"),
                "summary": meta.get("summary"),
                "type": meta.get("type"),
                "canonical_entity_id": meta.get("canonical_entity_id"),
            }
            sh = snapshot_hash(snap)
            
            ptr_node = ConversationNode(
                id=None,
                label=f"Ref: {kg_meta.get('label')}",
                type="reference_pointer",
                doc_id=None,
                summary=summary,
                role="system",  # type: ignore
                turn_index=turn_index,
                conversation_id=conversation_id,
                user_id=user_id,
                mentions=[Grounding(spans=[self_span])],
                properties={
                    "target_namespace": "kg",
                    "refers_to_collection": "nodes",
                    "refers_to_id": kg.id,
                    "entity_type": "knowledge_reference",
                    "snapshot_hash": sh
                },
                metadata={
                    "target_namespace": "kg",
                    "entity_type": "knowledge_reference",
                    "level_from_root": 0,
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "snapshot_hash": sh,
                    "in_conversation_chain": False,
                },
                domain_id=None,
                canonical_entity_id=None,
                
            )
            
            ptr_id = cast(str, ptr_node.id)
            self.conversation_engine.add_node(ptr_node)
            pinned_pointer_node_ids.append(ptr_id)
            
            # edge_id = str(uuid.uuid4())
            edge = ConversationEdge(
                id=None,
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
                metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)
            pinned_edge_ids.append(edge.id)
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(summary)
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1

        for kg in edges:
            # kg_got = self.ref_knowledge_engine.node_collection.get(ids=[kg_id], include=["documents", "embeddings", "metadatas"])
            
            # if not kg_got.get("ids"):
            #     continue
            
            
            # kg = ref_kg_engine.nodes_from_single_or_id_query_result( kg_got)
            
            kg_meta = kg.metadata
            summary = str(kg_meta.get("summary", ""))

            # ptr_id = str(uuid.uuid4())
            ptr_node = ConversationNode(
                id=None,
                label=f"Ref: {kg_meta.get('label')}",
                type="reference_pointer",
                doc_id=None,
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
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "in_conversation_chain": False,
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            self.conversation_engine.add_node(ptr_node)
            ptr_id = cast(str, ptr_node.id)
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
                metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)
            pinned_edge_ids.append(edge_id)
            
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(summary)
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        return pinned_pointer_node_ids, pinned_edge_ids
