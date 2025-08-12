from __future__ import annotations

import json
import itertools
from typing import List, Tuple, Any, Dict, Optional

from ..models import Node, Edge, AdjudicationCandidate
from ..strategies import MergeCandidateProposer, EngineLike


# ------------------------
# Utilities (local helpers)
# ------------------------

def _norm_label(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _edge_signature(e: Edge) -> str:
    """
    Prefer explicit signature_text if present; otherwise derive a stable signature
    from relation + sorted endpoints (node ids and meta-edge ids).
    """
    if e.properties and isinstance(e.properties, dict) and e.properties.get("signature_text"):
        return str(e.properties["signature_text"]).strip()

    s_nodes = tuple(sorted((e.source_ids or [])))
    t_nodes = tuple(sorted((e.target_ids or [])))
    s_edges = tuple(sorted((getattr(e, "source_edge_ids", []) or [])))
    t_edges = tuple(sorted((getattr(e, "target_edge_ids", []) or [])))
    return f"{e.relation}|S:{s_nodes}|T:{t_nodes}|SE:{s_edges}|TE:{t_edges}"

def _pairwise(items: List[Any]) -> List[Tuple[Any, Any]]:
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out

def _fetch_doc_node_ids(engine: EngineLike, doc_id: str) -> List[str]:
    """
    Uses node_docs index if present (fast), else falls back to scanning node metadata (slower).
    """
    try:
        rows = engine.node_docs_collection.get(where={"doc_id": doc_id}, include=["documents"])
        if rows.get("documents"):
            return list({json.loads(d)["node_id"] for d in rows["documents"]})
    except Exception:
        pass

    # Fallback: nodes with a denormalized doc_ids array that contains doc_id
    nodes = engine.node_collection.get(where={"doc_ids": {"$contains": doc_id}}, include=["ids"])
    if nodes.get("ids"):
        return list(nodes["ids"])

    # Last resort: scan by 'doc_id' direct (older schema)
    nodes = engine.node_collection.get(where={"doc_id": doc_id}, include=["ids"])
    return list(nodes.get("ids") or [])

def _fetch_doc_nodes(engine: EngineLike, doc_id: str) -> List[Node]:
    ids = _fetch_doc_node_ids(engine, doc_id)
    if not ids:
        return []
    got = engine.node_collection.get(ids=ids, include=["documents"])
    return [Node.model_validate_json(j) for j in (got.get("documents") or [])]

def _fetch_doc_edges(engine: EngineLike, doc_id: str) -> List[Edge]:
    eps = engine.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["documents"])
    edge_ids = list({json.loads(d)["edge_id"] for d in (eps.get("documents") or [])})
    if not edge_ids:
        return []
    got = engine.edge_collection.get(ids=edge_ids, include=["documents"])
    return [Edge.model_validate_json(j) for j in (got.get("documents") or [])]


# ------------------------
# Concrete proposers
# ------------------------

class VectorProposer(MergeCandidateProposer):
    """
    Existing single-node vector proposer (unchanged behavior from engine.generate_merge_candidates).
    """
    def __init__(self):
        self.limit_per_bucket = 100
    def for_new_node(
        self,
        engine: EngineLike,
        new_node: Node,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Node, Node]]:
        if not new_node.embedding:
            # Skip vector search if no embedding
            return []

        import chromadb

        candidates = []
        for col, NodeOrEdgeModel in [(self.node_collection, Node), (self.edge_collection, Edge)]:
            NodeOrEdgeType = type[Node] | type[Edge]
            col: chromadb.Collection
            NodeOrEdgeModel: NodeOrEdgeType
            results = col.query(
                query_embeddings=[new_node.embedding],
                n_results=top_k
            )
            for idx, doc_json in enumerate(results["documents"][0]):
                score = results["distances"][0][idx]
                if score >= similarity_threshold:
                    existing_node = NodeOrEdgeModel(**json.loads(doc_json))
                    # Don't match against itself
                    if existing_node.id != new_node.id:
                        candidates.append((existing_node, new_node))
        return candidates

    def same_kind_in_doc(self, engine: EngineLike, doc_id: str, kind) -> List[Tuple[Any, Any]]:
        """
        Same-kind candidates (node↔node and edge↔edge) within a document.

        Nodes are bucketed by (type, normalized label).
        Edges are bucketed by derived signature (or properties.signature_text).
        """
        # Nodes
        node_pairs: List[AdjudicationCandidate] = []
        if kind == 'node':
            nodes = _fetch_doc_nodes(engine, doc_id)
            node_buckets: Dict[tuple, List[Node]] = {}
            for n in nodes:
                key = (getattr(n, "type", "entity"), _norm_label(getattr(n, "label", None)))
                node_buckets.setdefault(key, []).append(n)

            
            for items in node_buckets.values():
                    if len(items) > 1:
                        for i in range(len(items)):
                            for j in range(i+1, len(items)):
                                node_pairs.append(AdjudicationCandidate(
                                    left=engine._target_from_node(items[i]),
                                    right=engine._target_from_node(items[j]),
                                    question="same_entity"
                                ))
            
        # Edges
        edge_pairs: List[AdjudicationCandidate] = []
        if kind == 'edge':
            edges = _fetch_doc_edges(engine, doc_id)
            edge_buckets: Dict[str, List[Edge]] = {}
            for e in edges:
                key = _edge_signature(e)
                edge_buckets.setdefault(key, []).append(e)

            
            for items in edge_buckets.values():
                        if len(items) > 1:
                            for i in range(len(items)):
                                for j in range(i+1, len(items)):
                                    edge_pairs.append(AdjudicationCandidate(
                                        left=engine._target_from_edge(items[i]),
                                        right=engine._target_from_edge(items[j]),
                                        question="same_entity"
                                    ))

        
        return list(node_pairs) + list(edge_pairs) # type: ignore for mixed list

    def cross_kind_in_doc(self, engine: EngineLike, doc_id: str) -> List[Tuple[Any, Any]]:
        
        # if not engine.allow_cross_kind_adjudication:
        #     return []

        
        nodes = _fetch_doc_nodes(engine, doc_id)
        edges = _fetch_doc_edges(engine, doc_id)


        # cheap blocking: lowercase tokens of node.label & edge.label/relation
        def toks(s): return set((s or "").lower().split())

        pairs = []
        for n in nodes:
            nt = toks(n.label) | toks(n.summary)
            for e in edges:
                et = toks(e.label) | toks(e.summary) | toks(e.relation)
                if nt and et and (nt & et):
                    pairs.append(AdjudicationCandidate(
                        left=engine._target_from_node(n),
                        right=engine._target_from_edge(e),
                        question="same_entity"
                    ))
                    if len(pairs) >= self.limit_per_bucket:
                        break
        return pairs

class CompositeProposer(MergeCandidateProposer):
    """
    Tiny orchestrator that composes the three flows but keeps the exact logic in VectorProposer.
    """

    def __init__(self, base: Optional[VectorProposer] = None):
        self.base = base or VectorProposer()

    def for_new_node(
        self,
        engine: EngineLike,
        new_node: Node,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Node, Node]]:
        return self.base.for_new_node(engine, new_node, top_k, similarity_threshold)

    def same_kind_in_doc(self, engine: EngineLike, doc_id: str) -> List[Tuple[Any, Any]]:
        return self.base.same_kind_in_doc(engine, doc_id)

    def cross_kind_in_doc(self, engine: EngineLike, doc_id: str) -> List[Tuple[Any, Any]]:
        return self.base.cross_kind_in_doc(engine, doc_id)