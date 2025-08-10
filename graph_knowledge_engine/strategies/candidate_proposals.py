# strategies/candidate_proposals.py
from __future__ import annotations
from typing import List, Tuple, Optional
from ..models import Node
import json

def by_vector_similarity(engine, new_node: Node, top_k: int = 8, similarity_threshold: float = 0.85) -> List[Tuple[Node, Node]]:
    """Default: vector NN on the node collection."""
    if not new_node.embedding:
        return []
    res = engine.node_collection.query(query_embeddings=[new_node.embedding], n_results=top_k, include=["documents","distances"])
    pairs: List[Tuple[Node, Node]] = []
    if not res or not res.get("documents"):
        return pairs
    for doc_json, dist in zip(res["documents"][0], res.get("distances", [[0]*top_k])[0]):
        if dist >= similarity_threshold:
            existing = Node.model_validate_json(doc_json)
            if existing.id != new_node.id:
                pairs.append((existing, new_node))
    return pairs

def by_label_bucket(engine, new_node: Node, top_k: int = 50) -> List[Tuple[Node, Node]]:
    """Cheap text bucket: same normalized label+type within the same doc (or same domain)."""
    key = (new_node.type, new_node.label.strip().lower())
    where = {"type": new_node.type, "label": new_node.label}  # requires label/type mirrored in metadata
    res = engine.node_collection.get(where=where, include=["documents"])
    pairs: List[Tuple[Node, Node]] = []
    for doc_json in (res.get("documents") or [])[:top_k]:
        existing = Node.model_validate_json(doc_json)
        if existing.id != new_node.id:
            pairs.append((existing, new_node))
    return pairs

def hybrid(engine, new_node: Node, top_k: int = 8, threshold: float = 0.85) -> List[Tuple[Node, Node]]:
    """Union of vector + label buckets (dedup by ids)."""
    seen = set()
    out: List[Tuple[Node, Node]] = []
    for pair in by_vector_similarity(engine, new_node, top_k=top_k, similarity_threshold=threshold) + \
                by_label_bucket(engine, new_node, top_k=max(2*top_k, 50)):
        key = (pair[0].id, pair[1].id)
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out