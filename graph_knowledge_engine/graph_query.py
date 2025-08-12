from __future__ import annotations
from collections import deque
from typing import Dict, Set, List, Optional, Tuple
import json

from .models import Node, Edge

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import GraphKnowledgeEngine

class GraphQuery:
    """
    Thin traversal/search layer built on top of GraphKnowledgeEngine's Chroma collections.

    Example
    -------
    >>> gq = GraphQuery(engine)
    >>> nbrs = gq.neighbors(node_id)
    >>> layers = gq.k_hop([node_id], k=2, doc_id=doc_id)
    >>> path = gq.shortest_path(node_a, node_b, max_depth=6)
    >>> eids = gq.find_edges(relation="causes", src_label_contains="smoking")
    """

    def __init__(self, engine: "GraphKnowledgeEngine"):
        self.e = engine

    # ---------- Internals ----------
    def _is_node(self, rid: str) -> bool:
        hit = self.e.node_collection.get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    def _is_edge(self, rid: str) -> bool:
        hit = self.e.edge_collection.get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    # ---------- Public API ----------
    def neighbors(self, rid: str, *, direction: str = "both", doc_id: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        Return neighbors for a node-id or edge-id.
        For node-id: neighbors are incident edges and opposite endpoint nodes.
        For edge-id: neighbors are endpoint nodes and meta-edges (if any).
        direction: "src"|"tgt"|"both" (when rid is an edge).
        """
        is_node = self._is_node(rid)
        is_edge = self._is_edge(rid)
        if not (is_node or is_edge):
            return {"nodes": set(), "edges": set()}

        nodes, edges = set(), set()
        if is_node:
            q = {"node_id": rid} if doc_id is None else {"$and": [{"node_id": rid}, {"doc_id": doc_id}]}
            eps = self.e.edge_endpoints_collection.get(where=q, include=["documents"])
            for d in eps.get("documents") or []:
                row = json.loads(d)
                edges.add(row["edge_id"])
                # pull opposite endpoints
                eps2 = self.e.edge_endpoints_collection.get(where={"edge_id": row["edge_id"]}, include=["documents"])
                for d2 in eps2.get("documents") or []:
                    r2 = json.loads(d2)
                    if r2.get("endpoint_type") == "node" and r2["endpoint_id"] != rid:
                        nodes.add(r2["endpoint_id"])

        if is_edge:
            clause = [{"edge_id": rid}]
            if direction in ("src", "tgt"):
                clause.append({"role": direction})
            q = {"$and": clause} if len(clause) > 1 else {"edge_id": rid}
            eps = self.e.edge_endpoints_collection.get(where=q, include=["documents"])
            for d in eps.get("documents") or []:
                row = json.loads(d)
                if row["endpoint_type"] == "node":
                    nodes.add(row["endpoint_id"])
                elif row["endpoint_type"] == "edge":
                    edges.add(row["endpoint_id"])

        return {"nodes": nodes, "edges": edges}

    def k_hop(self, start_ids: List[str], k: int = 2, *, doc_id: Optional[str] = None) -> List[Dict[str, Set[str]]]:
        """BFS k-hop expansion over nodes+edges. Returns per-hop {'nodes', 'edges'} sets."""
        visited: Set[str] = set()
        frontier: Set[str] = set(start_ids)
        layers: List[Dict[str, Set[str]]] = []

        for _ in range(max(0, k)):
            next_frontier: Set[str] = set()
            layer_nodes, layer_edges = set(), set()
            for rid in frontier:
                if rid in visited:
                    continue
                visited.add(rid)
                nbrs = self.neighbors(rid, doc_id=doc_id)
                layer_nodes |= nbrs["nodes"]
                layer_edges |= nbrs["edges"]
                next_frontier |= nbrs["nodes"] | nbrs["edges"]
            layers.append({"nodes": layer_nodes, "edges": layer_edges})
            frontier = next_frontier - visited
        return layers

    def shortest_path(self, src_id: str, dst_id: str, *, doc_id: Optional[str] = None, max_depth: int = 8) -> List[str]:
        """Unweighted BFS shortest path in the mixed graph; [] if not found."""
        if src_id == dst_id:
            return [src_id]
        q = deque([(src_id, [src_id])])
        seen = {src_id}
        depth = 0
        while q and depth <= max_depth:
            for _ in range(len(q)):
                cur, path = q.popleft()
                nbrs = self.neighbors(cur, doc_id=doc_id)
                for v in (nbrs["nodes"] | nbrs["edges"]):
                    if v in seen:
                        continue
                    if v == dst_id:
                        return path + [v]
                    seen.add(v)
                    q.append((v, path + [v]))
            depth += 1
        return []

    def find_edges(
        self,
        *,
        relation: Optional[str] = None,
        src_label_contains: Optional[str] = None,
        tgt_label_contains: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[str]:
        """
        Filter edges by relation/doc and post-filter by endpoint node labels.
        """
        where = {}
        if relation:
            where["relation"] = relation
        if doc_id:
            where["doc_id"] = doc_id

        edges = self.e.edge_collection.get(where=where, include=["ids", "documents"])
        out: List[str] = []
        for eid, edoc in zip(edges.get("ids") or [], edges.get("documents") or []):
            if not eid or not edoc:
                continue
            e = Edge.model_validate_json(edoc)

            ok_src = src_label_contains is None
            ok_tgt = tgt_label_contains is None
            if (src_label_contains or tgt_label_contains):
                srcs = self.e.node_collection.get(ids=e.source_ids or [], include=["documents"])
                tgts = self.e.node_collection.get(ids=e.target_ids or [], include=["documents"])
                src_labels = [Node.model_validate_json(j).label for j in (srcs.get("documents") or []) if j]
                tgt_labels = [Node.model_validate_json(j).label for j in (tgts.get("documents") or []) if j]
                if src_label_contains:
                    ok_src = any(src_label_contains.lower() in (s or "").lower() for s in src_labels)
                if tgt_label_contains:
                    ok_tgt = any(tgt_label_contains.lower() in (t or "").lower() for t in tgt_labels)

            if ok_src and ok_tgt:
                out.append(eid)
        return out

    def semantic_seed_then_expand(self, query_embedding: List[float], *, top_k: int = 5, hops: int = 1):
        """
        1) vector search seeds (nodes)
        2) k-hop expansion to pull structural context
        """
        hits = self.e.node_collection.query(query_embeddings=[query_embedding], n_results=top_k)
        seed_ids = [nid for nid in (hits.get("ids") or [[]])[0]]
        layers = self.k_hop(seed_ids, k=hops)
        return {"seeds": seed_ids, "layers": layers}