# graph_query.py — polished traversal layer with higher‑level APIs
from __future__ import annotations
from collections import deque
from typing import Dict, Set, List, Optional, Tuple, Iterable
import json

from .models import Node, Edge
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import GraphKnowledgeEngine


class GraphQuery:
    """
    Thin traversal/search layer on top of GraphKnowledgeEngine's Chroma collections.

    Existing API (kept):
      - neighbors(rid, direction="both", doc_id=None) -> {"nodes", "edges"}
      - k_hop(start_ids, k=2, doc_id=None)
      - shortest_path(src_id, dst_id, doc_id=None, max_depth=8)
      - find_edges(relation=None, src_label_contains=None, tgt_label_contains=None, doc_id=None)
      - semantic_seed_then_expand(query_embedding, top_k=5, hops=1)
      - semantic_seed_then_expand_text(query_text, top_k=5, hops=1)

    New higher‑level helpers:
      - nodes_in_doc(doc_id) / edges_in_doc(doc_id)
      - document_subgraph(doc_id, center_ids=None, hops=1)
      - final_summary_node_id(doc_id) and final_summary_node(doc_id)
      - search_nodes(label_contains=None, summary_contains=None, type=None, doc_id=None, limit=200)
      - path_between_labels(src_substr, dst_substr, doc_id=None, max_depth=8)
      - adjacency_list(node_ids: Iterable[str], doc_id=None)
    """

    # ---- construction ----
    def __init__(self, engine: "GraphKnowledgeEngine"):
        self.e = engine

    # ---- internals ----
    def _is_node(self, rid: str) -> bool:
        hit = self.e.backend.node_get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    def _is_edge(self, rid: str) -> bool:
        hit = self.e.backend.edge_get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    # ---- doc scoping ----
    def nodes_in_doc(self, doc_id: str) -> List[Node]:
        ids = self.e._nodes_by_doc(doc_id)
        got = self.e.backend.node_get(ids=ids, include=["documents"]) if ids else {"documents": []}
        return [Node.model_validate_json(d) for d in (got.get("documents") or []) if d]

    def edges_in_doc(self, doc_id: str) -> List[Edge]:
        ids = self.e._edge_ids_by_doc(doc_id)
        got = self.e.backend.edge_get(ids=ids, include=["documents"]) if ids else {"documents": []}
        return [Edge.model_validate_json(d) for d in (got.get("documents") or []) if d]

    def document_subgraph(self, doc_id: str, *, center_ids: Optional[Iterable[str]] = None, hops: int = 1) -> Dict[str, List]:
        """Return a small subgraph for a document: seeds + k‑hop neighborhood.
        If center_ids omitted, seeds are all nodes in the doc (bounded by hops=0/1 recommended).
        """
        if center_ids:
            seeds = list(center_ids)
        else:
            seeds = self.e._nodes_by_doc(doc_id)
        layers = self.k_hop(seeds, k=max(0, hops), doc_id=doc_id)
        # Flatten and dedupe
        node_ids: Set[str] = set(seeds)
        edge_ids: Set[str] = set()
        for L in layers:
            node_ids |= set(L["nodes"])  # discovered opposite endpoints
            edge_ids |= set(L["edges"])  # incident edges
        nodes = self.e.get_nodes(list(node_ids))
        edges = self.e.get_edges(list(edge_ids))
        return {"seed_ids": seeds, "nodes": nodes, "edges": edges, "layers": layers}

    # ---- document summary helpers ----
    def final_summary_node_id(self, doc_id: str) -> Optional[str]:
        """Find the single node that has a 'summarizes_document' edge -> docnode:{doc_id}."""
        tgt = f"docnode:{doc_id}"
        eps = self.e.backend.edge_endpoints_get(
            where={"$and": [
                {"endpoint_id": tgt}, {"endpoint_type": "node"}, {"role": "tgt"}, {"relation": "summarizes_document"}
            ]},
            include=["documents"],
        )
        eids = {json.loads(d)["edge_id"] for d in (eps.get("documents") or [])}
        if not eids:
            return None
        # For each edge, fetch its src node endpoint
        for eid in eids:
            srcs = self.e.backend.edge_endpoints_get(
                where={"$and": [
                    {"edge_id": eid}, {"endpoint_type": "node"}, {"role": "src"}
                ]},
                include=["documents"],
            )
            for d in (srcs.get("documents") or []):
                row = json.loads(d)
                return row.get("endpoint_id")
        return None

    def final_summary_node(self, doc_id: str) -> Optional[Node]:
        rid = self.final_summary_node_id(doc_id)
        if not rid:
            return None
        got = self.e.backend.node_get(ids=[rid], include=["documents"]) if rid else {"documents": []}
        if got.get("documents"):
            return Node.model_validate_json(got["documents"][0])
        return None

    # ---- generic traversals ----
    def neighbors(self, rid: str, *, direction: str = "both", doc_id: Optional[str] = None, allow_jump_edge = True) -> Dict[str, Set[str]]:
        """
        For a node-id: neighbors are incident edges and opposite endpoint nodes.
        For an edge-id: neighbors are endpoint nodes and meta-edges (if any).
        direction: "src"|"tgt"|"both" (when rid is an edge).
        """
        is_node = self._is_node(rid)
        is_edge = self._is_edge(rid)
        if not (is_node or is_edge):
            return {"nodes": set(), "edges": set()}
        
        nodes, edges = set(), set()
        if is_node:
            q = {"$and": [{"endpoint_type": 'node'},{"endpoint_id": rid}]} if doc_id is None else {"$and": [{"endpoint_type": 'node'},{"endpoint_id": rid}, {"doc_id": doc_id}]}
            eps = self.e.backend.edge_endpoints_get(where=q, include=["documents"])
            for d in eps.get("documents") or []:
                row = json.loads(d)
                edges.add(row["edge_id"])
                # pull opposite endpoints
                if allow_jump_edge:
                    eps2 = self.e.backend.edge_endpoints_get(where={"edge_id": row["edge_id"]}, include=["documents"])
                    for d2 in eps2.get("documents") or []:
                        r2 = json.loads(d2)
                        if r2.get("endpoint_type") == "node" and r2["endpoint_id"] != rid:
                            nodes.add(r2["endpoint_id"])

        if is_edge:
            clause = [{"edge_id": rid}]
            if direction in ("src", "tgt"):
                clause.append({"role": direction})
            q = {"$and": clause} if len(clause) > 1 else {"edge_id": rid}
            eps = self.e.backend.edge_endpoints_get(where=q, include=["documents"])
            for d in eps.get("documents") or []:
                row = json.loads(d)
                if row["endpoint_type"] == "node":
                    nodes.add(row["endpoint_id"])
                elif row["endpoint_type"] == "edge":
                    edges.add(row["endpoint_id"])

        return {"nodes": nodes, "edges": edges}

    def k_hop(self, start_ids: List[str], k: int = 2, *, doc_id: Optional[str] = None , allow_jump_edge=False) -> List[Dict[str, Set[str]]]:
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
                nbrs = self.neighbors(rid, doc_id=doc_id, allow_jump_edge = allow_jump_edge)
                layer_nodes |= nbrs["nodes"]
                layer_edges |= nbrs["edges"]
                next_frontier |= nbrs["nodes"] | nbrs["edges"]
            layers.append({"nodes": layer_nodes, "edges": layer_edges})
            frontier = next_frontier - visited
        return layers

    def shortest_path(self, src_id: str, dst_id: str, *, doc_id: Optional[str] = None, max_depth: int = 8) -> List[str]:
        if src_id == dst_id:
            return [src_id]
        q = deque([(src_id, [src_id])])
        seen = {src_id}
        depth = 0

        while q and depth <= max_depth:
            for _ in range(len(q)):
                cur, path = q.popleft()
                nbrs = self.neighbors(cur, doc_id=doc_id, allow_jump_edge= False)
                for v in (nbrs["nodes"] | nbrs["edges"]):
                    if v in seen:
                        continue
                    if v == dst_id:
                        return path + [v]
                    seen.add(v)
                    q.append((v, path + [v]))
            depth += 1
        return []

    # ---- search helpers ----
    def search_nodes(
        self,
        *,
        label_contains: Optional[str] = None,
        summary_contains: Optional[str] = None,
        type: Optional[str] = None,
        doc_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[str]:
        """Return node IDs filtered by simple metadata and post‑filtered by JSON fields."""
        where = {}
        if doc_id:
            where["doc_id"] = doc_id
        # Pull candidate set by doc scope, then filter by JSON to avoid over‑constraining Chroma metadata
        got = self.e.backend.node_get(where=(where or None), include=["documents"])
        out: List[str] = []
        for nid, ndoc in zip(got.get("ids") or [], got.get("documents") or []):
            if not nid or not ndoc:
                continue
            n = Node.model_validate_json(ndoc)
            if type and (n.type != type):
                continue
            if label_contains and (label_contains.lower() not in (n.label or "").lower()):
                continue
            if summary_contains and (summary_contains.lower() not in (n.summary or "").lower()):
                continue
            out.append(nid)
            if len(out) >= max(1, limit):
                break
        return out

    def path_between_labels(self, src_substr: str, dst_substr: str, *, doc_id: Optional[str] = None, max_depth: int = 8) -> List[str]:
        """Find a shortest path between any node whose label contains src_substr and any whose label contains dst_substr."""
        src_candidates = self.search_nodes(label_contains=src_substr, doc_id=doc_id, limit=50)
        dst_candidates = set(self.search_nodes(label_contains=dst_substr, doc_id=doc_id, limit=50))
        best: List[str] = []
        for s in src_candidates:
            for t in dst_candidates:
                p = self.shortest_path(s, t, doc_id=doc_id, max_depth=max_depth)
                if p and (not best or len(p) < len(best)):
                    best = p
        return best

    def find_edges(
        self,
        *,
        relation: Optional[str] = None,
        src_label_contains: Optional[str] = None,
        tgt_label_contains: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[str]:
        where = {}
        if relation:
            where["relation"] = relation
        if doc_id:
            where["doc_id"] = doc_id
        if not (where):
            where = None
        elif len(where) > 1:
            where = {"$and": [{k:v} for k,v in where.items()]}
        edges = self.e.backend.edge_get(where=where, include=["documents"])
        out: List[str] = []
        for eid, edoc in zip(edges.get("ids") or [], edges.get("documents") or []):
            if not eid or not edoc:
                continue
            e = Edge.model_validate_json(edoc)

            ok_src = src_label_contains is None
            ok_tgt = tgt_label_contains is None
            if (src_label_contains or tgt_label_contains):
                srcs = self.e.backend.node_get(ids=e.source_ids or [], include=["documents"])
                tgts = self.e.backend.node_get(ids=e.target_ids or [], include=["documents"])
                src_labels = [Node.model_validate_json(j).label for j in (srcs.get("documents") or []) if j]
                tgt_labels = [Node.model_validate_json(j).label for j in (tgts.get("documents") or []) if j]
                if src_label_contains:
                    ok_src = any(src_label_contains.lower() in (s or "").lower() for s in src_labels)
                if tgt_label_contains:
                    ok_tgt = any(tgt_label_contains.lower() in (t or "").lower() for t in tgt_labels)

            if ok_src and ok_tgt:
                out.append(eid)
        return out

    def adjacency_list(self, node_ids: Iterable[str], *, doc_id: Optional[str] = None) -> Dict[str, Dict[str, Set[str]]]:
        """For each node id, return {node_id: {"nodes": set(), "edges": set()}}"""
        out: Dict[str, Dict[str, Set[str]]] = {}
        for nid in node_ids:
            out[nid] = self.neighbors(nid, doc_id=doc_id)
        return out

    # ---- semantic seed ----
    def semantic_seed_then_expand(self, query_embedding: List[float], *, top_k: int = 5, hops: int = 1):
        hits = self.e.backend.node_query(query_embeddings=[query_embedding], n_results=top_k)
        seed_ids = [nid for nid in (hits.get("ids") or [[]])[0]]
        layers = self.k_hop(seed_ids, k=hops)
        return {"seeds": seed_ids, "layers": layers}
    def semantic_seed_then_expand_text(self, query_text: str, *, top_k: int = 5, hops: int = 1, doc_ids = None, where = None):
        """Seed by a TEXT query using the collection's default embedding function, then expand K hops.
        This avoids any custom embedding pipeline and uses the underlying vector store's default embeddings.
        """
        _where = {"doc_id" : doc_ids} if type(doc_ids) is str else None
        if type(doc_ids) is list:
            if _where is None:
                _where = {}
            _where['doc_id'] = {"$in": doc_ids}
        if where:
            if _where:
                if _where.get("and"):
                    if type(_where['and']) is list:
                        _where_and : list = _where['and']
                        _where_and.append(where)
                    else:
                        raise SyntaxError("vector backend syntax error: where invalid syntax")
                    # _where['and'].append()
                else:
                    _where = {"$and": [where, _where]}
        hits = self.e.backend.node_query(query_texts=[query_text], n_results=top_k, where = _where)
        seed_ids = [nid for nid in (hits.get("ids") or [[]])[0] if nid]
        layers = self.k_hop(seed_ids, k=hops)
        out_layers = [{'nodes': self.e.backend.node_get(ids=list(l['nodes']))['documents'] if l['nodes'] else [], 
          'edges':  self.e.backend.edge_get(ids=list(l['edges']))['documents'] if l['edges'] else []} for l in layers]
        res =  {"seeds": hits['documents'][0], "layers": out_layers}
        return res