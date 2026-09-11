# graph_query.py — polished traversal layer with higher‑level APIs
from __future__ import annotations
from collections import deque
from typing import Dict, Set, List, Optional, Iterable
import json
import asyncio
import inspect

from .engine_core.models import Node, Edge
from .engine_core.async_compat import run_awaitable_blocking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine_core.engine import GraphKnowledgeEngine


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

    def _read_nodes(self, *, ids=None, where=None) -> list[Node]:
        """Read nodes through the engine facade, including staged rows."""
        reader = getattr(self.e, "read", None)
        if reader is not None and callable(getattr(reader, "get_nodes", None)):
            try:
                return list(reader.get_nodes(ids=ids, where=where, include=["documents"]))
            except TypeError:
                # Tiny legacy read shims may expose only ids/include.
                if where is None:
                    return list(reader.get_nodes(ids))
                raw_get = getattr(reader, "_node_get_raw", None)
                if callable(raw_get):
                    got = raw_get(ids=ids, where=where, limit=10000, include=["documents"])
                    return [Node.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]
                return []
        if reader is not None:
            return []
        got = self.e.backend.node_get(ids=ids, where=where, include=["documents"])
        return [Node.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]

    def _read_edges(self, *, ids=None, where=None) -> list[Edge]:
        """Read edges through the engine facade, including staged rows."""
        reader = getattr(self.e, "read", None)
        if reader is not None and callable(getattr(reader, "get_edges", None)):
            try:
                return list(reader.get_edges(ids=ids, where=where, include=["documents"]))
            except TypeError:
                if where is None:
                    return list(reader.get_edges(ids))
                raw_get = getattr(reader, "_edge_get_raw", None)
                if callable(raw_get):
                    got = raw_get(ids=ids, where=where, limit=10000, include=["documents"])
                    return [Edge.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]
                return []
        if reader is not None:
            return []
        got = self.e.backend.edge_get(ids=ids, where=where, include=["documents"])
        return [Edge.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]

    # ---- internals ----
    def _is_node(self, rid: str) -> bool:
        reader = getattr(self.e, "read", None)
        exists = getattr(reader, "node_exists", None)
        if callable(exists):
            return bool(exists(ids=[rid]))
        if reader is not None:
            return bool(self._read_nodes(ids=[rid]))
        hit = self.e.backend.node_get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    def _is_edge(self, rid: str) -> bool:
        reader = getattr(self.e, "read", None)
        exists = getattr(reader, "edge_exists", None)
        if callable(exists):
            return bool(exists(ids=[rid]))
        if reader is not None:
            return bool(self._read_edges(ids=[rid]))
        hit = self.e.backend.edge_get(ids=[rid])
        return (hit.get("ids") or [None])[0] == rid

    def _stage1_endpoint_rows(self, *, edge_id: str | None = None) -> list[dict]:
        """Read pending structural endpoint rows from Stage 1.

        Stage 1 endpoint rows are short-lived structural projection data. They
        are removed after Stage 2 promotion; they are not canonical state.
        """
        if getattr(self.e, "persistence_mode", "single_stage") != "two_stage":
            return []
        adapter = getattr(self.e, "two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if not callable(query):
            adapter = getattr(self.e, "async_two_stage_projection_adapter", None)
            query = getattr(adapter, "stage1_query", None)
        if not callable(query):
            return []
        if inspect.iscoroutinefunction(query):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                # A synchronous traversal must not block an active async loop.
                # Use neighbors_async() for pending async Stage-1 data.
                return []
        try:
            rows = run_awaitable_blocking(
                query(
                    entity_kind="edge",
                    ids=[edge_id] if edge_id else None,
                    limit=10000,
                )
            )
        except Exception:
            rows = []
        if not rows:
            meta = getattr(self.e, "meta_sqlite", None)
            meta_query = getattr(meta, "query_stage1_node_projections", None)
            if callable(meta_query):
                try:
                    rows = meta_query(
                        str(getattr(self.e, "namespace", "default")),
                        entity_kind="edge",
                        ids=[edge_id] if edge_id else None,
                        limit=10000,
                    )
                except Exception:
                    rows = []

        endpoints: list[dict] = []
        for row in rows or []:
            payload = dict(row.get("payload") or row)
            document = payload.get("document")
            if not document:
                continue
            try:
                edge = Edge.model_validate_json(document)
            except Exception:
                continue
            current_edge_id = str(edge.safe_get_id())
            for role, ids, endpoint_type in (
                ("src", edge.source_ids or [], "node"),
                ("tgt", edge.target_ids or [], "node"),
                ("src", getattr(edge, "source_edge_ids", []) or [], "edge"),
                ("tgt", getattr(edge, "target_edge_ids", []) or [], "edge"),
            ):
                for endpoint_id in ids:
                    endpoints.append(
                        {
                            "id": f"{current_edge_id}::{role}::{endpoint_type}::{endpoint_id}",
                            "edge_id": current_edge_id,
                            "endpoint_id": str(endpoint_id),
                            "endpoint_type": endpoint_type,
                            "role": role,
                            "doc_id": edge.doc_id,
                            "relation": edge.relation,
                        }
                    )
        return endpoints

    async def _stage1_endpoint_rows_async(self, *, edge_id: str | None = None) -> list[dict]:
        """Read pending Stage-1 edge payloads without a sync bridge."""
        if getattr(self.e, "persistence_mode", "single_stage") != "two_stage":
            return []
        adapter = getattr(self.e, "async_two_stage_projection_adapter", None)
        query = getattr(adapter, "stage1_query", None)
        if not callable(query):
            return []
        rows = await query(
            entity_kind="edge",
            ids=[edge_id] if edge_id else None,
            limit=10000,
        )
        endpoints: list[dict] = []
        for row in rows or []:
            payload = dict(row.get("payload") or row)
            document = payload.get("document")
            if not document:
                continue
            try:
                edge = Edge.model_validate_json(document)
            except Exception:
                continue
            for role, ids, endpoint_type in (
                ("src", edge.source_ids or [], "node"),
                ("tgt", edge.target_ids or [], "node"),
                ("src", getattr(edge, "source_edge_ids", []) or [], "edge"),
                ("tgt", getattr(edge, "target_edge_ids", []) or [], "edge"),
            ):
                for endpoint_id in ids:
                    endpoints.append({
                        "id": f"{edge.safe_get_id()}::{role}::{endpoint_type}::{endpoint_id}",
                        "edge_id": str(edge.safe_get_id()),
                        "endpoint_id": str(endpoint_id),
                        "endpoint_type": endpoint_type,
                        "role": role,
                        "doc_id": edge.doc_id,
                        "relation": edge.relation,
                    })
        return endpoints

    async def _endpoint_rows_async(self, where: dict, *, edge_id: str | None = None) -> list[dict]:
        backend = getattr(self.e, "backend", None)
        rows: list[dict] = []
        got = await self._async_collection_call(
            "edge_endpoints", "get", where=where, include=["documents"]
        )
        if got is not None:
            rows.extend(
                json.loads(document)
                for document in (got.get("documents") or [])
                if document
            )

        def matches(row: dict) -> bool:
            clauses = where.get("$and", [where]) if isinstance(where, dict) else []
            return all(
                row.get(key) == value
                for clause in clauses if isinstance(clause, dict)
                for key, value in clause.items()
            )

        existing = {
            (str(row.get("edge_id")), str(row.get("endpoint_id")), str(row.get("role")))
            for row in rows
        }
        for row in await self._stage1_endpoint_rows_async(edge_id=edge_id):
            key = (str(row.get("edge_id")), str(row.get("endpoint_id")), str(row.get("role")))
            if matches(row) and key not in existing:
                rows.append(row)
                existing.add(key)
        return rows

    async def _async_collection_call(
        self, collection_key: str, method: str, **kwargs
    ) -> dict | None:
        """Call native async backend verbs without forcing a sync bridge."""
        backend = getattr(self.e, "backend", None)
        async_call = getattr(backend, "async_call", None)
        if callable(async_call):
            result = async_call(collection_key, method, **kwargs)
        else:
            fn = getattr(backend, f"{collection_key}_{method}", None)
            if not callable(fn):
                return None
            result = fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def neighbors_async(
        self, rid: str, *, direction: str = "both", doc_id: Optional[str] = None,
        allow_jump_edge: bool = True,
    ) -> Dict[str, Set[str]]:
        """Async traversal path for async backends and pending Stage-1 rows."""
        if getattr(self.e, "backend", None) is None:
            return self.neighbors(rid, direction=direction, doc_id=doc_id, allow_jump_edge=allow_jump_edge)

        node_hit = await self._async_collection_call(
            "node", "get", ids=[rid], include=["documents"]
        ) or {}
        edge_hit = await self._async_collection_call(
            "edge", "get", ids=[rid], include=["documents"]
        ) or {}
        stage1_rows = await self._stage1_endpoint_rows_async(edge_id=rid)
        adapter = getattr(self.e, "async_two_stage_projection_adapter", None)
        stage1_nodes = await adapter.stage1_query(entity_kind="node", ids=[rid], limit=1) \
            if callable(getattr(adapter, "stage1_query", None)) else []
        is_node = bool((node_hit.get("ids") or [None])[0] == rid)
        is_node = is_node or bool(stage1_nodes)
        is_edge = bool((edge_hit.get("ids") or [None])[0] == rid or any(
            row.get("edge_id") == rid for row in stage1_rows
        ))
        if not (is_node or is_edge):
            return {"nodes": set(), "edges": set()}

        nodes, edges = set(), set()
        if is_node:
            clauses = [{"endpoint_type": "node"}, {"endpoint_id": rid}]
            if doc_id is not None:
                clauses.append({"doc_id": doc_id})
            for row in await self._endpoint_rows_async({"$and": clauses}):
                edges.add(str(row["edge_id"]))
                if allow_jump_edge:
                    for other in await self._endpoint_rows_async(
                        {"edge_id": row["edge_id"]}, edge_id=str(row["edge_id"])
                    ):
                        if other.get("endpoint_type") == "node" and other["endpoint_id"] != rid:
                            nodes.add(str(other["endpoint_id"]))
        if is_edge:
            clauses = [{"edge_id": rid}]
            if direction in ("src", "tgt"):
                clauses.append({"role": direction})
            for row in await self._endpoint_rows_async(
                {"$and": clauses} if len(clauses) > 1 else clauses[0], edge_id=rid
            ):
                if row["endpoint_type"] == "node":
                    nodes.add(str(row["endpoint_id"]))
                elif row["endpoint_type"] == "edge":
                    edges.add(str(row["endpoint_id"]))
        return {"nodes": nodes, "edges": edges}

    def _endpoint_rows(self, where: dict, *, edge_id: str | None = None) -> list[dict]:
        """Return backend endpoint rows plus matching transient Stage-1 rows."""
        rows: list[dict] = []
        try:
            reader = getattr(self.e, "read", None)
            endpoint_get = getattr(reader, "get_edge_endpoints", None)
            if callable(endpoint_get):
                got = endpoint_get(where=where, include=["documents"])
            elif reader is not None:
                got = {"documents": []}
            else:
                got = self.e.backend.edge_endpoints_get(
                    where=where, include=["documents"]
                )
            rows.extend(
                json.loads(document)
                for document in (got.get("documents") or [])
                if document
            )
        except Exception:
            pass

        def matches(row: dict) -> bool:
            clauses = where.get("$and", [where]) if isinstance(where, dict) else []
            for clause in clauses:
                if not isinstance(clause, dict):
                    continue
                for key, value in clause.items():
                    if row.get(key) != value:
                        return False
            return True

        existing = {
            (str(row.get("edge_id")), str(row.get("endpoint_id")), str(row.get("role")))
            for row in rows
        }
        for row in self._stage1_endpoint_rows(edge_id=edge_id):
            if matches(row):
                key = (str(row.get("edge_id")), str(row.get("endpoint_id")), str(row.get("role")))
                if key not in existing:
                    rows.append(row)
                    existing.add(key)
        return rows

    # ---- doc scoping ----
    def nodes_in_doc(self, doc_id: str) -> List[Node]:
        ids = self.e.read.node_ids_by_doc(doc_id)
        return self._read_nodes(ids=ids) if ids else []

    def edges_in_doc(self, doc_id: str) -> List[Edge]:
        ids = self.e.read.edge_ids_by_doc(doc_id)
        return self._read_edges(ids=ids) if ids else []

    def document_subgraph(
        self, doc_id: str, *, center_ids: Optional[Iterable[str]] = None, hops: int = 1
    ) -> Dict[str, List]:
        """Return a small subgraph for a document: seeds + k‑hop neighborhood.
        If center_ids omitted, seeds are all nodes in the doc (bounded by hops=0/1 recommended).
        """
        if center_ids:
            seeds = list(center_ids)
        else:
            seeds = self.e.read.node_ids_by_doc(doc_id)
        layers = self.k_hop(seeds, k=max(0, hops), doc_id=doc_id)
        # Flatten and dedupe
        node_ids: Set[str] = set(seeds)
        edge_ids: Set[str] = set()
        for L in layers:
            node_ids |= set(L["nodes"])  # discovered opposite endpoints
            edge_ids |= set(L["edges"])  # incident edges
        nodes = self.e.read.get_nodes(list(node_ids))
        edges = self.e.read.get_edges(list(edge_ids))
        return {"seed_ids": seeds, "nodes": nodes, "edges": edges, "layers": layers}

    # ---- document summary helpers ----
    def final_summary_node_id(self, doc_id: str) -> Optional[str]:
        """Find the single node that has a 'summarizes_document' edge -> docnode:{doc_id}."""
        tgt = f"docnode:{doc_id}"
        eps = self._endpoint_rows(
            {"$and": [
                {"endpoint_id": tgt},
                {"endpoint_type": "node"},
                {"role": "tgt"},
                {"relation": "summarizes_document"},
            ]}
        )
        eids = {str(row["edge_id"]) for row in eps}
        if not eids:
            return None
        # For each edge, fetch its src node endpoint
        for eid in eids:
            srcs = self._endpoint_rows({"$and": [
                {"edge_id": eid}, {"endpoint_type": "node"}, {"role": "src"}
            ]}, edge_id=eid)
            for row in srcs:
                return row.get("endpoint_id")
        return None

    def final_summary_node(self, doc_id: str) -> Optional[Node]:
        rid = self.final_summary_node_id(doc_id)
        if not rid:
            return None
        nodes = self._read_nodes(ids=[rid]) if rid else []
        return nodes[0] if nodes else None

    # ---- generic traversals ----
    def neighbors(
        self,
        rid: str,
        *,
        direction: str = "both",
        doc_id: Optional[str] = None,
        allow_jump_edge=True,
    ) -> Dict[str, Set[str]]:
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
            q = (
                {"$and": [{"endpoint_type": "node"}, {"endpoint_id": rid}]}
                if doc_id is None
                else {
                    "$and": [
                        {"endpoint_type": "node"},
                        {"endpoint_id": rid},
                        {"doc_id": doc_id},
                    ]
                }
            )
            for row in self._endpoint_rows(q):
                edges.add(row["edge_id"])
                # pull opposite endpoints
                if allow_jump_edge:
                    for r2 in self._endpoint_rows(
                        {"edge_id": row["edge_id"]}, edge_id=row["edge_id"]
                    ):
                        if (
                            r2.get("endpoint_type") == "node"
                            and r2["endpoint_id"] != rid
                        ):
                            nodes.add(r2["endpoint_id"])

        if is_edge:
            clause = [{"edge_id": rid}]
            if direction in ("src", "tgt"):
                clause.append({"role": direction})
            q = {"$and": clause} if len(clause) > 1 else {"edge_id": rid}
            for row in self._endpoint_rows(q, edge_id=rid):
                if row["endpoint_type"] == "node":
                    nodes.add(row["endpoint_id"])
                elif row["endpoint_type"] == "edge":
                    edges.add(row["endpoint_id"])

        return {"nodes": nodes, "edges": edges}

    def k_hop(
        self,
        start_ids: List[str],
        k: int = 2,
        *,
        doc_id: Optional[str] = None,
        allow_jump_edge=False,
    ) -> List[Dict[str, Set[str]]]:
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
                nbrs = self.neighbors(
                    rid, doc_id=doc_id, allow_jump_edge=allow_jump_edge
                )
                layer_nodes |= nbrs["nodes"]
                layer_edges |= nbrs["edges"]
                next_frontier |= nbrs["nodes"] | nbrs["edges"]
            layers.append({"nodes": layer_nodes, "edges": layer_edges})
            frontier = next_frontier - visited
        return layers

    def shortest_path(
        self,
        src_id: str,
        dst_id: str,
        *,
        doc_id: Optional[str] = None,
        max_depth: int = 8,
    ) -> List[str]:
        if src_id == dst_id:
            return [src_id]
        q = deque([(src_id, [src_id])])
        seen = {src_id}
        depth = 0

        while q and depth <= max_depth:
            for _ in range(len(q)):
                cur, path = q.popleft()
                nbrs = self.neighbors(cur, doc_id=doc_id, allow_jump_edge=False)
                for v in nbrs["nodes"] | nbrs["edges"]:
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
        nodes = self._read_nodes(where=(where or None))
        out: List[str] = []
        for n in nodes:
            nid = n.safe_get_id()
            if not nid:
                continue
            if type and (n.type != type):
                continue
            if label_contains and (
                label_contains.lower() not in (n.label or "").lower()
            ):
                continue
            if summary_contains and (
                summary_contains.lower() not in (n.summary or "").lower()
            ):
                continue
            out.append(nid)
            if len(out) >= max(1, limit):
                break
        return out

    def path_between_labels(
        self,
        src_substr: str,
        dst_substr: str,
        *,
        doc_id: Optional[str] = None,
        max_depth: int = 8,
    ) -> List[str]:
        """Find a shortest path between any node whose label contains src_substr and any whose label contains dst_substr."""
        src_candidates = self.search_nodes(
            label_contains=src_substr, doc_id=doc_id, limit=50
        )
        dst_candidates = set(
            self.search_nodes(label_contains=dst_substr, doc_id=doc_id, limit=50)
        )
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
            where = {"$and": [{k: v} for k, v in where.items()]}
        edges = self._read_edges(where=where)
        out: List[str] = []
        for e in edges:
            eid = e.safe_get_id()
            if not eid:
                continue

            ok_src = src_label_contains is None
            ok_tgt = tgt_label_contains is None
            if src_label_contains or tgt_label_contains:
                src_labels = [n.label for n in self._read_nodes(ids=e.source_ids or [])]
                tgt_labels = [n.label for n in self._read_nodes(ids=e.target_ids or [])]
                if src_label_contains:
                    ok_src = any(
                        src_label_contains.lower() in (s or "").lower()
                        for s in src_labels
                    )
                if tgt_label_contains:
                    ok_tgt = any(
                        tgt_label_contains.lower() in (t or "").lower()
                        for t in tgt_labels
                    )

            if ok_src and ok_tgt:
                out.append(eid)
        return out

    def adjacency_list(
        self, node_ids: Iterable[str], *, doc_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Set[str]]]:
        """For each node id, return {node_id: {"nodes": set(), "edges": set()}}"""
        out: Dict[str, Dict[str, Set[str]]] = {}
        for nid in node_ids:
            out[nid] = self.neighbors(nid, doc_id=doc_id)
        return out

    # ---- semantic seed ----
    def semantic_seed_then_expand(
        self, query_embedding: List[float], *, top_k: int = 5, hops: int = 1
    ):
        query_reader = getattr(getattr(self.e, "read", None), "query_nodes", None)
        hits = (
            query_reader(query_embeddings=[query_embedding], n_results=top_k)
            if callable(query_reader)
            else [] if getattr(self.e, "read", None) is not None else self.e.backend.node_query(
                query_embeddings=[query_embedding], n_results=top_k
            )
        )
        if isinstance(hits, list):
            seed_ids = [str(node.safe_get_id()) for node in hits[0]] if hits else []
        else:
            seed_ids = [nid for nid in (hits.get("ids") or [[]])[0]]
        layers = self.k_hop(seed_ids, k=hops)
        return {"seeds": seed_ids, "layers": layers}

    def semantic_seed_then_expand_text(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        hops: int = 1,
        doc_ids=None,
        where=None,
    ):
        """Seed by a TEXT query using the collection's default embedding function, then expand K hops.
        This avoids any custom embedding pipeline and uses the underlying vector store's default embeddings.
        """
        _where = {"doc_id": doc_ids} if type(doc_ids) is str else None
        if type(doc_ids) is list:
            if _where is None:
                _where = {}
            _where["doc_id"] = {"$in": doc_ids}
        if where:
            if _where:
                if _where.get("and"):
                    if type(_where["and"]) is list:
                        _where_and: list = _where["and"]
                        _where_and.append(where)
                    else:
                        raise SyntaxError(
                            "vector backend syntax error: where invalid syntax"
                        )
                    # _where['and'].append()
                else:
                    _where = {"$and": [where, _where]}
        query_reader = getattr(getattr(self.e, "read", None), "query_nodes", None)
        hits = (
            query_reader(query=query_text, n_results=top_k, where=_where)
            if callable(query_reader)
            else [] if getattr(self.e, "read", None) is not None else self.e.backend.node_query(
                query_texts=[query_text], n_results=top_k, where=_where
            )
        )
        if isinstance(hits, list):
            seed_ids = [str(node.safe_get_id()) for node in hits[0]] if hits else []
            seed_docs = [node.model_dump_json() for node in hits[0]] if hits else []
        else:
            seed_ids = [nid for nid in (hits.get("ids") or [[]])[0] if nid]
            seed_docs = hits.get("documents", [[]])[0]
        layers = self.k_hop(seed_ids, k=hops)
        out_layers = [
            {
                "nodes": [n.model_dump_json() for n in self._read_nodes(ids=list(l["nodes"]))]
                if l["nodes"]
                else [],
                "edges": [e.model_dump_json() for e in self._read_edges(ids=list(l["edges"]))]
                if l["edges"]
                else [],
            }
            for l in layers
        ]
        res = {"seeds": seed_docs, "layers": out_layers}
        return res
