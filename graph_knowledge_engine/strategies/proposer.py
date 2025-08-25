# proposer.py
from __future__ import annotations

from typing import (
    List, Tuple, Optional, Iterable, Literal, Any, Sequence, Union, Dict, Set
)
from dataclasses import dataclass
import json

from ..models import Node, Edge
from .types import MergeCandidateProposer, EngineLike  # your existing protocol types

PairKind = Literal["node_node", "edge_edge", "node_edge", "any_any", "any_node", "any_edge", "node_any", "edge_any"]

@dataclass(frozen=True)
class _Pair:
    left: Any
    right: Any

def _and_where(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a ∧ b in Chroma 'where' syntax; tolerate None."""
    if a and b:
        return {"$and": [a, b]}
    return a or b

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _edge_sig(e: Edge) -> tuple:
    """Very light signature for equality-ish edge pairing."""
    rel = _norm(getattr(e, "relation", None))
    s = tuple(sorted((getattr(e, "source_ids", None) or [])))
    t = tuple(sorted((getattr(e, "target_ids", None) or [])))
    se = tuple(sorted((getattr(e, "source_edge_ids", None) or [])))
    te = tuple(sorted((getattr(e, "target_edge_ids", None) or [])))
    return (rel, s, t, se, te)

def _first_doc_id(engine: EngineLike, obj_id: str, kind: Literal["node", "edge"]) -> Optional[str]:
    """Infer a primary doc_id for an entity id using your existing indices."""
    try:
        if kind == "node":
            rows = engine.node_docs_collection.get(where={"node_id": obj_id}, include=["metadatas"])
            metas = rows.get("metadatas") or []
            if metas and metas[0] and metas[0].get("doc_id"):
                return metas[0]["doc_id"]
            rec = engine.node_collection.get(ids=[obj_id], include=["metadatas"])
            md = (rec.get("metadatas") or [None])[0] or {}
            if md.get("doc_id"):
                return md["doc_id"]
            doc_ids = md.get("doc_ids")
            if isinstance(doc_ids, list) and doc_ids:
                return doc_ids[0]
            return None
        else:
            rows = engine.edge_endpoints_collection.get(where={"edge_id": obj_id}, include=["metadatas"])
            metas = rows.get("metadatas") or []
            if metas and metas[0] and metas[0].get("doc_id"):
                return metas[0]["doc_id"]
            rec = engine.edge_collection.get(ids=[obj_id], include=["metadatas"])
            md = (rec.get("metadatas") or [None])[0] or {}
            return md.get("doc_id")
    except Exception:
        return None

def _load_nodes(engine: EngineLike, ids: Sequence[str]) -> List[Node]:
    if not ids:
        return []
    got = engine.node_collection.get(ids=list(ids), include=["documents"])
    docs = got.get("documents") or []
    out: List[Node] = []
    for d in docs:
        try:
            out.append(Node.model_validate_json(d))
        except Exception:
            pass
    return out

def _load_edges(engine: EngineLike, ids: Sequence[str]) -> List[Edge]:
    if not ids:
        return []
    got = engine.edge_collection.get(ids=list(ids), include=["documents"])
    docs = got.get("documents") or []
    out: List[Edge] = []
    for d in docs:
        try:
            out.append(Edge.model_validate_json(d))
        except Exception:
            pass
    return out

def _coerce_query_nodes(engine: EngineLike, spec: Union[Node, str, Sequence[Union[Node, str]]], is_edge = False) -> List[Node]:
    """Accept Node | id | list[...] and return concrete Node objects (with embeddings)."""
    if isinstance(spec, (Node, str)):
        specs = [spec]
    else:
        specs = list(spec)

    nodes: List[Node] = []
    need_fetch_ids: List[str] = []
    for item in specs:
        if isinstance(item, Node):
            nodes.append(item)
        elif isinstance(item, str):
            need_fetch_ids.append(item)

    if need_fetch_ids:
        if is_edge:
            got = engine.node_collection.get(ids=need_fetch_ids, include=["documents"])
        else:
            got = engine.edge_collection.get(ids=need_fetch_ids, include=["documents"])
        for dj in (got.get("documents") or []):
            try:
                nodes.append(Node.model_validate_json(dj))
            except Exception:
                pass

    # keep only those that actually have embeddings
    return [n for n in nodes if getattr(n, "embedding", None)]

class VectorProposer(MergeCandidateProposer):
    """
    Minimal refactor + batch vector search.

    generate_merge_candidates(...) now supports:
      - new_node as Node|id|List[Node|id]
      - batch Chroma query on nodes (and optionally edges)
      - a 'where' filter that flows to Chroma, e.g. {"insertion_method": "graph_extractor"}
    """
    def __init__(self, engine: EngineLike):
        self.e = engine
        self.limit_per_bucket = 100

    def generate_merge_candidates(
        self,
        engine: EngineLike,
        new_node: Optional[Union[Node, str, Sequence[Union[Node, str]]]],
        new_edge: Optional[Union[Node, str, Sequence[Union[Node, str]]]],
        top_k: int = 10,
        *,
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
        score_mode: Literal["distance", "similarity"] = "distance",
        max_distance: float = 0.25,
        min_similarity: float = 0.85,
        include_nodes: bool = True,
        include_edges: bool = True,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Node, Any]]:
        """
        Batch vector search for one or many *new* entities (nodes and/or edges)
        against existing nodes (and edges if enabled).
        Returns list of (query_entity, matched_entity) where matched_entity ∈ {Node, Edge}.
        """
        engine = self.e or engine
        if not( include_nodes or include_edges):
            return []
        # ---- seed query IDs when caller didn't specify --------------------------------
        if new_node is None:
            if anchor_doc_id:
                new_node = engine.node_ids_by_doc(anchor_doc_id)
            elif allowed_docs:
                new_node = [nid for d in allowed_docs for nid in engine.node_ids_by_doc(d)]
            else:
                new_node = (engine.node_collection.get(include=["ids"]) or {}).get("ids", [])  # all nodes

        if new_edge is None:
            if anchor_doc_id:
                new_edge = engine.edge_ids_by_doc(anchor_doc_id)  # <-- fixed (was node_ids_by_doc)
            elif allowed_docs:
                new_edge = [eid for d in allowed_docs for eid in engine.edge_ids_by_doc(d)]
            else:
                new_edge = (engine.edge_collection.get(include=["ids"]) or {}).get("ids", [])  # all edges
        if not (new_node or new_edge):
            return []
        # ---- tiny local coercers (ids -> objects) -------------------------------------
        def _coerce_query_nodes(e: EngineLike, xs) -> List[Node]:
            if xs is None:
                return []
            if isinstance(xs, (str, Node)):
                xs = [xs]
            out: List[Node] = []
            for item in xs:
                if isinstance(item, Node):
                    out.append(item)
                else:
                    got = e.node_collection.get(ids=[item], include=["documents"])
                    dj = (got.get("documents") or [None])[0]
                    if dj:
                        try:
                            out.append(Node.model_validate_json(dj))
                        except Exception:
                            try:
                                out.append(Node(**json.loads(dj)))
                            except Exception:
                                pass
            return out

        def _coerce_query_edges(e: EngineLike, xs) -> List[Edge]:
            if xs is None:
                return []
            if isinstance(xs, (str, Edge)):
                xs = [xs]
            out: List[Edge] = []
            for item in xs:
                if isinstance(item, Edge):
                    out.append(item)
                else:
                    got = e.edge_collection.get(ids=[item], include=["documents"])
                    dj = (got.get("documents") or [None])[0]
                    if dj:
                        try:
                            out.append(Edge.model_validate_json(dj))
                        except Exception:
                            try:
                                out.append(Edge(**json.loads(dj)))
                            except Exception:
                                pass
            return out

        # ---- build query set: nodes + edges ------------------------------------------
        q_nodes: List[Node] = _coerce_query_nodes(engine, new_node)
        q_edges: List[Edge] = _coerce_query_edges(engine, new_edge)
        queries: List[Node] = q_nodes + q_edges  # Edge ⊂ Node, so type is fine

        if not queries:
            return []

        # embeddings for queries (assume already present on objects)
        q_embs = [q.embedding for q in queries]

        # ---- global metadata filter for the retrieval corpus --------------------------
        extra = None
        if allowed_docs:
            # If you denormalize doc_id in collection metadata, this narrows the search corpus.
            extra = {"doc_id": {"$in": allowed_docs}}
        cand_where = _and_where(where, extra)

        # ---- doc rules (post-filter) --------------------------------------------------
        def _pass_doc_rules(lhs_doc: Optional[str], rhs_id: str, rhs_kind: Literal["node", "edge"]) -> bool:
            rhs_doc = _first_doc_id(engine, rhs_id, rhs_kind)
            if allowed_docs and (rhs_doc not in allowed_docs):
                return False
            if cross_doc_only and lhs_doc and rhs_doc and lhs_doc == rhs_doc:
                return False
            if anchor_doc_id and anchor_only and not ((lhs_doc == anchor_doc_id) or (rhs_doc == anchor_doc_id)):
                return False
            return True

        
        if include_nodes:
            # ---- search nodes given node and/or edge embedding in a list---------------------------------------
            
            node_ref_result = engine.node_refs_collection.get(where=cand_where)
            ok_node_ids = set((i['node_id']) for i in node_ref_result['metadatas']) 
            node_results = engine.node_collection.query(
                query_embeddings=q_embs,
                n_results=top_k,
                ids = list(ok_node_ids),
            )
            # set((i['node_id'], i['insertion_method'], i['doc_id']) for i in node_ref_result['metadatas']) 
            node_docs_per_q = node_results.get("documents") or []
            node_ids_per_q = node_results.get("ids") or []
            node_scores_per_q = node_results.get("distances") or node_results.get("similarities") or []

        # ---- search edges given node and/or edge embedding in a list --------------------------------------------------
        if include_edges:
            edge_ref_result = engine.edge_refs_collection.get(where=cand_where)
            ok_edge_ids = set((i['edge_id']) for i in edge_ref_result['metadatas']) 
            edge_results = engine.edge_collection.query(
                query_embeddings=q_embs,
                n_results=top_k,
                ids = list(ok_edge_ids),
            )
            # edge_results = engine.edge_collection.query(
            #     query_embeddings=q_embs,
            #     n_results=top_k,
            #     where=cand_where,
            # )
            edge_docs_per_q = edge_results.get("documents") or []
            edge_ids_per_q = edge_results.get("ids") or []
            edge_scores_per_q = edge_results.get("distances") or edge_results.get("similarities") or []
        else:
            edge_docs_per_q = edge_ids_per_q = edge_scores_per_q = []
        pairs: List[Tuple[Node, Any]] = []
        # ---- materialize matches per query -------------------------------------------
        for qi, qent in enumerate(queries):
            q_doc = getattr(qent, "doc_id", None)

            # node matches
            docs = node_docs_per_q[qi] if qi < len(node_docs_per_q) else []
            ids_ = node_ids_per_q[qi] if qi < len(node_ids_per_q) else []
            scs = node_scores_per_q[qi] if qi < len(node_scores_per_q) else []
            for mid, mj, score in zip(ids_, docs, scs):
                keep = (score <= max_distance) if score_mode == "distance" else (score >= min_similarity)
                if mid == queries[qi].id:
                    continue
                if not keep:
                    continue
                try:
                    match = Node.model_validate_json(mj)
                except Exception:
                    try:
                        match = Node(**json.loads(mj))
                    except Exception:
                        continue
                # skip reflexive (same id) candidates
                if getattr(match, "id", None) == getattr(qent, "id", None):
                    continue
                if not _pass_doc_rules(q_doc, match.id, "node"):
                    continue
                pairs.append((qent, match))

            # edge matches
            if include_edges and edge_docs_per_q:
                edocs = edge_docs_per_q[qi] if qi < len(edge_docs_per_q) else []
                eids_ = edge_ids_per_q[qi] if qi < len(edge_ids_per_q) else []
                escs = edge_scores_per_q[qi] if qi < len(edge_scores_per_q) else []
                for mid, mj, score in zip(eids_, edocs, escs):
                    keep = (score <= max_distance) if score_mode == "distance" else (score >= min_similarity)
                    if not keep:
                        continue
                    try:
                        match = Edge.model_validate_json(mj)
                    except Exception:
                        try:
                            match = Edge(**json.loads(mj))
                        except Exception:
                            continue
                    if not _pass_doc_rules(q_doc, match.id, "edge"):
                        continue
                    pairs.append((qent, match))

        return pairs

    # ---------------- UNIFIED ENTRY POINT (kept) ----------------
    def propose_any_kind_any_doc(
        self,
        *,
        engine: EngineLike,
        pair_kind: PairKind,
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
        limit_per_bucket: Optional[int] = None,
    ) -> List[Tuple[Any, Any]]:
        engine = self.e or engine

        # 0) doc universe
        if allowed_docs:
            doc_universe = list(dict.fromkeys(allowed_docs))
        else:
            try:
                rows = engine.document_collection.get(include=["metadatas"])
                metas = rows.get("metadatas") or []
                doc_universe = [m["doc_id"] for m in metas if m and m.get("doc_id")]
            except Exception:
                doc_universe = []

        if anchor_doc_id and anchor_doc_id not in doc_universe:
            doc_universe = [anchor_doc_id] + doc_universe

        def _node_ids(doc_id: str) -> List[str]:
            try:
                return engine._nodes_by_doc(doc_id)
            except Exception:
                return []

        def _edge_ids(doc_id: str) -> List[str]:
            try:
                return engine._edge_ids_by_doc(doc_id)
            except Exception:
                return []

        doc2nodes = {d: _node_ids(d) for d in doc_universe}
        doc2edges = {d: _edge_ids(d) for d in doc_universe}

        def _pool_nodes(doc_ids: Iterable[str]) -> List[Node]:
            all_ids: List[str] = []
            for d in doc_ids:
                all_ids.extend(doc2nodes.get(d, []))
            return _load_nodes(engine, all_ids)

        def _pool_edges(doc_ids: Iterable[str]) -> List[Edge]:
            all_ids: List[str] = []
            for d in doc_ids:
                all_ids.extend(doc2edges.get(d, []))
            return _load_edges(engine, all_ids)

        def _passes_doc_rules(id_a: str, kind_a: Literal["node", "edge"], id_b: str, kind_b: Literal["node", "edge"]) -> bool:
            da = _first_doc_id(engine, id_a, kind_a)
            db = _first_doc_id(engine, id_b, kind_b)
            if allowed_docs and (da not in allowed_docs or db not in allowed_docs):
                return False
            if anchor_doc_id and anchor_only and not (da == anchor_doc_id or db == anchor_doc_id):
                return False
            if cross_doc_only and da and db and da == db:
                return False
            return True

        def _cap(xs: List[_Pair]) -> List[_Pair]:
            if limit_per_bucket and limit_per_bucket > 0:
                return xs[:limit_per_bucket]
            return xs

        pairs: List[_Pair] = []

        def _doc_pairs(doc_a: str, doc_b: str) -> List[_Pair]:
            out: List[_Pair] = []
            left_kind, right_kind = pair_kind.split("_")
            lefts: List[Node|Edge] = []
            rights: List[Node|Edge] = []
            for assignee, kind, doc in zip([lefts,rights], [left_kind, right_kind], [doc_a, doc_b]):
                if kind == "node":
                    assignee += _pool_nodes([doc])
                elif kind == "edge":
                    assignee += _pool_edges([doc])
                elif kind == "any":
                    assignee += _pool_nodes([doc]) + _pool_edges([doc])
                else:
                    raise ValueError("kind is not in 'node', 'edge', 'any'")
            # lefts = {ne.id: ne for ne in lefts}
            # rights = {ne.id: ne for ne in rights}
            for i, L in enumerate(lefts):
                for j, R in enumerate(rights):
                    if (doc_a == doc_b and j <= i) or L.id == R.id: # dif doc ids can refer to same nodes, already adjudicated nodes potentially
                        continue
                    # if _passes_doc_rules(L.id, left_kind, R.id, right_kind):
                    out.append(_Pair(L, R))
            
            # if pair_kind == "node_node":
            #     lefts = _pool_nodes([doc_a])
            #     rights = _pool_nodes([doc_b])
            #     for i, L in enumerate(lefts):
            #         for j, R in enumerate(rights):
            #             if doc_a == doc_b and j <= i:
            #                 continue
            #             if _passes_doc_rules(L.id, "node", R.id, "node"):
            #                 out.append(_Pair(L, R))
            # elif pair_kind == "edge_edge":
            #     lefts = _pool_edges([doc_a])
            #     rights = _pool_edges([doc_b])
            #     for i, L in enumerate(lefts):
            #         for j, R in enumerate(rights):
            #             if doc_a == doc_b and j <= i:
            #                 continue
            #             if not _passes_doc_rules(L.id, "edge", R.id, "edge"):
            #                 continue
            #             # let adjudicator decide; we keep both close & far candidates
            #             out.append(_Pair(L, R))
            # else:  # node_edge
            #     lefts = _pool_nodes([doc_a])
            #     rights = _pool_edges([doc_b])
            #     for L in lefts:
            #         for R in rights:
            #             if not _passes_doc_rules(L.id, "node", R.id, "edge"):
            #                 continue
            #             out.append(_Pair(L, R))
            return _cap(out)

        if anchor_doc_id:
            for d in doc_universe:
                if d == anchor_doc_id and not cross_doc_only:
                    pairs.extend(_doc_pairs(anchor_doc_id, anchor_doc_id))
                elif d != anchor_doc_id:
                    pairs.extend(_doc_pairs(anchor_doc_id, d))
                    if not anchor_only and not cross_doc_only:
                        pairs.extend(_doc_pairs(d, d))
        else:
            for i, da in enumerate(doc_universe):
                if not cross_doc_only:
                    pairs.extend(_doc_pairs(da, da))
                for db in doc_universe[i + 1:]:
                    pairs.extend(_doc_pairs(da, db))

        # dedupe symmetric
        seen = set()
        ordered: List[Tuple[Any, Any]] = []
        for p in pairs:
            a = getattr(p.left, "id", None)
            b = getattr(p.right, "id", None)
            if a is None or b is None:
                continue
            key = (("node" if isinstance(p.left, Node) else "edge", a),
                   ("node" if isinstance(p.right, Node) else "edge", b))
            rkey = (key[1], key[0])
            if key in seen or rkey in seen:
                continue
            seen.add(key)
            ordered.append((p.left, p.right))
        return ordered

    # ---------------- BACK-COMPAT WRAPPERS ----------------

    def same_kind_in_doc(self, *, engine: EngineLike, doc_id: str, kind: Literal["node", "edge"] = "node") -> List[Tuple[Any, Any]]:
        pk: PairKind = "node_node" if kind == "node" else "edge_edge"
        return self.propose_any_kind_any_doc(
            engine=engine,
            pair_kind=pk,
            allowed_docs=[doc_id],
            anchor_doc_id=doc_id,
            cross_doc_only=False,
            anchor_only=True,
        )

    def cross_kind_in_doc(self, *, engine: EngineLike, doc_id: str, limit_per_bucket: int = 200) -> List[Tuple[Node, Edge]]:
        out = self.propose_any_kind_any_doc(
            engine=engine,
            pair_kind="node_edge",
            allowed_docs=[doc_id],
            anchor_doc_id=doc_id,
            cross_doc_only=False,
            anchor_only=True,
            limit_per_bucket=limit_per_bucket,
        )
        return [(l, r) for (l, r) in out]

    def for_new_node(
        self,
        engine: EngineLike,
        new_node: Node,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Node, Node]]:
        pairs = self.generate_merge_candidates(
            engine=engine,
            new_node=new_node,
            top_k=top_k,
            score_mode="similarity",
            min_similarity=similarity_threshold,
            include_edges=False,
        )
        return [(q, m) for (q, m) in pairs if isinstance(m, Node)]

class CompositeProposer(MergeCandidateProposer):
    def __init__(self, base: Optional[VectorProposer] = None):
        self.base = base or VectorProposer

    def for_new_node(
        self,
        engine: EngineLike,
        new_node: Node,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Node, Node]]:
        return self.base(engine).for_new_node(engine, new_node, top_k, similarity_threshold)

    def same_kind_in_doc(self, engine: EngineLike, doc_id: str) -> List[Tuple[Any, Any]]:
        return self.base(engine).same_kind_in_doc(engine=engine, doc_id=doc_id, kind="node")

    def cross_kind_in_doc(self, engine: EngineLike, doc_id: str) -> List[Tuple[Any, Any]]:
        return self.base(engine).cross_kind_in_doc(engine=engine, doc_id=doc_id)
