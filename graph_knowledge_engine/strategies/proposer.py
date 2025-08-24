# proposer.py
from __future__ import annotations

from typing import List, Tuple, Optional, Iterable, Literal, Any, Sequence, Union, Dict,
from dataclasses import dataclass
import json

from ..models import Node, Edge
from .types import MergeCandidateProposer, EngineLike  # your existing protocol types


PairKind = Literal["node_node", "edge_edge", "node_edge"]


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
            # some deployments denormalize as JSON string list
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


def _coerce_query_nodes(engine: EngineLike, spec: Union[Node, str, Sequence[Union[Node, str]]]) -> List[Node]:
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
        got = engine.node_collection.get(ids=need_fetch_ids, include=["documents"])
        for dj in (got.get("documents") or []):
            try:
                nodes.append(Node.model_validate_json(dj))
            except Exception:
                pass

    # keep only those that actually have embeddings
    return [n for n in nodes if getattr(n, "embedding", None)]

def _as_set(xs: Optional[Iterable[str]]) -> Set[str]:
    return set(xs or [])
class VectorProposer(MergeCandidateProposer):
    """
    Minimal refactor + batch vector search.

    - NEW: propose_any_kind_any_doc(...) remains the single brute-force/heuristic generator.
    - generate_merge_candidates(...) now supports new_node as Node|id|List[Node|id] and
      performs *batch* Chroma queries against both node & edge collections, then applies
      document scoping filters.
    """
    def __init__(self, engine: EngineLike):
        self.e = engine
        self.limit_per_bucket = 100

    # ---------- ORIGINAL surface kept, but now batch-aware ----------
    def generate_merge_candidates(
        self,
        engine: EngineLike,
        new_node: Union[Node, str, Sequence[Union[Node, str]]],
        top_k: int = 10,
        *,
        # Post-filter knobs (doc scoping):
        allowed_docs: Optional[List[str]] = None,
        anchor_doc_id: Optional[str] = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
        # Vector score thresholding:
        score_mode: Literal["distance", "similarity"] = "distance",
        max_distance: float = 0.25,    # used if score_mode="distance"
        min_similarity: float = 0.85,  # used if score_mode="similarity"
        include_edges: bool = True,    # search matches in edge_collection too
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Node, Any]]:
        """
        Batch vector search for one or many *new* nodes against existing nodes (and edges if enabled).

        Returns list of (query_node, matched_entity) where matched_entity ∈ {Node, Edge}.
        """
        extra = None
        if allowed_docs:
            # If your node metadata uses 'doc_id' for single-source nodes, this works directly.
            # If you denormalized 'doc_ids' (JSON), keep your existing approach or post-filter pairs.
            extra = {"doc_id": {"$in": allowed_docs}}
        cand_where = _and_where(where, extra)
        
        engine = self.e or engine
        queries: List[Node] = _coerce_query_nodes(engine, new_node)
        if not queries:
            return []

        # Run a single batched Chroma query per collection
        q_embs = [q.embedding for q in queries]

        # --- helper to filter by doc rules
        def _pass_doc_rules(lhs_doc: Optional[str], rhs_id: str, rhs_kind: Literal["node", "edge"]) -> bool:
            rhs_doc = _first_doc_id(engine, rhs_id, rhs_kind)
            # allowed set
            if allowed_docs and (rhs_doc not in allowed_docs):
                return False
            # cross-doc constraint
            if cross_doc_only and lhs_doc and rhs_doc and lhs_doc == rhs_doc:
                return False
            # anchor constraint
            if anchor_doc_id and anchor_only and not ((lhs_doc == anchor_doc_id) or (rhs_doc == anchor_doc_id)):
                return False
            return True

        pairs: List[Tuple[Node, Any]] = []

        # -------- query nodes collection
        node_results = engine.node_collection.query(query_embeddings=q_embs, n_results=top_k)
        node_docs_per_q = node_results.get("documents") or []
        node_ids_per_q = node_results.get("ids") or []
        node_scores_per_q = node_results.get("distances") or node_results.get("similarities") or []

        # -------- (optional) query edges collection
        if include_edges:
            edge_results = engine.edge_collection.query(query_embeddings=q_embs, n_results=top_k)
            edge_docs_per_q = edge_results.get("documents") or []
            edge_ids_per_q = edge_results.get("ids") or []
            edge_scores_per_q = edge_results.get("distances") or edge_results.get("similarities") or []
        else:
            edge_docs_per_q = edge_ids_per_q = edge_scores_per_q = []

        # Materialize matches
        for qi, qnode in enumerate(queries):
            q_doc = getattr(qnode, "doc_id", None)

            # nodes
            docs = node_docs_per_q[qi] if qi < len(node_docs_per_q) else []
            ids_ = node_ids_per_q[qi] if qi < len(node_ids_per_q) else []
            scs = node_scores_per_q[qi] if qi < len(node_scores_per_q) else []
            for mid, mj, score in zip(ids_, docs, scs):
                # threshold
                keep = (score <= max_distance) if score_mode == "distance" else (score >= min_similarity)
                if not keep:
                    continue
                try:
                    match = Node.model_validate_json(mj)
                except Exception:
                    try:
                        match = Node(**json.loads(mj))
                    except Exception:
                        continue
                if match.id == qnode.id:
                    continue
                if not _pass_doc_rules(q_doc, match.id, "node"):
                    continue
                pairs.append((qnode, match))

            # edges
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
                    pairs.append((qnode, match))

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
        """
        pair_kind:
          - "node_node": generate Node↔Node pairs
          - "edge_edge": generate Edge↔Edge pairs
          - "node_edge": generate Node↔Edge pairs
        Doc-scoping rules as described in the docstring of generate_merge_candidates.
        """
        engine = self.e or engine

        # --- 0) Doc universe
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

        # --- 1) ids by doc
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

        # --- 2) pools
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

        # --- rules
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
            if pair_kind == "node_node":
                lefts = _pool_nodes([doc_a])
                rights = _pool_nodes([doc_b])
                for i, L in enumerate(lefts):
                    for j, R in enumerate(rights):
                        if doc_a == doc_b and j <= i:
                            continue
                        if _passes_doc_rules(L.id, "node", R.id, "node"):
                            out.append(_Pair(L, R))
            elif pair_kind == "edge_edge":
                lefts = _pool_edges([doc_a])
                rights = _pool_edges([doc_b])
                for i, L in enumerate(lefts):
                    for j, R in enumerate(rights):
                        if doc_a == doc_b and j <= i:
                            continue
                        if not _passes_doc_rules(L.id, "edge", R.id, "edge"):
                            continue
                        if (_edge_sig(L)[0] == _edge_sig(R)[0]) or (_edge_sig(L) == _edge_sig(R)):
                            out.append(_Pair(L, R))
                        else:
                            out.append(_Pair(L, R))
            else:  # node_edge
                lefts = _pool_nodes([doc_a])
                rights = _pool_edges([doc_b])
                for L in lefts:
                    lk = _norm(L.label) or _norm(L.summary)
                    for R in rights:
                        if not _passes_doc_rules(L.id, "node", R.id, "edge"):
                            continue
                        ok = False
                        if lk:
                            in_sig = _norm(getattr(R, "relation", None))
                            ok = (lk in in_sig) or (in_sig and in_sig in lk)
                        out.append(_Pair(L, R) if ok else _Pair(L, R))
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
                for db in doc_universe[i + 1 :]:
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

    # ---------------- BACK-COMPAT THIN WRAPPERS ----------------

    def same_kind_in_doc(self, *, engine: EngineLike, doc_id: str, kind: Literal["node", "edge"]) -> List[Tuple[Any, Any]]:
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
        return [(l, r) for (l, r) in out]  # cast: Node,Edge

    # Legacy single-query helper (kept name; forwards to batch path)
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
            include_edges=False,  # legacy kept: node↔node only
        )
        # Filter to node↔node for strict back-compat
        return [(q, m) for (q, m) in pairs if isinstance(m, Node)]


class CompositeProposer(MergeCandidateProposer):
    """Tiny orchestrator that forwards to VectorProposer; public surface unchanged."""
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
