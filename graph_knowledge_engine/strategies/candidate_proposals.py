# graph_knowledge_engine/strategies/candidate_proposals.py
from __future__ import annotations
from typing import List, Tuple, Any, Optional, Literal
import json
from ..strategies import EngineLike, Proposer

class DefaultProposer(Proposer):
    """
    A pragmatic proposer that works without embeddings:
    - Nodes: bucket by (type, normalized label) within doc scope (or globally),
      emit all pairwise combos from buckets with size >=2.
    - Edges: bucket by (relation, normalized signature) where signature =
      sorted(source_ids)+sorted(target_ids), emit duplicates.
    - Cross-kind: pair reified nodes (with 'signature_text' in properties)
      with edges whose 'signature_text' matches.
    """
    def __init__(self, engine: EngineLike):
        self.e = engine

    # -------------- helpers --------------
    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    # -------------- Proposer API --------------
    def generate_merge_candidates(
        self,
        *,
        kind: Literal["node", "edge"],
        scope_doc_id: Optional[str] = None,
        top_k: int = 50,
        similarity_threshold: float = 0.85,  # kept for signature parity; not used in this heuristic impl
    ) -> List[Tuple[Any, Any]]:
        if kind == "node":
            return self._node_candidates(scope_doc_id, top_k)
        else:
            return self._edge_candidates(scope_doc_id, top_k)

    def generate_cross_kind_candidates(
        self,
        *,
        scope_doc_id: Optional[str] = None,
        limit_per_bucket: int = 50,
    ) -> List[Tuple[Any, Any]]:
        # Pull nodes (look for 'signature_text') and edges;
        # pair when signature_text matches.
        from ..models import Node, Edge

        # Load nodes
        if scope_doc_id:
            ids = self.e.node_ids_by_doc(scope_doc_id)
            got = self.e.node_collection.get(ids=ids, include=["documents"])
        else:
            got = self.e.node_collection.get(include=["ids", "documents"])

        nodes: List[Node] = [Node.model_validate_json(d) for d in (got.get("documents") or [])]

        # Load edges
        if scope_doc_id:
            eids = self.e.edge_ids_by_doc(scope_doc_id)
            egot = self.e.edge_collection.get(ids=eids, include=["documents"])
        else:
            egot = self.e.edge_collection.get(include=["ids", "documents"])
        edges: List[Edge] = [Edge.model_validate_json(d) for d in (egot.get("documents") or [])]

        n_by_sig = {}
        for n in nodes:
            sig = None
            if n.properties and "signature_text" in n.properties:
                sig = str(n.properties["signature_text"])
            if not sig:
                continue
            n_by_sig.setdefault(self._norm(sig), []).append(n)

        pairs: List[Tuple[Any, Any]] = []
        for e in edges:
            ps = e.properties or {}
            sig = self._norm(str(ps.get("signature_text", "")))
            if not sig or sig not in n_by_sig:
                continue
            for n in n_by_sig[sig][:limit_per_bucket]:
                pairs.append((n, e))
        return pairs

    # -------------- internal --------------
    def _node_candidates(self, scope_doc_id: Optional[str], top_k: int) -> List[Tuple[Any, Any]]:
        from ..models import Node
        if scope_doc_id:
            ids = self.e.node_ids_by_doc(scope_doc_id)
            got = self.e.node_collection.get(ids=ids, include=["documents"])
        else:
            got = self.e.node_collection.get(include=["ids", "documents"])

        buckets = {}
        out: List[Tuple[Any, Any]] = []
        for d in (got.get("documents") or []):
            n = Node.model_validate_json(d)
            key = (n.type, self._norm(n.label))
            buckets.setdefault(key, []).append(n)

        for b in buckets.values():
            if len(b) < 2:
                continue
            # all combs
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    out.append((b[i], b[j]))
                    if len(out) >= top_k:
                        return out
        return out

    def _edge_candidates(self, scope_doc_id: Optional[str], top_k: int) -> List[Tuple[Any, Any]]:
        from ..models import Edge
        if scope_doc_id:
            eids = self.e.edge_ids_by_doc(scope_doc_id)
            got = self.e.edge_collection.get(ids=eids, include=["documents"])
        else:
            got = self.e.edge_collection.get(include=["ids", "documents"])

        def sig(e: Edge) -> tuple:
            s = tuple(sorted(e.source_ids or [])) + tuple(sorted(getattr(e, "source_edge_ids", []) or []))
            t = tuple(sorted(e.target_ids or [])) + tuple(sorted(getattr(e, "target_edge_ids", []) or []))
            st = e.properties.get("signature_text") if (e.properties) else None
            return (e.relation, s, t, self._norm(str(st)) if st else "")

        edges: List[Edge] = [Edge.model_validate_json(d) for d in (got.get("documents") or [])]
        buckets = {}
        out: List[Tuple[Any, Any]] = []
        for e in edges:
            buckets.setdefault(sig(e), []).append(e)

        for b in buckets.values():
            if len(b) < 2:
                continue
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    out.append((b[i], b[j]))
                    if len(out) >= top_k:
                        return out
        return out