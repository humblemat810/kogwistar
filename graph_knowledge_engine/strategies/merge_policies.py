# strategies/merge_policies.py
from __future__ import annotations
from ..models import AdjudicationVerdict, AdjudicationTarget, Edge, Node, ReferenceSession, MentionVerification
from .types import EngineLike, MergePolicy
import uuid
import json
class PreferExistingCanonical(MergePolicy):
    def __init__(self, engine: EngineLike):
        self.e : EngineLike = engine
    def commit_merge(self, left: Node, right: Node, verdict: AdjudicationVerdict) -> str:
        """
        Apply a positive adjudication by assigning/propagating a canonical_entity_id
        and recording a `same_as` edge with provenance. Persists changes to Chroma.
        Returns the canonical id used.
        """
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")

        canonical_id = verdict.canonical_entity_id or (left.canonical_entity_id or right.canonical_entity_id)
        if not canonical_id:
            canonical_id = str(uuid.uuid1())

        # 1) Update in-memory nodes
        left.canonical_entity_id = canonical_id
        right.canonical_entity_id = canonical_id

        # 2) Persist node updates to Chroma (documents + metadatas)
        def _persist_node(n: Node):
            # Try to retain prior metadata (esp. doc_id)
            prior = self.e.node_collection.get(ids=[n.id], include=["metadatas"])
            doc_id = None
            if prior.get("metadatas") and prior["metadatas"][0]:
                doc_id = prior["metadatas"][0].get("doc_id")
            # Update document JSON
            self.e.node_collection.update(
                ids=[n.id],
                documents=[n.model_dump_json()],
                metadatas=[self.e._strip_none({
                    "doc_id": doc_id,
                    "label": n.label,
                    "type": n.type,
                    "summary": n.summary,
                    "domain_id": n.domain_id,
                    "canonical_entity_id": n.canonical_entity_id,
                    "properties": self.e._json_or_none(n.properties),
                    "references": self.e._json_or_none([ref.model_dump() for ref in (n.references or [])]),
                })],
            )
            self.e._index_node_docs(n)
            self.e._index_node_refs(n)
            # also mirror back onto the object for future calls
            n.doc_id = doc_id

        _persist_node(left)
        _persist_node(right)

        # 3) Build edge references from each side (pick a “best” mention per node)
        def _best_ref(n: Node) -> ReferenceSession:
            if n.references:
                refs = sorted(n.references, key=lambda r: (getattr(r, "start_page", 10**9), getattr(r, "start_char", 10**9)))
                ref = refs[0].model_copy(deep=True)
                if ref.verification is None:
                    ref.verification = MentionVerification(method="heuristic", is_verified=True, score=0.5, notes="adjudication evidence")
                else:
                    ref.verification.notes = (ref.verification.notes or "") + " | adjudication evidence"
                return ref
            # Fallback if ever empty (shouldn’t with your schema)
            did = getattr(n, "doc_id", None) or "unknown"
            return self.e._default_ref(did, snippet=n.summary if hasattr(n, "summary") else None)

        left_ref = _best_ref(left)
        right_ref = _best_ref(right)
        s_nodes, s_edges, t_nodes, t_edges = self.e._split_endpoints([left.id], [right.id])

        same_as = Edge(
            id=str(uuid.uuid1()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "Adjudicated same entity",
            domain_id=None,
            relation="same_as",
            source_ids=s_nodes,
            target_ids=t_nodes,
            properties={"confidence": verdict.confidence},
            references=[left_ref, right_ref],
            doc_id="__adjudication__",   # neutral; endpoints will carry per-node doc_id
            source_edge_ids=s_edges,
            target_edge_ids=t_edges,
        )
        self.e.add_edge(same_as)
        # 4) Persist the same_as edge and per-endpoint rows
        #    Main edge row
        # self.e.edge_collection.add(
        #     ids=[same_as.id],
        #     documents=[same_as.model_dump_json()],
        #     metadatas=[self.e._strip_none({
        #         "doc_id": getattr(same_as, "doc_id", None),
        #         "relation": same_as.relation,
        #         "source_ids": self.e._json_or_none(same_as.source_ids),
        #         "target_ids": self.e._json_or_none(same_as.target_ids),
        #         "type": same_as.type,
        #         "summary": same_as.summary,
        #         "domain_id": same_as.domain_id,
        #         "canonical_entity_id": same_as.canonical_entity_id,
        #         "properties": self.e._json_or_none(same_as.properties),
        #         "references": self.e._json_or_none([ref.model_dump() for ref in (same_as.references or [])]),
        #     })],
        # )

        #    Endpoint fanout with per-endpoint doc_id
        ep_ids, ep_docs, ep_metas = [], [], []
        for role, node_ids in (("src", same_as.source_ids or []), ("tgt", same_as.target_ids or [])):
            for nid in node_ids:
                ep_id = f"{same_as.id}::{role}::{nid}"
                n_meta = self.e.node_collection.get(ids=[nid], include=["metadatas"])
                per_doc = None
                if n_meta.get("metadatas") and n_meta["metadatas"][0]:
                    per_doc = n_meta["metadatas"][0].get("doc_id")
                m = self.e._strip_none({
                    "id": ep_id,
                    "edge_id": same_as.id,
                    "node_id": nid,
                    "role": role,
                    "relation": same_as.relation,
                    "doc_id": per_doc,
                })
                ep_ids.append(ep_id)
                ep_docs.append(json.dumps(m))
                ep_metas.append(m)

        if ep_ids:
            self.e.edge_endpoints_collection.add(ids=ep_ids, documents=ep_docs, metadatas=ep_metas)
        return canonical_id
    def commit_any_kind(self, node_or_edge_l: AdjudicationTarget, node_or_edge_r: AdjudicationTarget,
                        verdict: AdjudicationVerdict) -> str:
        if not verdict.same_entity:
            raise ValueError("Verdict not positive")

        # (Optionally) if user explicitly requested 'equivalent', you could propagate a canonical,
        # but usually we DO NOT mix canonical namespaces. So we default to a linking meta-edge.

        relation_name = "reifies" if self.e.cross_kind_strategy == "reifies" else "equivalent_node_edge"

        l = self.e._fetch_target(node_or_edge_l)   # Node
        r = self.e._fetch_target(node_or_edge_r)   # Edge

        # evidence: copy best ref from both sides
        left_ref = self.e._best_ref(l)
        right_ref = self.e._best_ref(r) if r.references else left_ref

        link = Edge(
            id=str(uuid.uuid4()),
            label=relation_name,
            type="relationship",
            summary=verdict.reason,
            relation=relation_name,
            source_ids=[l.id], target_ids=[],
            source_edge_ids=[], target_edge_ids=[r.id],   # <-- node → (meta)edge
            properties={"confidence": verdict.confidence},
            references=[left_ref, right_ref],
            doc_id="__adjudication__",
        )
        self.e.add_edge(link, doc_id=link.doc_id)
        return link.id
    def commit_merge_target(self, left: AdjudicationTarget, right: AdjudicationTarget, verdict: AdjudicationVerdict) -> str:
        """Generalized merge: node↔node or edge↔edge. Returns canonical id."""
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")
        if left.kind != right.kind:
            raise ValueError("Cannot merge cross-kind (node vs edge).")

        # Decide canonical id
        canonical_id = verdict.canonical_entity_id
        if not canonical_id:
            # prefer any existing canonical; else new
            l = self.e._fetch_target(left)
            r = self.e._fetch_target(right)
            canonical_id = getattr(l, "canonical_entity_id", None) or getattr(r, "canonical_entity_id", None) or str(uuid.uuid4())

        if left.kind == "node":
            # --- Node merge: as you already do ---
            l:Node = self.e._fetch_target(left)
            r:Node = self.e._fetch_target(right)
            l.canonical_entity_id = r.canonical_entity_id = canonical_id
            # persist
            self.e.node_collection.update(
                ids=[l.id],
                documents=[l.model_dump_json()],
                metadatas=[self.e._strip_none({
                    "doc_id": getattr(l, "doc_id", None),
                    "label": l.label, "type": l.type, "summary": l.summary,
                    "domain_id": l.domain_id, "canonical_entity_id": l.canonical_entity_id,
                    "properties": self.e._json_or_none(l.properties),
                    "references": self.e._json_or_none([ref.model_dump() for ref in (l.references or [])]),
                })],
            )
            self.e._index_node_docs(l)
            self.e.node_collection.update(
                ids=[r.id],
                documents=[r.model_dump_json()],
                metadatas=[self.e._strip_none({
                    "doc_id": getattr(r, "doc_id", None),
                    "label": r.label, "type": r.type, "summary": r.summary,
                    "domain_id": r.domain_id, "canonical_entity_id": r.canonical_entity_id,
                    "properties": self.e._json_or_none(r.properties),
                    "references": self.e._json_or_none([ref.model_dump() for ref in (r.references or [])]),
                })],
            )
            self.e._index_node_docs(r)
            # record same_as (node↔node)
            left_ref = self.e._best_ref(l)
            right_ref = self.e._best_ref(r)
            same_as = Edge(
                id=str(uuid.uuid4()),
                label="same_as", type="relationship", summary=verdict.reason or "merge",
                relation="same_as",
                source_ids=[l.id], target_ids=[r.id],
                source_edge_ids=[], target_edge_ids=[],
                properties={"confidence": verdict.confidence},
                references=[left_ref, right_ref],
                doc_id="__adjudication__",
            )
            self.e.add_edge(same_as, doc_id=same_as.doc_id)
            return canonical_id

        # --- Edge merge: mirror the same pattern, but meta-edge same_as(edge, edge) ---
        le: Edge = self.e._fetch_target(left)
        re: Edge = self.e._fetch_target(right)
        le.canonical_entity_id = re.canonical_entity_id = canonical_id
        # persist edge updates
        self.e.edge_collection.update(
            ids=[le.id],
            documents=[le.model_dump_json()],
            metadatas=[self.e._strip_none({
                "doc_id": getattr(le, "doc_id", None),
                "relation": le.relation,
                "source_ids": self.e._json_or_none(le.source_ids),
                "target_ids": self.e._json_or_none(le.target_ids),
                "type": le.type, "summary": le.summary,
                "domain_id": le.domain_id, "canonical_entity_id": le.canonical_entity_id,
                "properties": self.e._json_or_none(le.properties),
                "references": self.e._json_or_none([ref.model_dump() for ref in (le.references or [])]),
            })],
        )
        self.e.edge_collection.update(
            ids=[re.id],
            documents=[re.model_dump_json()],
            metadatas=[self.e._strip_none({
                "doc_id": getattr(re, "doc_id", None),
                "relation": re.relation,
                "source_ids": self.e._json_or_none(re.source_ids),
                "target_ids": self.e._json_or_none(re.target_ids),
                "type": re.type, "summary": re.summary,
                "domain_id": re.domain_id, "canonical_entity_id": re.canonical_entity_id,
                "properties": self.e._json_or_none(re.properties),
                "references": self.e._json_or_none([ref.model_dump() for ref in (re.references or [])]),
            })],
        )
        self.e._index_edge_refs(le)
        self.e._index_edge_refs(re)
        # meta same_as: edge↔edge (use edge-endpoint lists)
        same_as_meta = Edge(
            id=str(uuid.uuid4()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "merge",
            relation="same_as",
            source_ids=[], target_ids=[],
            source_edge_ids=[le.id], target_edge_ids=[re.id],
            properties={"confidence": verdict.confidence},
            references=[],  # you can copy best refs from both edges if you desire
            doc_id="__adjudication__",
        )
        self.e.add_edge(same_as_meta, doc_id=same_as_meta.doc_id)
        self.e._index_edge_refs(same_as_meta)
        return canonical_id
    def merge(self, left, right, verdict: AdjudicationVerdict) -> str:
        # node↔node, edge↔edge, cross-kind is handled inside engine methods you already wrote
        if hasattr(left, "kind"):   # AdjudicationTarget path
            if left.kind == right.kind:
                return self.e.commit_merge_target(left, right, verdict)
            # cross-kind: allow link
            return self.e.commit_merge_target(left, right, verdict)
        # Back-compat: raw Node/Edge
        if left.__class__.__name__ == right.__class__.__name__:
            return self.e.commit_merge(left, right, verdict)
        return self.e.commit_merge_target(self.e._target_from_node(left) if left.__class__.__name__=="Node" else self.e._target_from_edge(left),
                                      self.e._target_from_node(right) if right.__class__.__name__=="Node" else self.e._target_from_edge(right),
                                      verdict)