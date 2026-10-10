from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, cast

from kogwistar.id_provider import new_id_str

from ..engine_core.models import (
    AdjudicationTarget,
    AdjudicationVerdict,
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from ..engine_core.utils.refs import select_best_grounding
from .types import EngineLike, MergePolicy


def _backend_object(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _required_id(value: str | None) -> str:
    if value is None:
        raise ValueError("merge target is missing an entity id")
    return value


def adjundication_span(
    verdict: AdjudicationVerdict,
    left: Node,
    right: Node,
    adjundication_method: str,
) -> Span:
    if not verdict.same_entity:
        raise ValueError("only created when the result is positive verdict")
    self_span = Span(
        collection_page_url=f"adjudication/{left.safe_get_id()}-{right.safe_get_id()}",
        document_page_url=f"adjudication/{left.safe_get_id()}-{right.safe_get_id()}",
        doc_id=f"adjudication:{left.safe_get_id()}-{right.safe_get_id()}-{verdict.canonical_entity_id}",
        insertion_method="adjudication run",
        page_number=1,
        start_char=0,
        end_char=len(verdict.reason),
        excerpt=verdict.reason,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system",
            is_verified=True,
            score=verdict.confidence,
            notes=adjundication_method,
        ),
    )
    return self_span


class PreferExistingCanonical(MergePolicy):
    def __init__(self, engine: EngineLike) -> None:
        self.e: EngineLike = engine

    def commit_merge(
        self,
        left: Node,
        right: Node,
        verdict: AdjudicationVerdict,
        method: str = "unspecified",
    ) -> str:
        """
        Apply a positive adjudication by assigning/propagating a canonical_entity_id
        and recording a `same_as` edge with provenance. Persists changes to storage.
        Returns the canonical id used.
        """
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")

        canonical_id = verdict.canonical_entity_id or (
            left.canonical_entity_id or right.canonical_entity_id
        )
        if not canonical_id:
            canonical_id = str(new_id_str())

        left.canonical_entity_id = canonical_id
        right.canonical_entity_id = canonical_id

        def _persist_node(n: Node) -> None:
            prior = _backend_object(
                self.e.backend.node_get(ids=[_required_id(n.id)], include=["metadatas"])
            )
            doc_id = None
            if prior.get("metadatas") and prior["metadatas"][0]:
                doc_id = prior["metadatas"][0].get("doc_id")
            doc, meta = self.e.write.node_doc_and_meta(n)
            emb = self.e.embed.iterative_defensive_emb(doc)
            self.e.backend.node_update(
                ids=[n.id],
                embeddings=[emb],
                documents=[doc],
                metadatas=[meta],
            )
            self.e.write.index_node_docs(n)
            self.e.write.index_node_refs(n)
            n.doc_id = doc_id

        _persist_node(left)
        _persist_node(right)

        s_nodes, s_edges, t_nodes, t_edges = self.e.adjudicate.split_endpoints(
            [_required_id(left.id)], [_required_id(right.id)]
        )
        same_as = Edge(
            id=str(new_id_str()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "Adjudicated same entity",
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            metadata={},
            relation="same_as",
            source_ids=s_nodes,
            target_ids=t_nodes,
            properties={"confidence": verdict.confidence},
            mentions=[
                Grounding(spans=[adjundication_span(verdict, left, right, method)])
            ],
            doc_id="__adjudication__",
            source_edge_ids=s_edges,
            target_edge_ids=t_edges,
        )
        self.e.write.add_edge(same_as)

        ep_ids, ep_docs, ep_metas = [], [], []
        for role, node_ids in (
            ("src", same_as.source_ids or []),
            ("tgt", same_as.target_ids or []),
        ):
            for nid in node_ids:
                ep_id = f"{same_as.id}::{role}::{nid}"
                n_meta = _backend_object(
                    self.e.backend.node_get(ids=[nid], include=["metadatas"])
                )
                per_doc = None
                if n_meta.get("metadatas") and n_meta["metadatas"][0]:
                    per_doc = n_meta["metadatas"][0].get("doc_id")
                row = self.e.write.strip_none(
                    {
                        "id": ep_id,
                        "edge_id": same_as.id,
                        "node_id": nid,
                        "role": role,
                        "relation": same_as.relation,
                        "doc_id": per_doc,
                    }
                )
                ep_ids.append(ep_id)
                ep_docs.append(json.dumps(row))
                ep_metas.append(row)

        if ep_ids:
            self.e.backend.edge_endpoints_add(
                ids=ep_ids,
                documents=ep_docs,
                metadatas=ep_metas,
            )
        return canonical_id

    def commit_any_kind(
        self,
        node_or_edge_l: AdjudicationTarget,
        node_or_edge_r: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str:
        if not verdict.same_entity:
            raise ValueError("Verdict not positive")

        relation_name = (
            "reifies"
            if self.e.cross_kind_strategy == "reifies"
            else "equivalent_node_edge"
        )

        left_target = self.e.adjudicate.fetch_target(node_or_edge_l)
        right_target = self.e.adjudicate.fetch_target(node_or_edge_r)

        left_ref = select_best_grounding(left_target)
        right_ref = select_best_grounding(right_target)
        link = Edge(
            id=str(uuid.uuid4()),
            label=relation_name,
            type="relationship",
            summary=verdict.reason,
            domain_id=None,
            canonical_entity_id=None,
            relation=relation_name,
            source_ids=[_required_id(left_target.id)] if node_or_edge_l.kind == "node" else [],
            target_ids=[_required_id(right_target.id)] if node_or_edge_r.kind == "node" else [],
            source_edge_ids=[_required_id(left_target.id)] if node_or_edge_l.kind == "edge" else [],
            target_edge_ids=[_required_id(right_target.id)] if node_or_edge_r.kind == "edge" else [],
            properties={"confidence": verdict.confidence},
            mentions=[left_ref, right_ref],
            doc_id="__adjudication__",
            embedding=None,
            metadata={},
        )
        assert (
            bool(link.source_ids)
            + bool(link.target_ids)
            + bool(link.source_edge_ids)
            + bool(link.target_edge_ids)
            == 2
        )
        self.e.write.add_edge(link, doc_id=link.doc_id)
        return _required_id(link.id)

    def commit_merge_target(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str:
        """Generalized merge: node-node or edge-edge. Returns canonical id."""
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")
        if left.kind != right.kind:
            raise ValueError("Cannot merge cross-kind (node vs edge).")

        canonical_id = verdict.canonical_entity_id
        if not canonical_id:
            left_target = self.e.adjudicate.fetch_target(left)
            right_target = self.e.adjudicate.fetch_target(right)
            canonical_id = (
                getattr(left_target, "canonical_entity_id", None)
                or getattr(right_target, "canonical_entity_id", None)
                or str(uuid.uuid4())
            )

        if left.kind == "node":
            left_node = cast(Node, self.e.adjudicate.fetch_target(left))
            right_node = cast(Node, self.e.adjudicate.fetch_target(right))
            left_node.canonical_entity_id = right_node.canonical_entity_id = canonical_id
            self.e.backend.node_update(
                ids=[_required_id(left_node.id)],
                documents=[left_node.model_dump_json()],
                metadatas=[
                    self.e.write.strip_none(
                        {
                            "doc_id": getattr(left_node, "doc_id", None),
                            "label": left_node.label,
                            "type": left_node.type,
                            "summary": left_node.summary,
                            "domain_id": left_node.domain_id,
                            "canonical_entity_id": left_node.canonical_entity_id,
                            "properties": self.e.write.json_or_none(left_node.properties),
                            "references": self.e.write.json_or_none(
                                [ref.model_dump() for ref in (left_node.mentions or [])]
                            ),
                        }
                    )
                ],
            )
            self.e.write.index_node_docs(left_node)
            self.e.backend.node_update(
                ids=[_required_id(right_node.id)],
                documents=[right_node.model_dump_json()],
                metadatas=[
                    self.e.write.strip_none(
                        {
                            "doc_id": getattr(right_node, "doc_id", None),
                            "label": right_node.label,
                            "type": right_node.type,
                            "summary": right_node.summary,
                            "domain_id": right_node.domain_id,
                            "canonical_entity_id": right_node.canonical_entity_id,
                            "properties": self.e.write.json_or_none(right_node.properties),
                            "references": self.e.write.json_or_none(
                                [ref.model_dump() for ref in (right_node.mentions or [])]
                            ),
                        }
                    )
                ],
            )
            self.e.write.index_node_docs(right_node)
            same_as = Edge(
                id=str(uuid.uuid4()),
                label="same_as",
                type="relationship",
                summary=verdict.reason or "merge",
                domain_id=None,
                canonical_entity_id=None,
                relation="same_as",
                source_ids=[_required_id(left_node.id)],
                target_ids=[_required_id(right_node.id)],
                source_edge_ids=[],
                target_edge_ids=[],
                properties={"confidence": verdict.confidence},
                mentions=[select_best_grounding(left_node), select_best_grounding(right_node)],
                doc_id="__adjudication__",
                embedding=None,
                metadata={},
            )
            self.e.write.add_edge(same_as, doc_id=same_as.doc_id)
            return _required_id(canonical_id)

        le = cast(Edge, self.e.adjudicate.fetch_target(left))
        re = cast(Edge, self.e.adjudicate.fetch_target(right))
        le.canonical_entity_id = re.canonical_entity_id = canonical_id
        self.e.backend.edge_update(
            ids=[le.id],
            documents=[le.model_dump_json()],
            metadatas=[
                self.e.write.strip_none(
                    {
                        "doc_id": getattr(le, "doc_id", None),
                        "relation": le.relation,
                        "source_ids": self.e.write.json_or_none(le.source_ids),
                        "target_ids": self.e.write.json_or_none(le.target_ids),
                        "type": le.type,
                        "summary": le.summary,
                        "domain_id": le.domain_id,
                        "canonical_entity_id": le.canonical_entity_id,
                        "properties": self.e.write.json_or_none(le.properties),
                        "references": self.e.write.json_or_none(
                            [ref.model_dump() for ref in (le.mentions or [])]
                        ),
                    }
                )
            ],
        )
        self.e.backend.edge_update(
            ids=[re.id],
            documents=[re.model_dump_json()],
            metadatas=[
                self.e.write.strip_none(
                    {
                        "doc_id": getattr(re, "doc_id", None),
                        "relation": re.relation,
                        "source_ids": self.e.write.json_or_none(re.source_ids),
                        "target_ids": self.e.write.json_or_none(re.target_ids),
                        "type": re.type,
                        "summary": re.summary,
                        "domain_id": re.domain_id,
                        "canonical_entity_id": re.canonical_entity_id,
                        "properties": self.e.write.json_or_none(re.properties),
                        "references": self.e.write.json_or_none(
                            [ref.model_dump() for ref in (re.mentions or [])]
                        ),
                    }
                )
            ],
        )
        self.e.write.index_edge_refs(le)
        self.e.write.index_edge_refs(re)
        same_as_meta = Edge(
            id=str(uuid.uuid4()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "merge",
            domain_id=None,
            canonical_entity_id=None,
            relation="same_as",
            source_ids=[],
            target_ids=[],
            source_edge_ids=[_required_id(le.id)],
            target_edge_ids=[_required_id(re.id)],
            properties={"confidence": verdict.confidence},
            mentions=[select_best_grounding(le), select_best_grounding(re)],
            doc_id="__adjudication__",
            embedding=None,
            metadata={},
        )
        self.e.write.add_edge(same_as_meta, doc_id=same_as_meta.doc_id)
        self.e.write.index_edge_refs(same_as_meta)
        return canonical_id

    def merge(
        self,
        left: Node | Edge | AdjudicationTarget,
        right: Node | Edge | AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str:
        if isinstance(left, AdjudicationTarget) and isinstance(
            right, AdjudicationTarget
        ):
            return self.e.commit_merge_target(left, right, verdict)
        if isinstance(left, Node) and isinstance(right, Node):
            return self.e.commit_merge(left, right, verdict, "merge_policy")
        if isinstance(left, Edge) and isinstance(right, Edge):
            return self.e.commit_merge_target(
                self.e.adjudicate.target_from_edge(left),
                self.e.adjudicate.target_from_edge(right),
                verdict,
            )
        raise ValueError("Cannot merge mixed target representations")
