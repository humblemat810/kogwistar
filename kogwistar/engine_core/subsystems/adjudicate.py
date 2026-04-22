from __future__ import annotations

from typing import Any

from ..async_compat import run_awaitable_blocking
from ..models import AdjudicationTarget, Edge, Node
from .base import NamespaceProxy
from ...typing_interfaces import AdjudicateLike


class AdjudicateSubsystem(NamespaceProxy, AdjudicateLike):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def target_from_node(self, n: Node) -> AdjudicationTarget:
        return AdjudicationTarget(
            kind="node",
            id=n.id,
            label=n.label,
            type=n.type,
            summary=n.summary,
            domain_id=n.domain_id,
            canonical_entity_id=n.canonical_entity_id,
            properties=n.properties,
        )

    def target_from_edge(self, e: Edge) -> AdjudicationTarget:
        return AdjudicationTarget(
            kind="edge",
            id=e.id,
            label=e.label,
            type=e.type,
            summary=e.summary,
            relation=e.relation,
            source_ids=e.source_ids or [],
            target_ids=e.target_ids or [],
            source_edge_ids=e.source_edge_ids or [],
            target_edge_ids=e.target_edge_ids or [],
            domain_id=e.domain_id,
            canonical_entity_id=e.canonical_entity_id,
            properties=e.properties,
        )

    def fetch_target(self, t: AdjudicationTarget) -> Node | Edge:
        if t.kind == "node":
            got = run_awaitable_blocking(
                self._e.backend.node_get(ids=[t.id], include=["documents"])
            )
            if docs := got.get("documents"):
                return Node.model_validate_json(docs[0])
            raise ValueError(f"Node {t.id} not found")
        got = run_awaitable_blocking(
            self._e.backend.edge_get(ids=[t.id], include=["documents"])
        )
        if docs := got.get("documents"):
            return Edge.model_validate_json(docs[0])
        raise ValueError(f"Edge {t.id} not found")

    def classify_endpoint_id(self, rid: str) -> str:
        hit = run_awaitable_blocking(self._e.backend.node_get(ids=[rid]))
        if (hit.get("ids") or [None])[0] == rid:
            return "node"
        hit = run_awaitable_blocking(self._e.backend.edge_get(ids=[rid]))
        if (hit.get("ids") or [None])[0] == rid:
            return "edge"
        raise ValueError(f"Unknown endpoint id {rid!r} (not a node or edge)")

    def split_endpoints(
        self,
        src_ids: list[str] | None,
        tgt_ids: list[str] | None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        s_nodes, s_edges, t_nodes, t_edges = [], [], [], []
        for rid in src_ids or []:
            (s_nodes if self.classify_endpoint_id(rid) == "node" else s_edges).append(
                rid
            )
        for rid in tgt_ids or []:
            (t_nodes if self.classify_endpoint_id(rid) == "node" else t_edges).append(
                rid
            )
        return s_nodes, s_edges, t_nodes, t_edges

    def rebalance_same_as_edge(
        self, e: Edge, removed_node_id: str
    ) -> tuple[bool, Edge | None]:
        remain = [
            x
            for x in (e.source_ids or []) + (e.target_ids or [])
            if x != removed_node_id
        ]
        remain = list(dict.fromkeys(remain))
        if len(remain) < 2:
            return True, None
        anchor = self.choose_anchor(remain)
        e.source_ids = [anchor]
        e.target_ids = [x for x in remain if x != anchor]
        if not e.summary:
            e.summary = "Normalized same_as"
        return False, e

    def choose_anchor(self, node_ids: list[str]) -> str:
        if not node_ids:
            raise ValueError("No nodes to anchor")
        nodes = run_awaitable_blocking(
            self._e.backend.node_get(ids=node_ids, include=["documents"])
        )
        for nid, ndoc in zip(nodes.get("ids") or [], nodes.get("documents") or []):
            n = Node.model_validate_json(ndoc)
            if n.canonical_entity_id:
                return nid
        return min(node_ids)
