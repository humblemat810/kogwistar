"""Pure structural endpoint serialization shared by projection adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from ..json_types import JsonObject, JsonValue


class EdgeEndpointLike(Protocol):
    """Structural edge surface required by endpoint projection adapters."""

    @property
    def id(self) -> str | None: ...

    @property
    def source_ids(self) -> Sequence[str] | None: ...

    @property
    def target_ids(self) -> Sequence[str] | None: ...

    @property
    def source_edge_ids(self) -> Sequence[str] | None: ...

    @property
    def target_edge_ids(self) -> Sequence[str] | None: ...

    @property
    def doc_id(self) -> str | None: ...

    @property
    def relation(self) -> str | None: ...

    @property
    def metadata(self) -> Mapping[str, JsonValue] | None: ...

def edge_endpoint_rows(edge: EdgeEndpointLike) -> list[JsonObject]:
    """Return deterministic, non-semantic endpoint rows for an edge."""
    edge_id = edge.id
    if not edge_id:
        legacy_get_id = getattr(edge, "safe_get_id", None)
        if callable(legacy_get_id):
            edge_id = legacy_get_id()
    if not isinstance(edge_id, str) or not edge_id:
        raise ValueError("edge endpoint projection requires a non-empty edge id")
    rows: list[JsonObject] = []
    for role, endpoint_ids, endpoint_type in (
        ("src", edge.source_ids or [], "node"),
        ("tgt", edge.target_ids or [], "node"),
        ("src", getattr(edge, "source_edge_ids", []) or [], "edge"),
        ("tgt", getattr(edge, "target_edge_ids", []) or [], "edge"),
    ):
        for endpoint_id in endpoint_ids:
            rows.append({
                "id": f"{edge_id}::{role}::{endpoint_type}::{endpoint_id}",
                "edge_id": edge_id,
                "endpoint_id": str(endpoint_id),
                "endpoint_type": endpoint_type,
                "role": role,
                "doc_id": edge.doc_id,
                "relation": edge.relation,
                "causal_type": (edge.metadata or {}).get("causal_type"),
            })
    return [
        {key: value for key, value in row.items() if value is not None}
        for row in rows
    ]


__all__ = ["EdgeEndpointLike", "edge_endpoint_rows"]
