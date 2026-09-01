"""Pure structural endpoint serialization shared by projection adapters."""

from __future__ import annotations

from typing import Any


def edge_endpoint_rows(edge: Any) -> list[dict[str, Any]]:
    """Return deterministic, non-semantic endpoint rows for an edge."""
    edge_id = str(edge.safe_get_id())
    rows: list[dict[str, Any]] = []
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
            })
    return [
        {key: value for key, value in row.items() if value is not None}
        for row in rows
    ]


__all__ = ["edge_endpoint_rows"]
