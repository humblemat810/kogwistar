from __future__ import annotations

from dataclasses import dataclass

import pytest

from kogwistar.engine_core.edge_endpoint_rows import edge_endpoint_rows


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


@dataclass
class _Edge:
    source_ids: list[str]
    target_ids: list[str]
    source_edge_ids: list[str] | None = None
    target_edge_ids: list[str] | None = None
    doc_id: str | None = "doc-1"
    relation: str | None = "supports"

    def safe_get_id(self) -> str:
        return "edge-1"


def test_endpoint_projection_accepts_structural_edge_protocol() -> None:
    rows = edge_endpoint_rows(
        _Edge(source_ids=["source"], target_ids=["target"], source_edge_ids=["parent"])
    )

    assert [row["endpoint_type"] for row in rows] == ["node", "node", "edge"]
    assert rows[2]["id"] == "edge-1::src::edge::parent"
