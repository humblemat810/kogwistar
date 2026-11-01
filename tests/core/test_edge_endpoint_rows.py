from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from kogwistar.engine_core.edge_endpoint_rows import edge_endpoint_rows
from kogwistar.json_types import JsonValue

pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


@dataclass
class _Edge:
    source_ids: list[str]
    target_ids: list[str]
    source_edge_ids: list[str] | None = None
    target_edge_ids: list[str] | None = None
    doc_id: str | None = "doc-1"
    relation: str | None = "supports"
    metadata: dict[str, JsonValue] | None = None

    def safe_get_id(self) -> str:
        return "edge-1"


def test_endpoint_projection_accepts_structural_edge_protocol() -> None:
    rows = edge_endpoint_rows(
        _Edge(source_ids=["source"], target_ids=["target"], source_edge_ids=["parent"])
    )

    assert [row["endpoint_type"] for row in rows] == ["node", "node", "edge"]
    assert rows[2]["id"] == "edge-1::src::edge::parent"


def test_endpoint_rows_match_shared_rust_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "contracts/golden/edge-endpoint-rows-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["schema_version"] == 1
    edge = fixture["edge"]
    actual = edge_endpoint_rows(
        _Edge(
            source_ids=edge["source_ids"],
            target_ids=edge["target_ids"],
            source_edge_ids=edge["source_edge_ids"],
            target_edge_ids=edge["target_edge_ids"],
            doc_id=edge["doc_id"],
            relation=edge["relation"],
            metadata=edge["metadata"],
        )
    )
    assert actual == fixture["expected_rows"]
