from __future__ import annotations

import json
from asyncio import run
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.edge_endpoint_rows import edge_endpoint_rows
from kogwistar.engine_core.models import PureChromaEdge
from kogwistar.graph_query import GraphQuery
from kogwistar.json_types import JsonValue

pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


@dataclass
class _Edge:
    id: str
    source_ids: list[str]
    target_ids: list[str]
    source_edge_ids: list[str] | None = None
    target_edge_ids: list[str] | None = None
    doc_id: str | None = "doc-1"
    relation: str | None = "supports"
    metadata: dict[str, JsonValue] | None = None


def test_endpoint_projection_accepts_structural_edge_protocol() -> None:
    rows = edge_endpoint_rows(
        _Edge(
            id="edge-1",
            source_ids=["source"],
            target_ids=["target"],
            source_edge_ids=["parent"],
        )
    )

    assert [row["endpoint_type"] for row in rows] == ["node", "node", "edge"]
    assert rows[2]["id"] == "edge-1::src::edge::parent"


def test_endpoint_projection_accepts_pure_chroma_edge_model() -> None:
    edge = PureChromaEdge(
        id="pure-edge",
        label="related",
        type="relationship",
        summary="Connects two nodes",
        relation="supports",
        domain_id=None,
        canonical_entity_id=None,
        properties={},
        embedding=None,
        doc_id=None,
        source_ids=["source"],
        target_ids=["target"],
        source_edge_ids=[],
        target_edge_ids=[],
    )

    rows = edge_endpoint_rows(edge)

    assert [row["id"] for row in rows] == [
        "pure-edge::src::node::source",
        "pure-edge::tgt::node::target",
    ]


def test_endpoint_projection_preserves_legacy_safe_id_fallback() -> None:
    edge = _Edge(id="", source_ids=["source"], target_ids=[])
    edge.safe_get_id = lambda: "legacy-edge"  # type: ignore[attr-defined]

    assert edge_endpoint_rows(edge)[0]["id"] == "legacy-edge::src::node::source"


def test_endpoint_projection_rejects_edges_without_an_identity() -> None:
    with pytest.raises(ValueError, match="non-empty edge id"):
        edge_endpoint_rows(_Edge(id="", source_ids=[], target_ids=[]))


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
            id=edge["id"],
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


def test_graph_query_sync_endpoint_merge_preserves_cross_kind_same_id() -> None:
    backend_row = {
        "edge_id": "parent",
        "endpoint_id": "shared-id",
        "endpoint_type": "node",
        "role": "src",
    }

    class Read:
        def get_edge_endpoints(self, **_kwargs: object) -> dict[str, list[str]]:
            return {"documents": [json.dumps(backend_row)]}

        def get_nodes(self, **_kwargs: object) -> list[object]:
            return []

        def get_edges(
            self, ids: list[str] | None = None, **_kwargs: object
        ) -> list[object]:
            return [SimpleNamespace(id=edge_id) for edge_id in ids or []]

    query = GraphQuery(
        cast(
            GraphKnowledgeEngine,
            SimpleNamespace(
                read=Read(), persistence_mode="two_stage", backend=object()
            ),
        )
    )
    edge_endpoint_row = {
        **backend_row,
        "endpoint_type": "edge",
        "id": "parent::src::edge::shared-id",
    }
    query._stage1_endpoint_rows = lambda **_kwargs: [  # type: ignore[method-assign]
        edge_endpoint_row,
        dict(edge_endpoint_row),
    ]

    rows = query._endpoint_rows({"edge_id": "parent"}, edge_id="parent")

    assert {row["endpoint_type"] for row in rows} == {"node", "edge"}
    assert len(rows) == 2
    assert query.neighbors("parent") == {
        "nodes": {"shared-id"},
        "edges": {"shared-id"},
    }


def test_graph_query_async_endpoint_merge_preserves_cross_kind_same_id() -> None:
    backend_row = {
        "edge_id": "parent",
        "endpoint_id": "shared-id",
        "endpoint_type": "node",
        "role": "src",
    }

    class Backend:
        async def async_call(
            self, collection: str, method: str, **kwargs: object
        ) -> dict[str, list[str]]:
            if collection == "edge_endpoints" and method == "get":
                return {"documents": [json.dumps(backend_row)]}
            if collection == "edge" and method == "get":
                return {"ids": ["parent"]}
            return {"ids": []}

    class Read:
        def get_edges(
            self, ids: list[str] | None = None, **_kwargs: object
        ) -> list[object]:
            return [SimpleNamespace(id=edge_id) for edge_id in ids or []]

    query = GraphQuery(
        cast(
            GraphKnowledgeEngine,
            SimpleNamespace(
                read=Read(), persistence_mode="two_stage", backend=Backend()
            ),
        )
    )
    edge_endpoint_row = {
        **backend_row,
        "endpoint_type": "edge",
        "id": "parent::src::edge::shared-id",
    }

    async def staged_rows(**_kwargs: object) -> list[dict[str, str]]:
        return [edge_endpoint_row, dict(edge_endpoint_row)]

    query._stage1_endpoint_rows_async = staged_rows  # type: ignore[method-assign]

    rows = run(query._endpoint_rows_async({"edge_id": "parent"}, edge_id="parent"))

    assert {row["endpoint_type"] for row in rows} == {"node", "edge"}
    assert len(rows) == 2
    assert run(query.neighbors_async("parent")) == {
        "nodes": {"shared-id"},
        "edges": {"shared-id"},
    }
