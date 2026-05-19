from __future__ import annotations

import pytest

pytestmark = pytest.mark.core

from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.logical_refs import (
    LogicalRef,
    build_reference_edge_payload,
    build_reference_node_payload,
    logical_ref_from_entity,
    logical_ref_id,
)


def _span() -> Span:
    return Span.from_dummy_for_document()


def _grounding() -> Grounding:
    return Grounding(spans=[_span()])


def test_logical_ref_requires_explicit_marker_for_nodes_and_edges():
    node = Node(
        label="node",
        type="entity",
        summary="node",
        doc_id="doc-1",
        mentions=[_grounding()],
        properties={
            "target_namespace": "kg",
            "target_kind": "node",
            "target_id": "n1",
        },
        metadata={},
    )
    assert logical_ref_from_entity(node) is None

    node_ref = Node(
        label="ref",
        type="reference_pointer",
        summary="ref",
        doc_id="doc-1",
        mentions=[_grounding()],
        properties={
            "target_namespace": "kg",
            "target_kind": "node",
            "target_id": "n1",
            "refers_to_collection": "nodes",
        },
        metadata={"graph_space": "base_kg"},
    )
    assert logical_ref_from_entity(node_ref) == LogicalRef(
        target_namespace="kg",
        target_kind="node",
        target_id="n1",
    )

    edge = Edge(
        label="edge",
        type="relationship",
        summary="edge",
        relation="related",
        source_ids=["n1"],
        target_ids=["n2"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[_grounding()],
        properties={
            "target_namespace": "kg",
            "target_kind": "edge",
            "target_id": "e1",
        },
        metadata={},
        doc_id="doc-1",
    )
    assert logical_ref_from_entity(edge) is None

    edge_ref = Edge(
        label="edge ref",
        type="relationship",
        summary="edge ref",
        relation="related",
        source_ids=["n1"],
        target_ids=["n2"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[_grounding()],
        properties={
            "is_pointer": True,
            "target_namespace": "kg",
            "target_kind": "edge",
            "target_id": "e1",
            "refers_to_collection": "edges",
        },
        metadata={},
        doc_id="doc-1",
    )
    assert logical_ref_from_entity(edge_ref) == LogicalRef(
        target_namespace="kg",
        target_kind="edge",
        target_id="e1",
    )


def test_logical_ref_identity_is_namespace_sensitive():
    a = logical_ref_id(
        scope="ws:demo",
        pointer_kind="base_kg_node",
        logical_ref=LogicalRef("kg", "node", "n1"),
    )
    b = logical_ref_id(
        scope="ws:demo",
        pointer_kind="base_kg_node",
        logical_ref=LogicalRef("source", "node", "n1"),
    )
    c = logical_ref_id(
        scope="ws:demo",
        pointer_kind="base_kg_node",
        logical_ref=LogicalRef("kg", "node", "n1"),
    )

    assert a != b
    assert a == c


def test_reference_payload_builders_emit_explicit_marker_fields():
    node_payload = build_reference_node_payload(
        logical_ref=LogicalRef("source", "node", "n1"),
        pointer_kind="base_kg_node",
        pointer_id="ptr-1",
        label="Ref node",
        summary="summary",
        graph_space="base_kg",
    )
    edge_payload = build_reference_edge_payload(
        logical_ref=LogicalRef("source", "edge", "e1"),
        pointer_kind="base_kg_edge",
        pointer_id="ptr-2",
        source_ids=["ptr-src"],
        target_ids=["ptr-dst"],
        relation="related",
        label="Ref edge",
        summary="summary",
        graph_space="base_kg",
    )

    assert node_payload["type"] == "reference_pointer"
    assert node_payload["properties"]["is_pointer"] is True
    assert node_payload["properties"]["target_namespace"] == "source"
    assert node_payload["metadata"]["graph_space"] == "base_kg"

    assert edge_payload["type"] == "relationship"
    assert edge_payload["properties"]["is_pointer"] is True
    assert edge_payload["properties"]["target_kind"] == "edge"
    assert edge_payload["source_ids"] == ["ptr-src"]
    assert edge_payload["target_ids"] == ["ptr-dst"]
