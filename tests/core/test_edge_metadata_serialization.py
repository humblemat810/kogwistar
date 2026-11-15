from __future__ import annotations

import pytest

from kogwistar.engine_core.backend_payloads import node_doc_and_meta
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.utils.refs import edge_doc_and_meta
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge


def test_edge_doc_and_meta_preserves_backend_filter_metadata() -> None:
    edge = build_relationship_edge(
        edge_id="edge-1",
        src="source-1",
        tgt="target-1",
        doc_id="document-1",
        label="supports",
        relation="SUPPORTS",
        metadata={
            "workspace_id": "workspace-1",
            "graph_space": "base_kg",
            "security_scope": "tenant-1",
        },
    )

    _document, metadata = edge_doc_and_meta(edge)

    assert metadata["workspace_id"] == "workspace-1"
    assert metadata["graph_space"] == "base_kg"
    assert metadata["security_scope"] == "tenant-1"
    assert metadata["source_ids"] == '["source-1"]'
    assert metadata["target_ids"] == '["target-1"]'


def test_node_doc_and_meta_does_not_mutate_model_metadata() -> None:
    node = build_entity_node(
        node_id="node-1",
        doc_id="document-1",
        metadata={"workspace_id": "workspace-1", "security_scope": "tenant-1"},
    )
    original_metadata = dict(node.metadata)

    _document, metadata = node_doc_and_meta(node)

    assert node.metadata == original_metadata
    assert metadata is not node.metadata
    assert metadata["workspace_id"] == "workspace-1"
    assert metadata["security_scope"] == "tenant-1"


def test_engine_projection_metadata_uses_authoritative_namespace() -> None:
    engine = GraphKnowledgeEngine(
        backend_factory=build_fake_backend,
        namespace="workspace-kg",
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    node = build_entity_node(node_id="node", doc_id="doc")
    edge = build_relationship_edge(
        edge_id="edge", src="source", tgt="target", doc_id="doc"
    )

    _node_document, node_metadata = engine.write.node_doc_and_meta(node)
    _edge_document, edge_metadata = engine.write.edge_doc_and_meta(edge)

    assert node_metadata["namespace"] == "workspace-kg"
    assert edge_metadata["namespace"] == "workspace-kg"
    assert "namespace" not in (node.metadata or {})
    assert "namespace" not in (edge.metadata or {})

    conflicting_node = build_entity_node(
        node_id="conflicting-node",
        doc_id="doc",
        metadata={"namespace": "forged-namespace"},
    )
    with pytest.raises(ValueError, match="conflicts with the engine namespace"):
        engine.write.node_doc_and_meta(conflicting_node)

    conflicting_edge = build_relationship_edge(
        edge_id="conflicting-edge",
        src="source",
        tgt="target",
        doc_id="doc",
        metadata={"namespace": "forged-namespace"},
    )
    with pytest.raises(ValueError, match="conflicts with the engine namespace"):
        engine.write.edge_doc_and_meta(conflicting_edge)
