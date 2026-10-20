from __future__ import annotations

from kogwistar.engine_core.backend_payloads import node_doc_and_meta
from kogwistar.engine_core.utils.refs import edge_doc_and_meta
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
