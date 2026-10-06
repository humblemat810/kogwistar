from __future__ import annotations

from kogwistar.engine_core.utils.refs import edge_doc_and_meta
from tests._helpers.graph_builders import build_relationship_edge


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
