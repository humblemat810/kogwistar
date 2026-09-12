from __future__ import annotations

import json

import pytest

from kogwistar.conversation.models import ConversationEdge
from kogwistar.engine_core.models import Edge
from kogwistar.engine_core.subsystems.read import ReadSubsystem
from kogwistar.entity_registry import pick_edge_type
from kogwistar.runtime.models import WorkflowEdge


pytestmark = pytest.mark.core


def _stored_edge_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": "relationship",
        "relation": "related_to",
        "source_ids": ["n-source"],
        "target_ids": ["n-target"],
        "node_endpoint_count": 2,
        "total_endpoint_count": 2,
    }
    metadata.update(overrides)
    return metadata


def _stored_edge_document() -> str:
    return json.dumps(
        {
            "id": "edge-1",
            "label": "Edge 1",
            "type": "relationship",
            "summary": "A relationship",
            "relation": "related_to",
            "source_ids": ["n-source"],
            "target_ids": ["n-target"],
            "source_edge_ids": [],
            "target_edge_ids": [],
            "doc_id": "doc-1",
            "mentions": [
                {
                    "spans": [
                        {
                            "collection_page_url": "https://example.test/doc-1",
                            "document_page_url": "https://example.test/doc-1",
                            "doc_id": "doc-1",
                            "insertion_method": "test",
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 1,
                            "excerpt": "A",
                            "context_before": "",
                            "context_after": "",
                        }
                    ]
                }
            ],
            "domain_id": None,
            "canonical_entity_id": None,
            "properties": None,
        }
    )


def test_workflow_fallback_does_not_claim_legacy_relationship_edge() -> None:
    selected = pick_edge_type(
        metadata=_stored_edge_metadata(),
        fallback=WorkflowEdge,
    )

    assert selected is Edge


def test_workflow_edge_metadata_still_selects_workflow_edge() -> None:
    selected = pick_edge_type(
        metadata=_stored_edge_metadata(
            entity_type="workflow_edge",
            workflow_id="workflow-1",
            wf_predicate=None,
        ),
        fallback=WorkflowEdge,
    )

    assert selected is WorkflowEdge


def test_conversation_relationship_fallback_is_unchanged() -> None:
    selected = pick_edge_type(
        metadata=_stored_edge_metadata(entity_type="relationship"),
        fallback=ConversationEdge,
    )

    assert selected is ConversationEdge


def test_read_reconstructs_legacy_relationship_in_workflow_graph() -> None:
    read = object.__new__(ReadSubsystem)
    read._e = type("Engine", (), {"kg_graph_type": "workflow"})()

    result = read.edges_from_single_or_id_query_result(
        {
            "documents": [_stored_edge_document()],
            "metadatas": [_stored_edge_metadata()],
            "embeddings": [None],
        },
        edge_type=WorkflowEdge,
    )

    assert len(result) == 1
    assert isinstance(result[0], Edge)
    assert not isinstance(result[0], WorkflowEdge)
    assert result[0].metadata["type"] == "relationship"
