"""Construct typed graph artifacts emitted by workflow execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from kogwistar.engine_core.models import Grounding, MentionVerification
from kogwistar.json_types import JsonValue
from kogwistar.runtime.models import WorkflowRuntimeEdge


class TraceSpanPayload(TypedDict):
    """Payload for a workflow-generated source span."""

    collection_page_url: str
    document_page_url: str
    doc_id: str
    insertion_method: str
    page_number: int
    start_char: int
    end_char: int
    excerpt: str
    context_before: str
    context_after: str
    chunk_id: str | None
    source_cluster_id: str | None
    verification: MentionVerification


def _make_trace_span(
    *, conversation_id: str, excerpt: str, doc_id: str
) -> TraceSpanPayload:
    """Build the deterministic span attached to workflow trace artifacts."""
    return {
        "collection_page_url": f"conversation/{conversation_id}",
        "document_page_url": f"conversation/{conversation_id}",
        "doc_id": doc_id,
        "insertion_method": "workflow_trace",
        "page_number": 1,
        "start_char": 0,
        "end_char": len(excerpt),
        "excerpt": excerpt,
        "context_before": "",
        "context_after": "",
        "chunk_id": None,
        "source_cluster_id": None,
        "verification": MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="workflow trace",
        ),
    }


def _make_runtime_edge(
    *,
    edge_id: str,
    relation: str,
    label: str,
    summary: str,
    doc_id: str,
    conversation_id: str,
    source_ids: list[str],
    target_ids: list[str],
    mentions: list[Grounding],
    run_id: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> WorkflowRuntimeEdge:
    """Build a workflow trace edge with JSON-safe, domain-owned metadata."""
    edge_metadata: dict[str, JsonValue] = {
        "entity_type": "conversation_edge",
        "relation": relation,
        "conversation_id": conversation_id,
    }
    if run_id is not None:
        edge_metadata["run_id"] = run_id
    if metadata:
        edge_metadata.update(metadata)
    return WorkflowRuntimeEdge(
        id=edge_id,
        source_ids=source_ids,
        target_ids=target_ids,
        relation=relation,
        label=label,
        type="relationship",
        summary=summary,
        doc_id=doc_id,
        mentions=mentions,
        domain_id=None,
        canonical_entity_id=None,
        properties={},
        embedding=None,
        metadata=edge_metadata,
        source_edge_ids=[],
        target_edge_ids=[],
    )
