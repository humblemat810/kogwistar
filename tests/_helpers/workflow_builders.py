from __future__ import annotations

from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode


def build_workflow_span(
    *,
    doc_id: str = "test",
    insertion_method: str = "test",
    excerpt: str = "test",
) -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=1,
        start_char=0,
        end_char=len(excerpt),
        excerpt=excerpt,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="test",
        ),
    )


def build_workflow_grounding(
    *,
    doc_id: str = "test",
    insertion_method: str = "test",
    excerpt: str = "test",
) -> Grounding:
    return Grounding(
        spans=[
            build_workflow_span(
                doc_id=doc_id,
                insertion_method=insertion_method,
                excerpt=excerpt,
            )
        ]
    )


def build_workflow_node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    label: str | None = None,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
    wf_join: bool = False,
) -> WorkflowNode:
    metadata = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": op,
        "wf_start": start,
        "wf_terminal": terminal,
        "wf_version": "v1",
    }
    if fanout:
        metadata["wf_fanout"] = True
    if wf_join:
        metadata["wf_join"] = True
    return WorkflowNode(
        id=node_id,
        label=label or node_id,
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[build_workflow_grounding()],
        properties={},
        metadata=metadata,
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


def build_workflow_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    label: str = "wf_next",
    predicate: str | None = None,
    priority: int = 100,
    is_default: bool = True,
    multiplicity: str = "one",
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label=label,
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[build_workflow_grounding()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": multiplicity,
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )
