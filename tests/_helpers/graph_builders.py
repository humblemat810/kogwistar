from __future__ import annotations

from typing import Any, Sequence

from kogwistar.engine_core.models import Edge, Grounding, MentionVerification, Node, Span


def mk_document_span(doc_id: str) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="test",
        ),
        collection_page_url="N/A",
        document_page_url="N/A",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
    )


def mk_conversation_span(doc_id: str) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="test",
        ),
        collection_page_url="N/A",
        document_page_url="N/A",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
    )


def mk_grounding_from_span(span: Span) -> Grounding:
    return Grounding(spans=[span])


def mk_document_grounding(doc_id: str) -> Grounding:
    return mk_grounding_from_span(mk_document_span(doc_id))


def mk_conversation_grounding(doc_id: str) -> Grounding:
    return mk_grounding_from_span(mk_conversation_span(doc_id))


def mk_excerpt_span(
    excerpt: str,
    *,
    doc_id: str = "D:test",
    insertion_method: str = "test",
    collection_page_url: str = "url",
    document_page_url: str = "url",
    page_number: int = 1,
    start_char: int = 0,
    context_before: str = "",
    context_after: str = "",
    notes: str = "test",
) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes=notes,
        ),
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        insertion_method=insertion_method,
        page_number=page_number,
        start_char=start_char,
        end_char=start_char + len(excerpt),
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
    )


def build_entity_node(
    *,
    node_id: str,
    doc_id: str,
    label: str | None = None,
    summary: str | None = None,
    entity_type: str = "kg_entity",
    embedding: Sequence[float] | None = None,
    mentions: list[Grounding] | None = None,
    metadata: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
) -> Node:
    base_meta = {"level_from_root": 0, "entity_type": entity_type}
    if metadata:
        base_meta.update(metadata)
    return Node(
        id=node_id,
        label=label or f"Node {node_id}",
        type="entity",
        summary=summary or f"Summary {node_id}",
        doc_id=doc_id,
        mentions=mentions or [mk_document_grounding(doc_id)],
        metadata=base_meta,
        embedding=list(embedding) if embedding is not None else None,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=properties,
    )


def build_relationship_edge(
    *,
    edge_id: str,
    src: str,
    tgt: str,
    doc_id: str,
    label: str | None = None,
    summary: str | None = None,
    relation: str = "related_to",
    entity_type: str = "kg_relation",
    embedding: Sequence[float] | None = None,
    mentions: list[Grounding] | None = None,
    metadata: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    source_edge_ids: list[str] | None = None,
    target_edge_ids: list[str] | None = None,
) -> Edge:
    base_meta = {"level_from_root": 0, "entity_type": entity_type}
    if metadata:
        base_meta.update(metadata)
    return Edge(
        id=edge_id,
        label=label or f"Edge {edge_id}",
        type="relationship",
        summary=summary or f"Summary {edge_id}",
        relation=relation,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=source_edge_ids,
        target_edge_ids=target_edge_ids,
        doc_id=doc_id,
        mentions=mentions or [mk_document_grounding(doc_id)],
        metadata=base_meta,
        embedding=list(embedding) if embedding is not None else None,
        domain_id=None,
        canonical_entity_id=None,
        properties=properties,
    )
