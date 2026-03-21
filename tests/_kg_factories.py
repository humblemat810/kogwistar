from __future__ import annotations

from typing import Any

from kogwistar.engine_core.models import (
    Document,
    Grounding,
    MentionVerification,
    Span,
)


def kg_verification(
    *,
    method: str = "heuristic",
    is_verified: bool = False,
    notes: str | None = None,
    score: float = 0.9,
) -> MentionVerification:
    return MentionVerification(
        method=method,
        is_verified=is_verified,
        notes=notes,
        score=score,
    )


def kg_document(
    *,
    doc_id: str,
    content: str | dict[str, Any],
    source: str,
    doc_type: str = "text",
    metadata: dict[str, Any] | None = None,
    domain_id: str | None = None,
    processed: bool = False,
    embeddings: Any = None,
    source_map: dict[str, Any] | None = None,
) -> Document:
    doc_metadata = dict(metadata or {})
    doc_metadata.setdefault("source", source)
    return Document(
        id=doc_id,
        content=content,
        type=doc_type,
        metadata=doc_metadata,
        domain_id=domain_id,
        processed=processed,
        embeddings=embeddings,
        source_map=source_map,
    )


def kg_span(
    doc_id: str,
    *,
    excerpt: str = "x",
    start_char: int = 0,
    end_char: int | None = None,
    page_number: int = 1,
    collection_page_url: str = "c",
    document_page_url: str | None = None,
    insertion_method: str = "pytest-manual",
    context_before: str = "",
    context_after: str = "",
    chunk_id: str | None = None,
    source_cluster_id: str | None = None,
    verification: MentionVerification | None = None,
) -> Span:
    if document_page_url is None:
        document_page_url = f"document/{doc_id}"
    if end_char is None:
        end_char = max(start_char + len(excerpt), start_char + 1)
    return Span(
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
        chunk_id=chunk_id,
        source_cluster_id=source_cluster_id,
        verification=verification or kg_verification(),
    )


def kg_grounding(doc_id: str, **span_kwargs: Any) -> Grounding:
    return Grounding(spans=[kg_span(doc_id, **span_kwargs)])


def kg_llm_grounding_payload(doc_id: str, **span_kwargs: Any) -> dict[str, Any]:
    span = kg_span(doc_id, **span_kwargs)
    return {
        "spans": [
            {
                "collection_page_url": span.collection_page_url,
                "document_page_url": span.document_page_url,
                "doc_id": span.doc_id,
                "page_number": span.page_number,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "excerpt": span.excerpt,
                "context_before": span.context_before,
                "context_after": span.context_after,
                "chunk_id": span.chunk_id,
                "source_cluster_id": span.source_cluster_id,
            }
        ]
    }
