import pytest

from kogwistar.document_types import Chunk
from kogwistar.engine_core.models import (
    MentionVerification,
    Span,
    Chunk as ModelChunk,
    MentionVerification as ModelMentionVerification,
    Span as ModelSpan,
)
from kogwistar.engine_core.span_models import (
    MentionVerification as SpanMentionVerification,
    Span as ProvenanceSpan,
)
from kogwistar.splitter import Chunk as SplitterChunk


def test_document_chunk_and_span_imports_keep_identity() -> None:
    assert Chunk is ModelChunk is SplitterChunk
    assert MentionVerification is ModelMentionVerification is SpanMentionVerification
    assert Span is ModelSpan is ProvenanceSpan


def test_span_source_range_validation_remains_enforced() -> None:
    with pytest.raises(ValueError, match="end_char must be > start_char"):
        Span(
            collection_page_url="collection",
            document_page_url="document",
            doc_id="doc",
            insertion_method="system",
            start_char=4,
            end_char=4,
        )
