import pytest

from kogwistar.engine_core.models import Document, Span
from kogwistar.extraction import OcrDocSpanValidator

pytestmark = pytest.mark.core


def _ocr_document() -> Document:
    return Document.from_ocr(
        id="ocr-doc-1",
        ocr_content={
            "sample.pdf": [
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [
                        {
                            "text": "Hello world",
                            "bb_x_min": 0.0,
                            "bb_x_max": 10.0,
                            "bb_y_min": 0.0,
                            "bb_y_max": 10.0,
                            "cluster_number": 0,
                        }
                    ],
                    "non_text_objects": [],
                    "is_empty_page": False,
                    "printed_page_number": "1",
                    "meaningful_ordering": [0],
                    "page_x_min": 0.0,
                    "page_x_max": 100.0,
                    "page_y_min": 0.0,
                    "page_y_max": 100.0,
                    "estimated_rotation_degrees": 0.0,
                    "incomplete_words_on_edge": False,
                    "incomplete_text": False,
                    "data_loss_likelihood": 0.0,
                    "scan_quality": "high",
                    "contains_table": False,
                }
            ]
        },
        type="ocr",
    )


def _ocr_span(*, excerpt: str) -> Span:
    return Span.model_validate(
        {
            "collection_page_url": "document_collection/ocr-doc-1",
            "document_page_url": "doc://ocr-doc-1#p1_c0",
            "doc_id": "ocr-doc-1",
            "insertion_method": "document_parser_v1",
            "page_number": 1,
            "start_char": 0,
            "end_char": len(excerpt),
            "excerpt": excerpt,
            "context_before": "",
            "context_after": "",
            "chunk_id": None,
            "source_cluster_id": "p1_c0",
        },
        context={"insertion_method": "document_parser_v1"},
    )


def test_ocr_document_span_validation_uses_source_map() -> None:
    doc = _ocr_document()
    span = _ocr_span(excerpt="Hello")

    assert doc.type == "ocr_document"
    assert doc.get_content_by_span(span) == "Hello"

    validator = OcrDocSpanValidator()
    result = validator.validate_span(span=span, doc=doc)

    assert result["correctness"] is True
    assert result["excerpt_from_start_end_index"] == "Hello"


def test_ocr_document_span_validation_detects_mismatch() -> None:
    doc = _ocr_document()
    span = _ocr_span(excerpt="Hello!")

    validator = OcrDocSpanValidator()
    result = validator.validate_span(span=span, doc=doc)

    assert result["correctness"] is False
    assert result["excerpt_from_start_end_index"] == "Hello "


def test_ocr_document_span_validation_rebuilds_missing_source_map() -> None:
    doc = _ocr_document()
    doc.source_map = None
    span = _ocr_span(excerpt="Hello")

    validator = OcrDocSpanValidator()
    result = validator.validate_span(span=span, doc=doc)

    assert result["correctness"] is True


def test_ocr_document_span_validation_rebuilds_missing_source_map_from_json_string() -> None:
    doc = _ocr_document()
    doc.source_map = None
    doc.content = doc.model_dump(mode="json")["content"]
    span = _ocr_span(excerpt="Hello")

    validator = OcrDocSpanValidator()
    result = validator.validate_span(span=span, doc=doc)

    assert result["correctness"] is True


def test_ocr_document_span_validation_tolerates_common_ocr_normalization() -> None:
    doc = _ocr_document()
    doc.source_map = None
    doc.content = doc.model_dump(mode="json")["content"]
    span = _ocr_span(excerpt="Hello world")
    span = span.model_copy(update={"excerpt": "Hello  world"})

    validator = OcrDocSpanValidator()
    result = validator.validate_span(span=span, doc=doc)

    assert result["correctness"] is True
