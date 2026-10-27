from __future__ import annotations

import json
from pathlib import Path

import pytest

from kogwistar.engine_core import (
    GraphExtractionContractError,
    validate_graph_extraction_request,
)


def _fixture() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/graph-extraction-v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_python_accepts_shared_graph_extraction_fixture() -> None:
    request = validate_graph_extraction_request(_fixture())

    assert request.contract_version == 1
    assert request.doc_id == "doc-1"
    assert request.nodes[0]["references"][0]["excerpt"] == "Alpha"
    assert request.edges[0]["source_ids"] == ["nn:first"]


def test_graph_extraction_validates_exact_text_spans_with_unicode_offsets() -> None:
    request = validate_graph_extraction_request(_fixture())
    request.validate_source_spans({"doc-1": "Alpha connects to Beta."})

    unicode_payload = {
        "contract_version": 1,
        "doc_id": "unicode",
        "content": "naïve猫",
        "nodes": [
            {
                "label": "猫",
                "type": "entity",
                "summary": "猫",
                "references": [{"start_char": 5, "end_char": 6, "excerpt": "猫"}],
            }
        ],
    }
    unicode_request = validate_graph_extraction_request(unicode_payload)
    unicode_request.validate_source_spans({"unicode": "naïve猫"})

    altered = _fixture()
    node = altered["nodes"][0]
    assert isinstance(node, dict)
    reference = node["references"][0]
    assert isinstance(reference, dict)
    reference["excerpt"] = "wrong"
    with pytest.raises(GraphExtractionContractError, match="do not match"):
        validate_graph_extraction_request(altered).validate_source_spans(
            {"doc-1": "Alpha connects to Beta."}
        )


def test_graph_extraction_source_validation_fails_closed_for_missing_text_and_flat_ids() -> None:
    request = validate_graph_extraction_request(_fixture())
    with pytest.raises(GraphExtractionContractError, match="unavailable"):
        request.validate_source_spans({})

    payload = _fixture()
    node = payload["nodes"][0]
    assert isinstance(node, dict)
    node["mentions"] = [{"span_ids": ["span-1"]}]
    node.pop("references")
    with pytest.raises(GraphExtractionContractError, match="root span records"):
        validate_graph_extraction_request(payload).validate_source_spans(
            {"doc-1": "Alpha connects to Beta."}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", 2),
        ("doc_id", "  "),
        ("doc_type", "binary"),
        ("insertion_method", " "),
        ("nodes", ["not-an-object"]),
    ],
)
def test_graph_extraction_rejects_invalid_envelope(field: str, value: object) -> None:
    payload = _fixture()
    payload[field] = value

    with pytest.raises(GraphExtractionContractError):
        validate_graph_extraction_request(payload)


def test_graph_extraction_rejects_unknown_root_fields() -> None:
    payload = _fixture()
    payload["unexpected"] = True

    with pytest.raises(GraphExtractionContractError):
        validate_graph_extraction_request(payload)
