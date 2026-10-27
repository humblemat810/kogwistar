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
