from __future__ import annotations

import pytest

from kogwistar.engine_core.subsystems.write import _backend_object


def test_backend_response_decoder_accepts_known_result_fields() -> None:
    decoded = _backend_object(
        {
            "ids": ["node-1"],
            "documents": ["{}"],
            "metadatas": [{"doc_id": "document-1"}, None],
            "ignored": object(),
        }
    )

    assert decoded == {
        "ids": ["node-1"],
        "documents": ["{}"],
        "metadatas": [{"doc_id": "document-1"}, None],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ids", [1]),
        ("documents", [None]),
        ("metadatas", [{1: "not a string key"}]),
    ],
)
def test_backend_response_decoder_rejects_malformed_known_fields(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=f"backend response field '{field}'"):
        _backend_object({field: value})


def test_backend_response_decoder_treats_non_mapping_as_empty() -> None:
    assert _backend_object(None) == {}
