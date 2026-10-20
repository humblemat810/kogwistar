from __future__ import annotations

from kogwistar.engine_core.backend_payloads import decode_backend_ids_response


def test_decode_backend_ids_response_filters_non_string_ids() -> None:
    assert decode_backend_ids_response({"ids": ["doc-a", 4, None, "doc-b"]}) == [
        "doc-a",
        "doc-b",
    ]


def test_decode_backend_ids_response_preserves_valid_empty_result() -> None:
    assert decode_backend_ids_response({"ids": []}) == []


def test_decode_backend_ids_response_rejects_malformed_shapes() -> None:
    assert decode_backend_ids_response(None) is None
    assert decode_backend_ids_response({"ids": "doc-a"}) is None
    assert decode_backend_ids_response({}) is None
