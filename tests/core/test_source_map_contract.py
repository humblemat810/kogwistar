from __future__ import annotations

import pytest

from kogwistar.document_types import Chunk
from kogwistar.runtime.source_map_contract import (
    SourceMapContractError,
    source_map_from_wire,
    source_map_to_wire,
)


def test_source_map_contract_round_trips_unicode_chunks_deterministically() -> None:
    source = "猫🙂 alpha beta"
    chunks = {
        "chunk-2": Chunk("doc", 8, len(source), " beta"),
        "chunk-1": Chunk("doc", 0, 8, "猫🙂 alpha"),
    }

    wire = source_map_to_wire(doc_id="doc", chunks=chunks, source_text=source)
    assert [chunk["chunk_id"] for chunk in wire["chunks"]] == ["chunk-1", "chunk-2"]
    assert source_map_from_wire(wire, source_text=source) == chunks


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: {**payload, "unexpected": True}, "unknown"),
        (
            lambda payload: {
                **payload,
                "chunks": [{**payload["chunks"][0], "text": "altered"}],
            },
            "does not match",
        ),
    ],
)
def test_source_map_contract_rejects_untrusted_payloads(mutator, message: str) -> None:
    source = "alpha"
    payload = {
        "contract_version": 1,
        "doc_id": "doc",
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "doc_id": "doc",
                "start_char": 0,
                "end_char": 5,
                "text": source,
            }
        ],
    }
    with pytest.raises(SourceMapContractError, match=message):
        source_map_from_wire(mutator(payload), source_text=source)
