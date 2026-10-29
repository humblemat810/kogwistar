"""Python adapter for the versioned Rust source-map wire contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kogwistar.document_types import Chunk
from kogwistar.json_types import JsonValue

SOURCE_MAP_WIRE_VERSION = 1
_SOURCE_MAP_FIELDS = frozenset({"contract_version", "doc_id", "chunks"})
_SOURCE_MAP_CHUNK_FIELDS = frozenset(
    {"chunk_id", "doc_id", "start_char", "end_char", "text"}
)


class SourceMapContractError(ValueError):
    """Raised when a source-map payload cannot cross the language seam."""


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceMapContractError(f"{field} must be non-empty text")
    return value


def _required_uint(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceMapContractError(f"{field} must be a non-negative integer")
    return value


def source_map_to_wire(
    *,
    doc_id: str,
    chunks: Mapping[str, Chunk],
    source_text: str,
) -> dict[str, JsonValue]:
    """Project textual chunks to the deterministic Rust-compatible payload."""
    document_id = _required_text(doc_id, field="doc_id")
    entries: list[JsonValue] = []
    for chunk_id, chunk in sorted(
        chunks.items(), key=lambda item: (item[1].end_char, item[1].start_char, item[0])
    ):
        identifier = _required_text(chunk_id, field="chunk_id")
        if chunk.doc_id != document_id:
            raise SourceMapContractError("chunk doc_id must match map doc_id")
        start = _required_uint(chunk.start_char, field="start_char")
        end = _required_uint(chunk.end_char, field="end_char")
        if end <= start or end > len(source_text):
            raise SourceMapContractError("chunk bounds are invalid")
        if source_text[start:end] != chunk.text:
            raise SourceMapContractError("chunk text does not match source text")
        entries.append(
            {
                "chunk_id": identifier,
                "doc_id": document_id,
                "start_char": start,
                "end_char": end,
                "text": chunk.text,
            }
        )
    return {
        "contract_version": SOURCE_MAP_WIRE_VERSION,
        "doc_id": document_id,
        "chunks": entries,
    }


def source_map_from_wire(
    payload: Mapping[str, object], *, source_text: str
) -> dict[str, Chunk]:
    """Validate and decode a Rust-compatible source-map payload."""
    unknown = set(payload) - _SOURCE_MAP_FIELDS
    if unknown:
        raise SourceMapContractError(f"unknown source-map fields: {sorted(unknown)!r}")
    version = payload.get("contract_version", SOURCE_MAP_WIRE_VERSION)
    if version != SOURCE_MAP_WIRE_VERSION:
        raise SourceMapContractError("unsupported source-map contract version")
    document_id = _required_text(payload.get("doc_id"), field="doc_id")
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, Sequence) or isinstance(raw_chunks, (str, bytes)):
        raise SourceMapContractError("chunks must be a list")

    decoded: dict[str, Chunk] = {}
    previous_end = 0
    for raw in raw_chunks:
        if not isinstance(raw, Mapping):
            raise SourceMapContractError("each chunk must be an object")
        unknown_chunk = set(raw) - _SOURCE_MAP_CHUNK_FIELDS
        if unknown_chunk:
            raise SourceMapContractError(
                f"unknown source-map chunk fields: {sorted(unknown_chunk)!r}"
            )
        chunk_id = _required_text(raw.get("chunk_id"), field="chunk_id")
        if chunk_id in decoded:
            raise SourceMapContractError("chunk IDs must be unique")
        chunk_doc_id = _required_text(raw.get("doc_id"), field="chunk doc_id")
        if chunk_doc_id != document_id:
            raise SourceMapContractError("chunk doc_id must match map doc_id")
        start = _required_uint(raw.get("start_char"), field="start_char")
        end = _required_uint(raw.get("end_char"), field="end_char")
        if end <= start or end <= previous_end or end > len(source_text):
            raise SourceMapContractError("chunk bounds or ordering are invalid")
        text = raw.get("text")
        if not isinstance(text, str) or source_text[start:end] != text:
            raise SourceMapContractError("chunk text does not match source text")
        decoded[chunk_id] = Chunk(
            doc_id=document_id,
            start_char=start,
            end_char=end,
            text=text,
        )
        previous_end = end
    return decoded
