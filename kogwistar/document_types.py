"""Small shared document value types used by splitting and graph models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A source-preserving text chunk with document-relative character offsets."""

    doc_id: str
    start_char: int
    end_char: int
    text: str
