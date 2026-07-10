from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal


SourcePointerEndMode = Literal["exclusive", "inclusive"]


class SourcePointerValidationError(ValueError):
    """Raised when an LLM/source-map pointer is not safe to trust."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ValidatedSourcePointer:
    source_cluster_id: str | None
    doc_id: str | None
    start_char: int
    end_char: int
    end_mode: SourcePointerEndMode
    text: str | None = None

    @property
    def slice_end_char(self) -> int:
        return self.end_char + 1 if self.end_mode == "inclusive" else self.end_char


def source_pointer_has_character_span(pointer: object) -> bool:
    """Return whether a provenance pointer is trying to be a character span."""

    return any(
        _pointer_field(pointer, key) is not None
        for key in ("start_char", "end_char", "excerpt", "verbatim_text", "context_before", "context_after")
    )


def validate_source_pointer(
    pointer: object,
    *,
    source_text_by_cluster: Mapping[str, str] | None = None,
    parent_pointers: Iterable[object] = (),
    end_mode: SourcePointerEndMode = "exclusive",
    require_source_cluster: bool = False,
    require_source_text: bool = False,
    require_parent_containment: bool = False,
    require_text_match: bool = False,
) -> ValidatedSourcePointer:
    """Validate a source pointer before treating it as locatable evidence.

    Kogwistar graph spans use exclusive ``end_char``. Some parser-facing
    source maps use inclusive pointer ends, so callers choose the mode instead
    of open-coding subtly different checks at each LLM seam.
    """

    source_cluster_id = _optional_str(_pointer_field(pointer, "source_cluster_id"))
    doc_id = _optional_str(_pointer_field(pointer, "doc_id") or _pointer_field(pointer, "source_document_id"))
    if require_source_cluster and not source_cluster_id:
        raise SourcePointerValidationError("missing_source_cluster", "source_cluster_id is required")

    start_char = _required_int(pointer, "start_char")
    end_char = _required_int(pointer, "end_char")
    if start_char < 0:
        raise SourcePointerValidationError("invalid_span", "start_char must be >= 0")
    if end_mode == "inclusive":
        if end_char < start_char:
            raise SourcePointerValidationError("invalid_span", "inclusive end_char must be >= start_char")
    elif end_mode == "exclusive":
        if end_char <= start_char:
            raise SourcePointerValidationError("invalid_span", "exclusive end_char must be > start_char")
    else:
        raise SourcePointerValidationError("invalid_end_mode", f"unsupported end mode: {end_mode}")

    source_text: str | None = None
    if source_cluster_id and source_text_by_cluster is not None and source_cluster_id in source_text_by_cluster:
        source_text = source_text_by_cluster[source_cluster_id]
        if _slice_end(end_char, end_mode) > len(source_text):
            raise SourcePointerValidationError("out_of_bounds", "pointer span is outside the source text")
    elif require_source_text:
        raise SourcePointerValidationError("source_not_found", "source text was not found for source_cluster_id")

    if require_parent_containment:
        if not _is_contained_by_parent(
            source_cluster_id=source_cluster_id,
            start_char=start_char,
            end_char=end_char,
            end_mode=end_mode,
            parent_pointers=parent_pointers,
        ):
            raise SourcePointerValidationError("outside_parent_span", "pointer span is outside the current parent span")

    text_value = _pointer_field(pointer, "verbatim_text")
    if text_value is None:
        text_value = _pointer_field(pointer, "excerpt")
    text = str(text_value) if text_value is not None else None
    if require_text_match:
        if source_text is None:
            raise SourcePointerValidationError("source_not_found", "source text is required to validate pointer text")
        if text is None:
            raise SourcePointerValidationError("missing_text", "verbatim_text or excerpt is required")
        expected = source_text[start_char : _slice_end(end_char, end_mode)]
        if text != expected:
            raise SourcePointerValidationError("text_mismatch", "pointer text does not match the source span")

    return ValidatedSourcePointer(
        source_cluster_id=source_cluster_id,
        doc_id=doc_id,
        start_char=start_char,
        end_char=end_char,
        end_mode=end_mode,
        text=text,
    )


def _pointer_field(pointer: object, key: str) -> object | None:
    if isinstance(pointer, Mapping):
        return pointer.get(key)
    return getattr(pointer, key, None)


def _optional_str(value: object | None) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _required_int(pointer: object, key: str) -> int:
    value = _pointer_field(pointer, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourcePointerValidationError("span_not_int", f"{key} must be an integer")
    return value


def _slice_end(end_char: int, end_mode: SourcePointerEndMode) -> int:
    return end_char + 1 if end_mode == "inclusive" else end_char


def _is_contained_by_parent(
    *,
    source_cluster_id: str | None,
    start_char: int,
    end_char: int,
    end_mode: SourcePointerEndMode,
    parent_pointers: Iterable[object],
) -> bool:
    slice_end = _slice_end(end_char, end_mode)
    for parent_pointer in parent_pointers:
        parent_source_cluster_id = _optional_str(_pointer_field(parent_pointer, "source_cluster_id"))
        if source_cluster_id and parent_source_cluster_id != source_cluster_id:
            continue
        try:
            parent_start = _required_int(parent_pointer, "start_char")
            parent_end = _required_int(parent_pointer, "end_char")
        except SourcePointerValidationError:
            continue
        if parent_start <= start_char and slice_end <= _slice_end(parent_end, end_mode):
            return True
    return False
