from __future__ import annotations

from .fuzzy_offsets import FuzzySpanHit, find_best_fuzzy_span, find_fuzzy_spans
from .source_pointers import (
    SourcePointerValidationError,
    ValidatedSourcePointer,
    source_pointer_has_character_span,
    validate_source_pointer,
)

__all__ = [
    "FuzzySpanHit",
    "SourcePointerValidationError",
    "ValidatedSourcePointer",
    "find_best_fuzzy_span",
    "find_fuzzy_spans",
    "source_pointer_has_character_span",
    "validate_source_pointer",
]
