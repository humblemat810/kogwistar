# strategies/verifiers.py
from __future__ import annotations
from typing import Dict, Optional
from ..models import ReferenceSession, MentionVerification

def ensemble_default(engine, extracted_text: str, full_text: str, ref: ReferenceSession,
                     *, min_ngram: int = 5,
                     weights: Dict[str, float] = {"rapidfuzz":0.5,"coverage":0.3,"embedding":0.2},
                     threshold: float = 0.70) -> ReferenceSession:
    """Wraps the engine’s built-in multi-signal verifier."""
    return engine._verify_one_reference(
        extracted_text, full_text, ref,
        min_ngram=min_ngram, weights=weights, threshold=threshold
    )

def coverage_only(engine, extracted_text: str, full_text: str, ref: ReferenceSession,
                  *, min_ngram: int = 7, threshold: float = 0.50) -> ReferenceSession:
    """Fast, no-ML fallback: only n-gram coverage; marks verified if above threshold."""
    # This reuses engine’s coverage function and writes a MentionVerification.
    cv = engine._score_coverage(extracted_text, engine._slice_span(full_text, ref.start_char or 0, ref.end_char or 0), min_ngram=min_ngram) or 0.0
    out = ref.model_copy(deep=True)
    out.verification = MentionVerification(method="coverage", is_verified=(cv >= threshold), score=cv, notes=f"min_ngram={min_ngram}")
    return out

def strict_with_min_span(engine, extracted_text: str, full_text: str, ref: ReferenceSession,
                         *, min_span_chars: int = 24, min_score: float = 0.8) -> ReferenceSession:
    """Require a minimal span and high ensemble score."""
    if (ref.end_char or 0) - (ref.start_char or 0) < min_span_chars:
        out = ref.model_copy(deep=True)
        out.verification = MentionVerification(method="strict", is_verified=False, score=0.0, notes="span_too_short")
        return out
    return ensemble_default(engine, extracted_text, full_text, ref, min_ngram=6, weights={"rapidfuzz":0.4,"coverage":0.4,"embedding":0.2}, threshold=min_score)