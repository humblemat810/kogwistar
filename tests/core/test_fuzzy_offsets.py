from __future__ import annotations

from difflib import SequenceMatcher

import pytest

import kogwistar.fuzzy_offsets as fuzzy_offsets
from kogwistar.fuzzy_offsets import (
    default_offset_repair_scorer,
    find_best_fuzzy_span,
    find_fuzzy_spans,
    offset_repair_threshold,
)

pytestmark = pytest.mark.ci


def test_offset_repair_threshold_schedule_is_stable() -> None:
    assert offset_repair_threshold(1) == 95.0
    assert offset_repair_threshold(8) == 95.0
    assert offset_repair_threshold(9) == 92.0
    assert offset_repair_threshold(20) == 92.0
    assert offset_repair_threshold(21) == 88.0
    assert offset_repair_threshold(60) == 88.0
    assert offset_repair_threshold(61) == 85.0
    assert offset_repair_threshold(120) == 85.0
    assert offset_repair_threshold(121) == 82.0


def test_default_offset_repair_scorer_uses_fast_path_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRapidFuzz:
        @staticmethod
        def partial_ratio(candidate: str, excerpt: str) -> float:
            _ = candidate, excerpt
            return 77.0

    monkeypatch.setattr(fuzzy_offsets, "_rapidfuzz", _FakeRapidFuzz())
    assert default_offset_repair_scorer("alpha", "beta") == 77.0

    monkeypatch.setattr(fuzzy_offsets, "_rapidfuzz", None)
    expected = SequenceMatcher(None, "Proof step one", "Proof-step one").ratio() * 100.0
    assert default_offset_repair_scorer("Proof step one", "Proof-step one") == pytest.approx(expected)


def test_find_fuzzy_spans_prefers_nearer_origin() -> None:
    first = "Proof step one"
    content = f"xxx {first} yyy {first} zzz"
    origin_start = content.rfind(first) - 1

    hits = find_fuzzy_spans(
        content=content,
        excerpt=first,
        origin_start=origin_start,
        max_hits=2,
    )

    assert len(hits) >= 2
    assert hits[0].start == content.rfind(first)
    assert hits[1].start == content.find(first)


def test_find_best_fuzzy_span_candidate_filter_can_reject_match() -> None:
    content = "alpha beta gamma"
    hit = find_best_fuzzy_span(
        content=content,
        excerpt=content,
        origin_start=0,
        candidate_filter=lambda candidate: candidate != content,
    )

    assert hit is None


def test_find_best_fuzzy_span_empty_excerpt_returns_none() -> None:
    assert find_best_fuzzy_span(content="abc", excerpt="", origin_start=0) is None
