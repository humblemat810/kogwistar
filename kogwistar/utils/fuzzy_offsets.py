from __future__ import annotations

from dataclasses import dataclass
import difflib
from typing import Callable, List, Optional

try:
    from rapidfuzz import fuzz as rfuzz
except Exception:  # pragma: no cover
    rfuzz = None


@dataclass(frozen=True, slots=True)
class FuzzyHit:
    start: int
    end: int
    score: float


FuzzySpanHit = FuzzyHit


def _len_based_threshold(n: int) -> int:
    if n <= 8:
        return 95
    if n <= 20:
        return 92
    if n <= 60:
        return 88
    if n <= 120:
        return 85
    return 82


def _choose_fuzzy_scorer(target: str) -> Callable[[str, str], float]:
    whitespace_heavy = target.count(" ") >= max(3, len(target) // 10)

    if rfuzz is not None:
        if whitespace_heavy and len(target) <= 80:
            return rfuzz.token_sort_ratio
        return rfuzz.partial_ratio if len(target) >= 20 else rfuzz.ratio

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    return _ratio


def _coerce_score(raw_score: object) -> float:
    if not isinstance(raw_score, (int, float)):
        return 0.0
    score = float(raw_score)
    if not (score == score):  # NaN-safe check
        return 0.0
    if 0.0 <= score <= 1.0:
        score *= 100.0
    return max(0.0, score)


def _pick_best_hit(
    hits: list[FuzzySpanHit],
    *,
    origin_start: int,
) -> FuzzySpanHit | None:
    if not hits:
        return None
    hits.sort(key=lambda h: (-h.score, abs(h.start - origin_start), (h.end - h.start)))
    seen_starts: set[int] = set()
    for hit in hits:
        if hit.start in seen_starts:
            continue
        seen_starts.add(hit.start)
        return hit
    return None


def fuzzy_find_best_spans(
    haystack: str,
    target: str,
    orig_start: int,
    *,
    max_hits: int = 20,
    scan_band: Optional[int] = None,
    candidate_filter: Callable[[str], bool] | None = None,
    scorer: Callable[[str, str], float] | None = None,
) -> List[FuzzySpanHit]:
    if not target:
        return []

    n = len(target)
    threshold = _len_based_threshold(n)
    scorer_fn = scorer or _choose_fuzzy_scorer(target)

    if scan_band is None:
        scan_band = max(2000, n * 50)

    lo = max(0, orig_start - scan_band)
    hi = min(len(haystack), orig_start + scan_band)

    region = haystack[lo:hi]
    region_offset = lo

    deltas = [0]
    if n >= 20:
        deltas += [max(1, n // 20), -max(1, n // 20)]
    if n >= 60:
        deltas += [max(2, n // 10), -max(2, n // 10)]

    step = 1 if n <= 40 else max(2, n // 25)

    hits: List[FuzzySpanHit] = []
    for delta in deltas:
        win = n + delta
        if win <= 0:
            continue

        for i in range(0, max(0, len(region) - win + 1), step):
            chunk = region[i : i + win]
            if candidate_filter is not None and not candidate_filter(chunk):
                continue
            score = _coerce_score(scorer_fn(chunk, target))
            if score >= threshold:
                start = region_offset + i
                end = start + win
                hits.append(FuzzySpanHit(start=start, end=end, score=score))

    if not hits:
        return []

    hits.sort(key=lambda h: (-h.score, abs(h.start - orig_start), (h.end - h.start)))

    dedup: List[FuzzySpanHit] = []
    seen_starts = set()
    for hit in hits:
        if hit.start in seen_starts:
            continue
        seen_starts.add(hit.start)
        dedup.append(hit)
        if len(dedup) >= max_hits:
            break
    return dedup


def find_best_fuzzy_span(
    *,
    content: str,
    excerpt: str,
    origin_start: int,
    scan_band: int | None = None,
    candidate_filter: Callable[[str], bool] | None = None,
    scorer: Callable[[str, str], float] | None = None,
) -> FuzzySpanHit | None:
    hits = fuzzy_find_best_spans(
        content,
        excerpt,
        origin_start,
        max_hits=1,
        scan_band=scan_band,
        candidate_filter=candidate_filter,
        scorer=scorer,
    )
    return hits[0] if hits else None

