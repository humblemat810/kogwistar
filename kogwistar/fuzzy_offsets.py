from __future__ import annotations

from dataclasses import dataclass
import difflib
from typing import Callable

try:  # pragma: no cover - optional dependency
    from rapidfuzz import fuzz as _rapidfuzz
except Exception:  # pragma: no cover
    _rapidfuzz = None


@dataclass(frozen=True, slots=True)
class FuzzySpanHit:
    start: int
    end: int
    score: float


def offset_repair_threshold(excerpt_len: int) -> float:
    if excerpt_len <= 8:
        return 95.0
    if excerpt_len <= 20:
        return 92.0
    if excerpt_len <= 60:
        return 88.0
    if excerpt_len <= 120:
        return 85.0
    return 82.0


def default_offset_repair_scorer(candidate: str, excerpt: str) -> float:
    if not excerpt:
        return 0.0
    if _rapidfuzz is not None:
        return float(_rapidfuzz.partial_ratio(candidate, excerpt))
    return float(difflib.SequenceMatcher(None, candidate, excerpt).ratio() * 100.0)


def _coerce_offset_score(raw_score: object) -> float:
    if not isinstance(raw_score, (int, float)):
        return 0.0
    score = float(raw_score)
    if not score == score or score in (float("inf"), float("-inf")):  # NaN / inf guard
        return 0.0
    if 0.0 <= score <= 1.0:
        score *= 100.0
    return max(0.0, score)


def _scan_fuzzy_spans(
    *,
    content: str,
    excerpt: str,
    origin_start: int,
    scorer: Callable[[str, str], float],
    scan_band: int | None = None,
    candidate_filter: Callable[[str], bool] | None = None,
    max_hits: int | None = None,
) -> list[FuzzySpanHit]:
    if not excerpt:
        return []

    excerpt_len = len(excerpt)
    if excerpt_len == 0:
        return []

    threshold = offset_repair_threshold(excerpt_len)
    if scan_band is None:
        scan_band = max(2000, excerpt_len * 50)

    lo = max(0, origin_start - scan_band)
    hi = min(len(content), origin_start + scan_band)
    region = content[lo:hi]
    if not region:
        return []

    deltas = [0]
    if excerpt_len >= 20:
        delta_5 = max(1, excerpt_len // 20)
        deltas.extend([delta_5, -delta_5])
    if excerpt_len >= 60:
        delta_10 = max(2, excerpt_len // 10)
        deltas.extend([delta_10, -delta_10])

    step = 1 if excerpt_len <= 40 else max(2, excerpt_len // 25)

    hits: list[FuzzySpanHit] = []
    for delta in deltas:
        width = excerpt_len + delta
        if width <= 0 or width > len(region):
            continue
        max_i = len(region) - width
        for i in range(0, max_i + 1, step):
            candidate = region[i : i + width]
            if candidate_filter is not None and not candidate_filter(candidate):
                continue
            score = _coerce_offset_score(scorer(candidate, excerpt))
            if score < threshold:
                continue
            hits.append(FuzzySpanHit(start=lo + i, end=lo + i + width, score=score))

    if not hits:
        return []

    hits.sort(key=lambda hit: (-hit.score, abs(hit.start - origin_start), hit.end - hit.start))

    deduped: list[FuzzySpanHit] = []
    seen_starts: set[int] = set()
    for hit in hits:
        if hit.start in seen_starts:
            continue
        seen_starts.add(hit.start)
        deduped.append(hit)
        if max_hits is not None and len(deduped) >= max_hits:
            break
    return deduped


def find_fuzzy_spans(
    *,
    content: str,
    excerpt: str,
    origin_start: int,
    scorer: Callable[[str, str], float] | None = None,
    scan_band: int | None = None,
    candidate_filter: Callable[[str], bool] | None = None,
    max_hits: int = 20,
) -> list[FuzzySpanHit]:
    return _scan_fuzzy_spans(
        content=content,
        excerpt=excerpt,
        origin_start=origin_start,
        scorer=scorer or default_offset_repair_scorer,
        scan_band=scan_band,
        candidate_filter=candidate_filter,
        max_hits=max_hits,
    )


def find_best_fuzzy_span(
    *,
    content: str,
    excerpt: str,
    origin_start: int,
    scorer: Callable[[str, str], float] | None = None,
    scan_band: int | None = None,
    candidate_filter: Callable[[str], bool] | None = None,
) -> FuzzySpanHit | None:
    hits = find_fuzzy_spans(
        content=content,
        excerpt=excerpt,
        origin_start=origin_start,
        scorer=scorer,
        scan_band=scan_band,
        candidate_filter=candidate_filter,
        max_hits=1,
    )
    return hits[0] if hits else None
