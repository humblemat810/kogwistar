from dataclasses import dataclass

from graph_knowledge_engine.models import Span, Document, MentionVerification
from graph_knowledge_engine.typing_interfaces import EngineLike
from typing import Type, Optional, List, Iterable, Callable
try:
    # pip install rapidfuzz
    from rapidfuzz import fuzz as rfuzz
except Exception:  # pragma: no cover
    rfuzz = None

import difflib
@dataclass(frozen=True)
class FuzzyHit:
    start: int
    end: int
    score: float

# ---------- exact matching ----------

def find_all_exact(text: str, needle: str) -> List[int]:
    if not needle:
        return []
    out = []
    i = text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


def pick_nearest(starts: Iterable[int], origin: int) -> Optional[int]:
    starts = list(starts)
    if not starts:
        return None
    return min(starts, key=lambda s: (abs(s - origin), s))


# ---------- context refresh ----------

def refresh_context(
    text: str,
    start: int,
    end: int,
    *,
    window_chars: int = 40,
) -> tuple[str, str]:
    return (
        text[max(0, start - window_chars): start].strip(),
        text[end: min(len(text), end + window_chars)].strip(),
    )


def _get_doc(doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        if (doc is not None) and doc_id is not None:
            if doc.id == doc_id:
                pass # ok they agree
            else:
                raise ValueError("Either doc and doc_id specified and they disagree")
        if doc is not None:
            pass            
        else:
            if doc_id is None:
                # unreachable
                raise Exception("Unreacheable")
            else:
                if engine is None:
                    raise ValueError("Engine is requried to resolve doc_id")
                else:
                    doc = engine.get_document(doc_id)
        return doc


# -----------------------------
# Helpers: fuzzy match + nearest
# -----------------------------

def _len_based_threshold(n: int) -> int:
    """
    Sensible thresholds:
    - short strings: require very high similarity
    - long strings: allow a bit more noise
    """
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
    """
    Choose a scorer:
    - For spans (contiguous substrings), partial_ratio is usually best.
    - token_sort_ratio can help if whitespace/token order weirdness exists, but spans are contiguous,
      so we use it only when target has lots of whitespace.
    """
    n = len(target)
    whitespace_heavy = (target.count(" ") >= max(3, n // 10))

    if rfuzz:
        if whitespace_heavy and n <= 80:
            return rfuzz.token_sort_ratio
        return rfuzz.partial_ratio if n >= 20 else rfuzz.ratio

    # Fallback via difflib: return 0..100 similar to rapidfuzz
    def difflib_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    # For fallback, partial-like behavior is approximated by comparing same-length windows, so ratio is fine.
    return difflib_ratio




def fuzzy_find_best_spans(
    haystack: str,
    target: str,
    orig_start: int,
    *,
    max_hits: int = 20,
    scan_band: Optional[int] = None,
) -> List[FuzzyHit]:
    """
    Return up to `max_hits` candidate spans (start,end,score) with score >= threshold,
    preferring hits near orig_start.

    Strategy (fast + simple, no heavy indexing):
    - If scan_band is provided, only scan within [orig_start - band, orig_start + band]
    - Sliding windows around target length +/- deltas
    - Step size scales with target length
    """
    if not target:
        return []

    n = len(target)
    threshold = _len_based_threshold(n)
    scorer = _choose_fuzzy_scorer(target)

    # scanning band: limit work for large docs
    if scan_band is None:
        scan_band = max(2000, n * 50)  # decent default

    lo = max(0, orig_start - scan_band)
    hi = min(len(haystack), orig_start + scan_band)

    region = haystack[lo:hi]
    region_offset = lo

    # Candidate window lengths: allow small drift (OCR / whitespace / punctuation)
    deltas = [0]
    if n >= 20:
        deltas += [max(1, n // 20), -max(1, n // 20)]          # +/- 5%
    if n >= 60:
        deltas += [max(2, n // 10), -max(2, n // 10)]          # +/- 10%

    # Step size: tradeoff accuracy vs speed
    step = 1 if n <= 40 else max(2, n // 25)  # ~4% of length for longer targets

    hits: List[FuzzyHit] = []
    for delta in deltas:
        win = n + delta
        if win <= 0:
            continue

        # Slide across region
        for i in range(0, max(0, len(region) - win + 1), step):
            chunk = region[i:i + win]
            score = float(scorer(chunk, target))
            if score >= threshold:
                start = region_offset + i
                end = start + win
                hits.append(FuzzyHit(start=start, end=end, score=score))

    if not hits:
        return []

    # Sort:
    # 1) higher score
    # 2) nearer to orig_start
    # 3) shorter span if tie (prefer minimal drift)
    hits.sort(key=lambda h: (-h.score, abs(h.start - orig_start), (h.end - h.start)))

    # Deduplicate near-identical hits (same start) keeping best score
    dedup: List[FuzzyHit] = []
    seen_starts = set()
    for h in hits:
        if h.start in seen_starts:
            continue
        seen_starts.add(h.start)
        dedup.append(h)
        if len(dedup) >= max_hits:
            break
    return dedup
    
import json



class BaseDocValidator:
    def fix_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None, nodes_edges = None, source_map = None):
        # must coerce plain text into Document for processing
        # TO-DO fix logic start
        # 1) Validate existing coordinates quickly
        doc= _get_doc(doc_id, doc, engine)
        text = doc.content
        origin = max(0, span.start_char)
        excerpt = span.excerpt or ""
        # --- preserve the LLM-provided evidence for scoring + audit ---
        orig_start = max(0, int(span.start_char))
        orig_excerpt = span.excerpt or ""
        orig_cb = span.context_before or ""
        orig_ca = span.context_after or ""
        # ------------------
        # 1) EXACT MATCH
        # ------------------
        exact_starts = find_all_exact(text, excerpt)
        start = pick_nearest(exact_starts, origin)

        if start is not None:
            end = start + len(excerpt)
            before, after = refresh_context(text, start, end)
            
            span = span.model_copy(update={
                "start_page": 1,
                "end_page": 1,
                "start_char": start,
                "end_char": end,
                "excerpt": excerpt,
                "context_before": before,
                "context_after": after,
                "verification": MentionVerification(
                    method="regex",
                    is_verified=True,
                    score=1.0,
                    notes = json.dumps(
                        {
                            "reason": "fuzzy_repair",
                            "orig_start": orig_start,
                            "fixed_start": start,
                            "fuzzy_score": 1.0,
                            "orig_excerpt": orig_excerpt,
                            "orig_context_before": orig_cb,
                            "orig_context_after": orig_ca,
                        },
                        ensure_ascii=False,
                    )
                ),
            })
            return self.validate_span(span, doc_id, doc, engine)

        # ------------------
        # 2) FUZZY MATCH
        # ------------------
        hits = fuzzy_find_best_spans(text, excerpt, origin)
        if hits:
            best = hits[0]
            fixed_excerpt = text[best.start:best.end]
            before, after = refresh_context(text, best.start, best.end)

            span = span.model_copy(update={
                "start_page": 1,
                "end_page": 1,
                "start_char": best.start,
                "end_char": best.end,
                "excerpt": fixed_excerpt,
                "context_before": before,
                "context_after": after,
                "verification": MentionVerification(
                    method="levenshtein",
                    is_verified=True,
                    score=best.score / 100.0,
                    notes=json.dumps(
                        {
                            "reason": "fuzzy_repair",
                            "orig_start": orig_start,
                            "fixed_start": best.start,
                            "fuzzy_score": round(best.score / 100.0, 4),
                            "orig_excerpt": orig_excerpt,
                            "orig_context_before": orig_cb,
                            "orig_context_after": orig_ca,
                        },
                        ensure_ascii=False,
                    )
                ),
            })
            return self.validate_span(span, doc_id, doc, engine)

        # ------------------
        # 3) FAILED TO FIX
        # ------------------
        span = span.model_copy(update={
            "verification": MentionVerification(
                method="heuristic",
                is_verified=False,
                score=None,
                notes=json.dumps(dict(orig_start=orig_start, orig_excerpt=orig_excerpt)) # f"orig_start={orig_start}, orig_excerpt={(orig_excerpt)!r}"
            )
        })
        
        # TO-DO fix logic end
        return self.validate_span(span, doc_id, doc, engine)
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):

        if not doc:
            raise RuntimeError("fail to resolve document")
        excerpt_from_span = doc.get_content_by_span(span)
        return {"correctness": excerpt_from_span == span.excerpt, "excerpt_from_start_end_index": excerpt_from_span, "except_llm_copied": span.excerpt}
            
class PlainTextDocSpanValidator(BaseDocValidator):
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        return super().validate_span(span=span, doc_id = doc_id, doc = doc, engine = engine)
        if (doc is not None) and doc_id is not None:
            raise ValueError("Either doc or doc_id can be non None")
        if doc is not None:
            pass
        else:
            if doc_id is None:
                # unreachable
                pass
            else:
                if engine is None:
                    raise ValueError("Engine is requried to resolve doc_id")
                else:
                    doc = engine.get_document(doc_id)
        if not doc:
            raise RuntimeError("fail to resolve document")
        
        pass
        
    
    pass
class ChunkedDocValidator:
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        raise NotImplementedError

class OcrDocSpanValidator(BaseDocValidator):
    def validate_span(self, span: Span, doc_id: str | None = None, doc: Document | None = None, engine: EngineLike | None = None):
        raise NotImplementedError
        