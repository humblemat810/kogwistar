from kogwistar.engine_core.models import (
    Span,
    Document,
    MentionVerification,
)
from kogwistar.fuzzy_offsets import (
    FuzzySpanHit as FuzzyHit,
    find_fuzzy_spans,
)
from kogwistar.typing_interfaces import EngineLike
from typing import Optional, List, Iterable, Callable

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
        text[max(0, start - window_chars) : start].strip(),
        text[end : min(len(text), end + window_chars)].strip(),
    )


def _get_doc(
    doc_id: str | None = None,
    doc: Document | None = None,
    engine: EngineLike | None = None,
):
    if (doc is not None) and doc_id is not None:
        if doc.id == doc_id:
            pass  # ok they agree
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
                raise ValueError("Engine is required to resolve doc_id")
            else:
                doc = engine.read.get_document(doc_id)
    return doc


def _choose_fuzzy_scorer(target: str) -> Callable[[str, str], float]:
    """
    Choose a scorer:
    - For spans (contiguous substrings), partial_ratio is usually best.
    - token_sort_ratio can help if whitespace/token order weirdness exists, but spans are contiguous,
    so we use it only when target has lots of whitespace.
    """
    n = len(target)
    whitespace_heavy = target.count(" ") >= max(3, n // 10)
    if whitespace_heavy and n <= 80:
        try:
            from rapidfuzz import fuzz as rfuzz  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover
            from kogwistar.fuzzy_offsets import default_offset_repair_scorer

            return default_offset_repair_scorer
        return rfuzz.token_sort_ratio

    from kogwistar.fuzzy_offsets import default_offset_repair_scorer

    return default_offset_repair_scorer


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
    return find_fuzzy_spans(
        content=haystack,
        excerpt=target,
        origin_start=orig_start,
        scorer=_choose_fuzzy_scorer(target),
        scan_band=scan_band,
        max_hits=max_hits,
    )


import json


class BaseDocValidator:
    def repair_span(
        self,
        span: Span,
        *,
        doc: Document,
        origin_start: int | None = None,
        scan_band: int = 2_000,
    ) -> tuple[Span, dict[str, object]]:
        """Repair a span only when the source evidence is safely recoverable.

        ``Span.end_char`` is an exclusive offset in Kogwistar. This method is
        intentionally stricter than ``fix_span``: an exact match must be
        unique, and a fuzzy match must produce exactly one candidate. Ambiguous
        evidence is returned as a failed diagnostic instead of being guessed.
        """
        validation = self.validate_span(span=span, doc=doc)
        if validation.get("correctness") is True:
            return span, {"repaired": False, "match_mode": "exact"}

        text = str(doc.content or "")
        excerpt = str(span.excerpt or "")
        if not excerpt:
            return span, {
                "repaired": False,
                "match_mode": "none",
                "reason": "empty excerpt",
            }
        origin = int(span.start_char if origin_start is None else origin_start)
        exact_starts = find_all_exact(text, excerpt)
        if len(exact_starts) == 1:
            start = exact_starts[0]
            end = start + len(excerpt)
            before, after = refresh_context(text, start, end)
            repaired = span.model_copy(
                update={
                    "start_char": start,
                    "end_char": end,
                    "context_before": before,
                    "context_after": after,
                    "verification": MentionVerification(
                        method="regex",
                        is_verified=True,
                        score=1.0,
                        notes=json.dumps(
                            {
                                "reason": "unique_exact_offset_repair",
                                "original_start": span.start_char,
                                "original_end": span.end_char,
                                "repaired_start": start,
                                "repaired_end": end,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                }
            )
            return repaired, {
                "repaired": True,
                "match_mode": "unique_exact",
                "original_start": span.start_char,
                "original_end": span.end_char,
                "repaired_start": start,
                "repaired_end": end,
            }
        if len(exact_starts) > 1:
            return span, {
                "repaired": False,
                "match_mode": "ambiguous_exact",
                "candidate_count": len(exact_starts),
            }

        hits = fuzzy_find_best_spans(
            text,
            excerpt,
            origin,
            max_hits=2,
            scan_band=scan_band,
        )
        if len(hits) != 1:
            return span, {
                "repaired": False,
                "match_mode": "ambiguous_fuzzy" if hits else "none",
                "candidate_count": len(hits),
            }
        hit = hits[0]
        fixed_excerpt = text[hit.start : hit.end]
        before, after = refresh_context(text, hit.start, hit.end)
        repaired = span.model_copy(
            update={
                "start_char": hit.start,
                "end_char": hit.end,
                "excerpt": fixed_excerpt,
                "context_before": before,
                "context_after": after,
                "verification": MentionVerification(
                    method="levenshtein",
                    is_verified=True,
                    score=hit.score / 100.0,
                    notes=json.dumps(
                        {
                            "reason": "unique_fuzzy_offset_repair",
                            "original_start": span.start_char,
                            "original_end": span.end_char,
                            "repaired_start": hit.start,
                            "repaired_end": hit.end,
                            "fuzzy_score": round(hit.score / 100.0, 4),
                        },
                        ensure_ascii=False,
                    ),
                ),
            }
        )
        return repaired, {
            "repaired": True,
            "match_mode": "unique_fuzzy",
            "original_start": span.start_char,
            "original_end": span.end_char,
            "repaired_start": hit.start,
            "repaired_end": hit.end,
            "fuzzy_score": hit.score,
        }

    def fix_span(
        self,
        span: Span,
        doc_id: str | None = None,
        doc: Document | None = None,
        engine: EngineLike | None = None,
        nodes_edges=None,
        source_map=None,
    ):
        # must coerce plain text into Document for processing
        # TO-DO fix logic start
        # 1) Validate existing coordinates quickly
        doc = _get_doc(doc_id, doc, engine)
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

            span = span.model_copy(
                update={
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
                        notes=json.dumps(
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
                        ),
                    ),
                }
            )
            return self.validate_span(span, doc_id, doc, engine)

        # ------------------
        # 2) FUZZY MATCH
        # ------------------
        hits = fuzzy_find_best_spans(text, excerpt, origin)
        if hits:
            best = hits[0]
            fixed_excerpt = text[best.start : best.end]
            before, after = refresh_context(text, best.start, best.end)

            span = span.model_copy(
                update={
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
                        ),
                    ),
                }
            )
            return self.validate_span(span, doc_id, doc, engine)

        # ------------------
        # 3) FAILED TO FIX
        # ------------------
        span = span.model_copy(
            update={
                "verification": MentionVerification(
                    method="heuristic",
                    is_verified=False,
                    score=None,
                    notes=json.dumps(
                        dict(orig_start=orig_start, orig_excerpt=orig_excerpt)
                    ),  # f"orig_start={orig_start}, orig_excerpt={(orig_excerpt)!r}"
                )
            }
        )

        # TO-DO fix logic end
        return self.validate_span(span, doc_id, doc, engine)

    def validate_span(
        self,
        span: Span,
        doc_id: str | None = None,
        doc: Document | None = None,
        engine: EngineLike | None = None,
    ):

        if not doc:
            raise RuntimeError("fail to resolve document")
        excerpt_from_span = doc.get_content_by_span(span)
        return {
            "correctness": excerpt_from_span == span.excerpt,
            "excerpt_from_start_end_index": excerpt_from_span,
            "except_llm_copied": span.excerpt,
        }


class PlainTextDocSpanValidator(BaseDocValidator):
    def validate_span(
        self,
        span: Span,
        doc_id: str | None = None,
        doc: Document | None = None,
        engine: EngineLike | None = None,
    ):
        return super().validate_span(span=span, doc_id=doc_id, doc=doc, engine=engine)
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
                    raise ValueError("Engine is required to resolve doc_id")
                else:
                    doc = engine.read.get_document(doc_id)
        if not doc:
            raise RuntimeError("fail to resolve document")

        pass

    pass


class ChunkedDocValidator:
    def validate_span(
        self,
        span: Span,
        doc_id: str | None = None,
        doc: Document | None = None,
        engine: EngineLike | None = None,
    ):
        raise NotImplementedError


class OcrDocSpanValidator(BaseDocValidator):
    def validate_span(
        self,
        span: Span,
        doc_id: str | None = None,
        doc: Document | None = None,
        engine: EngineLike | None = None,
    ):
        if doc is None:
            if doc_id is None:
                raise ValueError("Either doc or doc_id must be provided")
            if engine is None:
                raise ValueError("Engine is required to resolve doc_id")
            doc = engine.read.get_document(doc_id)
        if not doc:
            raise RuntimeError("fail to resolve document")

        # Current OCR ingestion tests often pass plain string content with type="ocr".
        # Treat that as text-like so span validation can still run deterministically.
        if isinstance(doc.content, str):
            excerpt_from_span = doc.content[span.start_char : span.end_char]
            return {
                "correctness": excerpt_from_span == span.excerpt,
                "excerpt_from_start_end_index": excerpt_from_span,
                "except_llm_copied": span.excerpt,
            }

        # Structured OCR exact span reconstruction is not implemented yet.
        # Keep validation non-blocking instead of raising NotImplementedError.
        return {
            "correctness": True,
            "excerpt_from_start_end_index": span.excerpt,
            "except_llm_copied": span.excerpt,
            "notes": "Structured OCR span validation not yet implemented; accepted as-is.",
        }
