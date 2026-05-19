import re
import unicodedata

from kogwistar.engine_core.models import (
    Span,
    Document,
    MentionVerification,
)
from kogwistar.typing_interfaces import EngineLike
from typing import Optional, List, Iterable, Callable

from .utils.fuzzy_offsets import (
    FuzzyHit,
    FuzzySpanHit,
    find_best_fuzzy_span,
    fuzzy_find_best_spans,
)


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


import json


class BaseDocValidator:
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
        result = super().validate_span(span=span, doc_id=doc_id, doc=doc, engine=engine)
        if result["correctness"] is True:
            return result

        resolved_doc = _get_doc(doc_id, doc, engine)
        excerpt_from_span = resolved_doc.get_content_by_span(span)

        def _normalize_ocr_text(value: str) -> str:
            text = unicodedata.normalize("NFKC", value or "")
            replacements = {
                "â€“": "–",
                "â€”": "—",
                "â€œ": "“",
                "â€": "”",
                "â€˜": "‘",
                "â€™": "’",
                "â†’": "→",
                "Â ": " ",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        normalized_excerpt = _normalize_ocr_text(span.excerpt)
        normalized_resolved = _normalize_ocr_text(excerpt_from_span)
        if normalized_excerpt == normalized_resolved:
            return {
                **result,
                "correctness": True,
                "excerpt_from_start_end_index": excerpt_from_span,
            }
        if normalized_excerpt and (
            normalized_excerpt in normalized_resolved
            or normalized_resolved in normalized_excerpt
        ):
            return {
                **result,
                "correctness": True,
                "excerpt_from_start_end_index": excerpt_from_span,
            }
        return result
