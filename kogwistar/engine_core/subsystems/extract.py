from __future__ import annotations

from copy import deepcopy
import difflib
import importlib
import json
import math
from typing import Any, List, Literal, Type, cast

from ...id_provider import stable_id
from ...extraction import BaseDocValidator
from ...llm_tasks import ExtractGraphTaskRequest
from pydantic import BaseModel
from ..models import (
    AssocFlattenedLLMGraphExtraction,
    Document,
    Edge,
    Grounding,
    LLMGraphExtraction,
    Node,
    Span,
)
from ..types import (
    ExtractionSchemaMode,
    OffsetMismatchPolicy,
    OffsetRepairScorer,
    ResolvedExtractionSchemaMode,
)
from ..utils.aliasing import AliasBook, base62_to_uuid, uuid_to_base62
from .base import NamespaceProxy
from ...typing_interfaces import ExtractLike

# Optional: RapidFuzz
try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


class ExtractSubsystem(NamespaceProxy, ExtractLike):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def _engine_const(self, name: str, default: Any) -> Any:
        mod = importlib.import_module(self._e.__class__.__module__)
        return getattr(mod, name, default)

    def _doc_alias(self) -> str:
        return str(self._engine_const("_DOC_ALIAS", "::DOC::"))

    def _id_strategy(self) -> str:
        return str(self._engine_const("ID_STRATEGY", "session_alias"))

    def resolve_extraction_schema_mode(
        self,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
    ) -> ResolvedExtractionSchemaMode:
        requested = extraction_schema_mode or self._e.extraction_schema_mode
        allowed: set[str] = {"auto", "full", "lean", "flattened_lean", "flattened_full"}
        if requested not in allowed:
            raise ValueError(
                f"Unsupported extraction_schema_mode={requested!r}. "
                f"Expected one of {sorted(allowed)}"
            )
        if requested == "auto":
            provider = getattr(
                self._e.llm_tasks.provider_hints, "extract_graph_provider", "unknown"
            )
            if provider == "gemini":
                return "lean"
            return "full"
        return cast(ResolvedExtractionSchemaMode, requested)

    def schema_prompt_rules(self, mode: ResolvedExtractionSchemaMode) -> str:
        common = (
            "Rules:\n"
            "1) When referring to existing items, use ONLY the given aliases.\n"
            "2) If creating new items, omit their id.\n"
            "3) Do not invent aliases; use the provided ones only.\n"
            "4) Do NOT invent real UUIDs.\n"
        )
        if mode in {"lean", "flattened_lean"}:
            return (
                common
                + "5) Each node/edge MUST include at least one grounding span.\n"
                + "6) Every span MUST include page_number, start_char, end_char, and excerpt.\n"
                + "7) excerpt MUST exactly equal document[start_char:end_char].\n"
            )
        doc_alias = self._doc_alias()
        return (
            common
            + "5) Each node/edge MUST include at least one grounding span.\n"
            + "6) Every span MUST include collection_page_url, document_page_url, doc_id, page_number, start_char, end_char, excerpt, context_before, context_after.\n"
            + f"7) Use '{doc_alias}' as doc_id in output spans.\n"
            + f"8) Use document_page_url='document/{doc_alias}' and collection_page_url='document_collection/{doc_alias}'.\n"
            + "9) excerpt MUST exactly equal document[start_char:end_char].\n"
        )

    def structured_schema_for_mode(self, mode: ResolvedExtractionSchemaMode):
        if mode == "full":
            return LLMGraphExtraction["llm"], False
        if mode == "lean":
            return LLMGraphExtraction["llm_in"], False
        if mode == "flattened_lean":
            return AssocFlattenedLLMGraphExtraction["llm_in"], True
        if mode == "flattened_full":
            return AssocFlattenedLLMGraphExtraction["llm"], True
        raise ValueError(f"Unsupported resolved extraction schema mode: {mode!r}")

    @staticmethod
    def _offset_repair_threshold(excerpt_len: int) -> float:
        if excerpt_len <= 8:
            return 95.0
        if excerpt_len <= 20:
            return 92.0
        if excerpt_len <= 60:
            return 88.0
        if excerpt_len <= 120:
            return 85.0
        return 82.0

    @staticmethod
    def _clip_offset_excerpt(text: str, *, max_chars: int = 80) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-(max_chars // 2) :]
        return f"{head}...{tail}"

    @staticmethod
    def _find_all_exact_occurrences(content: str, excerpt: str) -> list[int]:
        if not excerpt:
            return []
        out: list[int] = []
        idx = content.find(excerpt)
        while idx != -1:
            out.append(idx)
            idx = content.find(excerpt, idx + 1)
        return out

    @staticmethod
    def _coerce_offset_score(raw_score: Any) -> float:
        if not isinstance(raw_score, (int, float)):
            return 0.0
        score = float(raw_score)
        if not math.isfinite(score):
            return 0.0
        if 0.0 <= score <= 1.0:
            score = score * 100.0
        return max(0.0, score)

    def default_offset_repair_scorer(self, candidate: str, excerpt: str) -> float:
        if not excerpt:
            return 0.0
        if _HAS_RAPIDFUZZ:
            return float(fuzz.partial_ratio(candidate, excerpt))
        return float(difflib.SequenceMatcher(None, candidate, excerpt).ratio() * 100.0)

    def resolve_offset_repair_scorer(
        self,
        override: OffsetRepairScorer | None,
    ) -> OffsetRepairScorer:
        if override is not None:
            return override
        if self._e.offset_repair_scorer is not None:
            return self._e.offset_repair_scorer
        return self.default_offset_repair_scorer

    def _iter_lean_spans_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        payload: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        rows: list[tuple[str, dict[str, Any]]] = []
        if mode == "flattened_lean":
            for i_sp, span_row in enumerate(payload.get("spans") or []):
                if isinstance(span_row, dict):
                    rows.append((f"spans[{i_sp}]", span_row))
            return rows

        if mode != "lean":
            return rows

        for section in ("nodes", "edges"):
            for i_entry, entry in enumerate(payload.get(section) or []):
                if not isinstance(entry, dict):
                    continue
                mentions = entry.get("mentions")
                if mentions is None:
                    mentions = entry.get("groundings")
                if not isinstance(mentions, list):
                    continue
                for i_g, grounding in enumerate(mentions):
                    if not isinstance(grounding, dict):
                        continue
                    spans = grounding.get("spans")
                    if not isinstance(spans, list):
                        continue
                    for i_sp, span_row in enumerate(spans):
                        if isinstance(span_row, dict):
                            rows.append(
                                (
                                    f"{section}[{i_entry}].mentions[{i_g}].spans[{i_sp}]",
                                    span_row,
                                )
                            )
        return rows

    def _build_offset_failure_detail(
        self,
        *,
        path: str,
        content: str,
        excerpt: str,
        start_char: int,
        end_char: int,
        exact_hits: int,
        best_fuzzy_score: float | None,
    ) -> str:
        hinted_slice = "<out-of-bounds>"
        if 0 <= start_char < end_char <= len(content):
            hinted_slice = content[start_char:end_char]
        fuzzy_text = "n/a" if best_fuzzy_score is None else f"{best_fuzzy_score:.2f}"
        return (
            f"{path}: start={start_char}, end={end_char}, "
            f"excerpt='{self._clip_offset_excerpt(excerpt)}', "
            f"hinted_slice='{self._clip_offset_excerpt(hinted_slice)}', "
            f"exact_hits={exact_hits}, best_fuzzy={fuzzy_text}"
        )

    def _find_best_fuzzy_span(
        self,
        *,
        content: str,
        excerpt: str,
        origin_start: int,
        scorer: OffsetRepairScorer,
    ) -> tuple[int, int, float] | None:
        if not excerpt:
            return None
        excerpt_len = len(excerpt)
        if excerpt_len == 0:
            return None

        scan_band = max(2000, excerpt_len * 50)
        lo = max(0, origin_start - scan_band)
        hi = min(len(content), origin_start + scan_band)
        region = content[lo:hi]
        if not region:
            return None

        deltas = [0]
        if excerpt_len >= 20:
            delta_5 = max(1, excerpt_len // 20)
            deltas.extend([delta_5, -delta_5])
        if excerpt_len >= 60:
            delta_10 = max(2, excerpt_len // 10)
            deltas.extend([delta_10, -delta_10])

        step = 1 if excerpt_len <= 40 else max(2, excerpt_len // 25)
        threshold = self._offset_repair_threshold(excerpt_len)
        best: tuple[int, int, float] | None = None

        for delta in deltas:
            width = excerpt_len + delta
            if width <= 0 or width > len(region):
                continue
            max_i = len(region) - width
            for i in range(0, max_i + 1, step):
                cand = region[i : i + width]
                score = self._coerce_offset_score(scorer(cand, excerpt))
                if score < threshold:
                    continue
                start = lo + i
                end = start + width
                if best is None:
                    best = (start, end, score)
                    continue
                prev_start, prev_end, prev_score = best
                prev_dist = abs(prev_start - origin_start)
                cur_dist = abs(start - origin_start)
                prev_len = prev_end - prev_start
                cur_len = end - start
                if (score > prev_score) or (
                    score == prev_score
                    and (
                        cur_dist < prev_dist
                        or (cur_dist == prev_dist and cur_len < prev_len)
                    )
                ):
                    best = (start, end, score)
        return best

    def to_canonical_extraction_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        parsed: Any,
        content: str,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ) -> LLMGraphExtraction:
        doc_alias = self._doc_alias()
        if mode in {"lean", "flattened_lean"}:
            payload = (
                parsed.model_dump()
                if isinstance(parsed, BaseModel)
                else deepcopy(parsed)
            )
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected parsed payload as dict/BaseModel for mode={mode}, got {type(payload).__name__}"
                )
            parsed = self.repair_lean_offsets_for_mode(
                mode=mode,
                payload=payload,
                content=content,
                policy=offset_mismatch_policy,
                offset_repair_scorer=offset_repair_scorer,
            )
        if mode == "full":
            return LLMGraphExtraction.from_normal_llm(
                parsed,
                insertion_method="llm",
                doc_id=doc_alias,
                content=content,
            )
        if mode == "lean":
            return LLMGraphExtraction.from_llm_in_payload(
                parsed,
                insertion_method="llm",
                doc_id=doc_alias,
                content=content,
            )
        if mode == "flattened_lean":
            return AssocFlattenedLLMGraphExtraction.to_canonical_from_llm_in_payload(
                parsed,
                doc_id=doc_alias,
                content=content,
                insertion_method="llm",
            )
        if mode == "flattened_full":
            assoc_payload = (
                parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
            )
            assoc = AssocFlattenedLLMGraphExtraction.model_validate(
                assoc_payload,
                context={"insertion_method": "llm"},
            )
            return assoc.to_canonical(insertion_method="llm")
        raise ValueError(f"Unsupported resolved extraction schema mode: {mode!r}")

    def extract_graph_with_llm_aliases(
        self,
        content: str,
        alias_nodes_str: str,
        alias_edges_str: str,
        instruction_for_node_edge_contents_parsing_inclusion: None | str = None,
        last_iteration_result: dict | None = None,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        doc_alias = self._doc_alias()
        resolved_mode = self.resolve_extraction_schema_mode(extraction_schema_mode)
        prompt_rules = self.schema_prompt_rules(resolved_mode)
        if instruction_for_node_edge_contents_parsing_inclusion is None:
            instruction_for_node_edge_contents_parsing_inclusion = (
                "Nodes should include at least: Parties, Obligations, Rights, Deliverables, Payment Terms, Termination Conditions, Confidentiality Clauses, Governing Law, Dates, and Penalties.  "
                "Edges should capture: (Party -> Obligation), (Obligation -> Condition), (Party -> Right), (Obligation -> Deliverable), (Clause -> Governing Law).  "
            )
        last_parsed_payload: dict[str, object] | None = None
        last_error: str | None = None
        if last_iteration_result and last_iteration_result.get("error"):
            last_error = str(last_iteration_result.get("error"))
            last_parsed = last_iteration_result.get("parsed")
            if isinstance(last_parsed, BaseModel):
                last_parsed_payload = cast(
                    dict[str, object], last_parsed.model_dump(mode="python")
                )
            elif isinstance(last_parsed, dict):
                last_parsed_payload = cast(dict[str, object], last_parsed)

        result = self._e.llm_tasks.extract_graph(
            ExtractGraphTaskRequest(
                content=content,
                alias_nodes=alias_nodes_str,
                alias_edges=alias_edges_str,
                doc_alias=doc_alias,
                instruction=instruction_for_node_edge_contents_parsing_inclusion,
                prompt_rules=prompt_rules,
                schema_mode=cast(
                    Literal["full", "lean", "flattened_lean", "flattened_full"],
                    resolved_mode,
                ),
                last_parsed=last_parsed_payload,
                last_error=last_error,
            )
        )

        parsed_payload = result.parsed_payload
        parsed_canonical = None
        if parsed_payload is not None:
            parsed_canonical = self.to_canonical_extraction_for_mode(
                mode=resolved_mode,
                parsed=parsed_payload,
                content=content,
                offset_mismatch_policy=offset_mismatch_policy,
                offset_repair_scorer=offset_repair_scorer,
            )
        return result.raw, parsed_canonical, result.parsing_error

    def de_alias_ids_in_result(
        self, doc_id: str, parsed: LLMGraphExtraction
    ) -> LLMGraphExtraction:
        if self._id_strategy() == "base62":

            def r(s: str):
                if not s:
                    raise ValueError("s cannot be None or Falsy")
                if s.startswith("N~"):
                    return base62_to_uuid(s[2:])
                if s.startswith("E~"):
                    return base62_to_uuid(s[2:])
                return s

        else:
            book = self._e._alias_book(doc_id)

            def r(s: str):
                if not s:
                    raise ValueError("s cannot be None or Falsy")
                return book.alias_to_real.get(s, s)

        for n in parsed.nodes:
            if n.id:
                n.id = r(n.id)
        for e in parsed.edges:
            if e.id:
                e.id = r(e.id)
            e.source_ids = [r(x) for x in e.source_ids]
            e.target_ids = [r(x) for x in e.target_ids]
        return parsed

    def aliasify_for_prompt(
        self, doc_id: str, ctx_nodes: list[dict], ctx_edges: list[dict]
    ):
        if self._id_strategy() == "base62":
            aliased_nodes = []
            for n in ctx_nodes:
                aliased_nodes.append(
                    {
                        "id": f"N~{uuid_to_base62(n['id'])}",
                        "label": n["label"],
                        "type": n["type"],
                        "summary": n.get("summary", ""),
                    }
                )
            aliased_edges = []
            for e in ctx_edges:
                aliased_edges.append(
                    {
                        "id": f"E~{uuid_to_base62(e['id'])}",
                        "relation": e["relation"],
                        "source_ids": [
                            f"N~{uuid_to_base62(s)}" for s in e.get("source_ids", [])
                        ],
                        "target_ids": [
                            f"N~{uuid_to_base62(t)}" for t in e.get("target_ids", [])
                        ],
                    }
                )
            return (
                aliased_nodes,
                aliased_edges,
                "Node aliases: (implicit base62)",
                "Edge aliases: (implicit base62)",
            )

        book = self._e._alias_book(doc_id)
        node_ids = [n["id"] for n in ctx_nodes]
        edge_ids = [e["id"] for e in ctx_edges]
        new_nodes, new_edges = book.legend_delta(node_ids, edge_ids)

        def a_node(x):
            return book.real_to_alias[x]

        def a_edge(x):
            return book.real_to_alias[x]

        aliased_nodes = [
            {
                "id": a_node(n["id"]),
                "label": n["label"],
                "type": n["type"],
                "summary": n.get("summary", ""),
            }
            for n in ctx_nodes
        ]
        aliased_edges = [
            {
                "id": a_edge(e["id"]),
                "relation": e["relation"],
                "source_ids": [a_node(s) for s in e.get("source_ids", [])],
                "target_ids": [a_node(t) for t in e.get("target_ids", [])],
            }
            for e in ctx_edges
        ]

        if new_nodes:
            lines = [
                f"- {book.real_to_alias[rid]}: {next(n for n in ctx_nodes if n['id'] == rid)['label']}"
                for rid, _ in new_nodes
            ]
            nodes_str = "New node aliases:\n" + "\n".join(lines)
        else:
            nodes_str = "New node aliases: (none)"

        if new_edges:
            lines = []
            for rid, _ in new_edges:
                e = next(e for e in ctx_edges if e["id"] == rid)
                lines.append(f"- {book.real_to_alias[rid]}: {e['relation']}")
            edges_str = "New edge aliases:\n" + "\n".join(lines)
        else:
            edges_str = "New edge aliases: (none)"

        return aliased_nodes, aliased_edges, nodes_str, edges_str

    def repair_lean_offsets_for_mode(
        self,
        *,
        mode: ResolvedExtractionSchemaMode,
        payload: dict[str, Any],
        content: str,
        policy: OffsetMismatchPolicy,
        offset_repair_scorer: OffsetRepairScorer | None,
    ) -> dict[str, Any]:
        if policy not in {"strict", "exact", "exact_fuzzy"}:
            raise ValueError(
                f"Unsupported offset_mismatch_policy={policy!r}. "
                "Expected one of ['strict', 'exact', 'exact_fuzzy']"
            )
        if mode not in {"lean", "flattened_lean"}:
            return payload

        spans = self._iter_lean_spans_for_mode(mode=mode, payload=payload)
        if not spans:
            return payload

        scorer = self.resolve_offset_repair_scorer(offset_repair_scorer)
        failures: list[str] = []

        for path, span_row in spans:
            excerpt = str(span_row.get("excerpt") or "")
            start_char = span_row.get("start_char")
            end_char = span_row.get("end_char")
            if not isinstance(start_char, int) or not isinstance(end_char, int):
                failures.append(
                    f"{path}: start_char/end_char must be integers, got "
                    f"{type(start_char).__name__}/{type(end_char).__name__}"
                )
                continue

            if (
                0 <= start_char < end_char <= len(content)
                and content[start_char:end_char] == excerpt
            ):
                continue

            exact_matches = self._find_all_exact_occurrences(content, excerpt)
            if policy in {"exact", "exact_fuzzy"} and exact_matches:
                best_start = min(exact_matches, key=lambda s: (abs(s - start_char), s))
                span_row["start_char"] = best_start
                span_row["end_char"] = best_start + len(excerpt)
                continue

            best_fuzzy: tuple[int, int, float] | None = None
            if policy == "exact_fuzzy":
                best_fuzzy = self._find_best_fuzzy_span(
                    content=content,
                    excerpt=excerpt,
                    origin_start=max(0, start_char),
                    scorer=scorer,
                )
                if best_fuzzy is not None:
                    best_start, best_end, _score = best_fuzzy
                    span_row["start_char"] = best_start
                    span_row["end_char"] = best_end
                    span_row["excerpt"] = content[best_start:best_end]
                    continue

            failures.append(
                self._build_offset_failure_detail(
                    path=path,
                    content=content,
                    excerpt=excerpt,
                    start_char=start_char,
                    end_char=end_char,
                    exact_hits=len(exact_matches),
                    best_fuzzy_score=best_fuzzy[2] if best_fuzzy is not None else None,
                )
            )

        if failures:
            sample = " | ".join(failures[:3])
            raise ValueError(
                f"Offset repair failed mode={mode} policy={policy} "
                f"failed_spans={len(failures)} total_spans={len(spans)} :: {sample}"
            )
        return payload

    def coerce_pages(self, content_or_pages):
        if isinstance(content_or_pages, dict):
            items = sorted(
                ((int(k), v) for k, v in content_or_pages.items()), key=lambda x: x[0]
            )
            return [(p, str(t or "")) for p, t in items]

        if isinstance(content_or_pages, (list, tuple)):
            if not content_or_pages:
                return []
            first: Any = content_or_pages[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                return [(int(p), str(t or "")) for p, t in content_or_pages]
            if isinstance(first, dict) and "text" in first:
                out = []
                for i, item in enumerate(content_or_pages, start=1):
                    p = int(item.get("page", i))
                    out.append(
                        (
                            p,
                            str(
                                item.get(
                                    "text",
                                    "\n".join(
                                        i["text"]
                                        for i in item.get("OCR_text_clusters", "")
                                    ),
                                )
                                or ""
                            ),
                        )
                    )
                return out
            if isinstance(first, list) and first and "pdf_page_num" in first[0]:
                try:
                    return [(t["pdf_page_num"], str(t or "")) for t in content_or_pages]
                except KeyError:
                    raise Exception(
                        "Value inconsistency, each page is a dict, some have 'pdf_page_num' but some do not"
                    )
            return [(i, str(t or "")) for i, t in enumerate(content_or_pages, start=1)]

        if isinstance(content_or_pages, str):
            s = content_or_pages.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    loaded = json.loads(s)
                    return self.coerce_pages(loaded)
                except Exception:
                    pass
            if "\f" in s:
                parts = s.split("\f")
                return [(i, p) for i, p in enumerate(parts, start=1)]
            return [(1, s)]
        return [(1, str(content_or_pages))]

    def alias_doc_in_prompt(self) -> str:
        return f"Use '{self._doc_alias()}' whenever you need to reference the current document in ReferenceSession fields."

    def delias_one_span(self, span: Span, real_doc_id: str) -> Span:
        span = span.model_copy(deep=True)
        doc_alias = self._doc_alias()
        if span.document_page_url and doc_alias in span.document_page_url:
            span.document_page_url = span.document_page_url.replace(
                doc_alias, real_doc_id
            )
        if span.collection_page_url and doc_alias in span.collection_page_url:
            span.collection_page_url = span.collection_page_url.replace(
                doc_alias, real_doc_id
            )
        if (
            getattr(span, "doc_id", None) == doc_alias
            or getattr(span, "doc_id", None) is None
        ):
            span.doc_id = real_doc_id
        if (
            span.start_char is not None
            and span.end_char is not None
            and span.end_char < span.start_char
        ):
            span.end_char = span.start_char
        return span

    # Back-compat alias spelling.
    def dealias_one_span(self, span: Span, real_doc_id: str) -> Span:
        return self.delias_one_span(span, real_doc_id)

    def dealias_one_grounding(
        self, grounding: Grounding, real_doc_id: str
    ) -> Grounding:
        out: list[Span] = []
        for span in grounding.spans:
            out.append(self.delias_one_span(span, real_doc_id))
        grounding_type: Type = type(grounding)
        return grounding_type.model_validate({"spans": out})

    def dealias_span(
        self,
        mentions: List[Grounding] | None,
        real_doc_id: str,
    ):
        if not mentions or len(mentions) == 0:
            raise ValueError("No reference to dealias")
        return [self.dealias_one_grounding(r, real_doc_id) for r in mentions]

    def fetch_document_text(self, document_id: str) -> str:
        got = self._e.backend.document_get(ids=[document_id], include=["documents"])
        if got and got.get("documents"):
            docs = got.get("documents")
            if docs:
                return docs[0] or ""
            raise Exception("document lost")
        got = self._e.backend.document_get(
            where={"doc_id": document_id}, include=["documents"]
        )
        if got and got.get("documents"):
            docs = got.get("documents")
            if docs:
                return docs[0] or ""
            raise Exception("document lost")
        return ""

    def get_span_validator_of_doc_type(
        self,
        *,
        doc_id: str | None = None,
        doc_type: Literal["text", "ocr", "ocr_document"] | str | None = None,
        document: Document | None = None,
    ) -> BaseDocValidator:
        if (doc_id is not None) + (doc_type is not None) + (document is not None) != 1:
            raise ValueError("Must only specify one of doc_id, doc_type or document")

        from kogwistar.extraction import (
            OcrDocSpanValidator,
            PlainTextDocSpanValidator,
        )

        if doc_type is None:
            if doc_id is not None:
                document = self._e.read.get_document(doc_id)
            if document:
                doc_type = document.type
            else:
                raise ValueError("Unreachable")

        if doc_type == "text":
            return PlainTextDocSpanValidator()
        if doc_type in {"ocr", "ocr_document"}:
            return OcrDocSpanValidator()
        raise ValueError(f"No validator associated with document type {doc_type}")

    def extract_graph_with_llm(
        self,
        *,
        content: str,
        doc_type: str,
        alias_nodes_str="[Empty]",
        alias_edges_str="[Empty]",
        with_parsed=True,
        instruction_for_node_edge_contents_parsing_inclusion: None | str = None,
        validate=True,
        autofix: bool | str = True,
        last_iteration_result=None,
        extraction_schema_mode=None,
        offset_mismatch_policy="exact_fuzzy",
        offset_repair_scorer=None,
    ):
        """Pure: run LLM + parse + alias resolution. No writes."""
        # Keep compatibility seam: tests may monkeypatch engine._extract_graph_with_llm_aliases.
        raw, parsed, error = self._e._extract_graph_with_llm_aliases(
            content,
            alias_nodes_str=alias_nodes_str,
            alias_edges_str=alias_edges_str,
            instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
            last_iteration_result=last_iteration_result,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )
        if parsed is None:
            raise ValueError("parsed is None")
        if error:
            raise ValueError(error)
        validation_error_group = []
        if validate:
            temp_alias_book = AliasBook()
            parsed_copy = parsed.model_copy(deep=True)
            self._e._preflight_validate(parsed_copy, "", alias_book=temp_alias_book)

            if not (
                set([j for i in parsed_copy.edges for j in i.target_ids]).union(
                    set([j for i in parsed_copy.edges for j in i.source_ids])
                )
                <= set([i.id for i in parsed_copy.edges]).union(
                    set([i.id for i in parsed_copy.nodes])
                )
            ):
                raise Exception("LLM error, new uuid hallucinated")
            span_validator: BaseDocValidator = self._e.get_span_validator_of_doc_type(
                doc_type=doc_type
            )

            dummy_doc = Document(
                content=content,
                id=str(stable_id(f"doc::{doc_type}", content)),
                type=doc_type,
                metadata={},
                domain_id=None,
                processed=False,
                embeddings=None,
                source_map=None,
            )
            pre_parse_nodes_or_edges: list[Node | Edge] = parsed.nodes + parsed.edges
            for i, node_or_edge in enumerate(parsed_copy.nodes + parsed_copy.edges):
                node_or_edge: Node | Edge
                for g in node_or_edge.mentions:
                    for sp in g.spans:
                        result = span_validator.validate_span(doc=dummy_doc, span=sp)
                        if result["correctness"] is True:
                            pass
                        else:
                            if autofix:
                                if autofix is True:
                                    fix_result = span_validator.fix_span(
                                        doc=dummy_doc,
                                        span=sp,
                                        nodes_edges=parsed_copy.nodes
                                        + parsed_copy.edges,
                                    )
                                    result = fix_result
                                else:
                                    raise NotImplementedError(
                                        "string method options not iplemented"
                                    )
                            if result["correctness"] is False:
                                pre_parsed_node_or_edge: Node | Edge = (
                                    pre_parse_nodes_or_edges[i]
                                )
                                validation_error_group.append(
                                    f"Error found for {pre_parsed_node_or_edge.model_dump(field_mode='backend')}: {str(result)}"
                                )

        if with_parsed:
            return {
                "raw": raw,
                "parsed": parsed,
                "error": validation_error_group or None,
            }
        return {"raw": raw, "error": validation_error_group or None}

    def cached_extract_graph_with_llm(self, *args, **kwargs):
        return self._e._cached_extract_graph_with_llm(*args, **kwargs)
