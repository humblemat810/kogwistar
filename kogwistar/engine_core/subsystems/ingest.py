from __future__ import annotations

from typing import Any, Optional, Tuple

from ..async_compat import run_awaitable_blocking
from ...llm_tasks import ExtractGraphTaskRequest

from ..models import (
    AdjudicationVerdict,
    Document,
    LLMGraphExtraction,
    Node,
)
from ..types import ExtractionSchemaMode, OffsetMismatchPolicy, OffsetRepairScorer
from ..utils.pages import coerce_pages
from .base import NamespaceProxy


class IngestSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def ingest_document_with_llm(
        self,
        document: Document,
        *,
        mode: str = "append",
        instruction_for_node_edge_contents_parsing_inclusion=None,
        raw_with_parsed=None,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        if raw_with_parsed is None:
            raw_with_parsed = {}
        self._e.write.add_document(document)
        extracted = self._e.extract_graph_with_llm(
            content=str(document.content),
            doc_type=document.type,
            instruction_for_node_edge_contents_parsing_inclusion=instruction_for_node_edge_contents_parsing_inclusion,
            last_iteration_result=raw_with_parsed,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )
        parsed = extracted["parsed"]
        self._e.persist.preflight_validate(parsed, document.id)
        return self._e.persist.persist_graph_extraction(
            document=document,
            parsed=parsed,
            mode=mode,
        )

    def ingest_text_with_llm(
        self,
        *,
        doc_id: str,
        content: str,
        auto_adjudicate: bool = False,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        ctx_nodes, ctx_edges = self._e.persist.select_doc_context(doc_id)
        _, _, alias_nodes_str, alias_edges_str = self._e.extract.aliasify_for_prompt(
            doc_id,
            ctx_nodes,
            ctx_edges,
        )

        raw, parsed, error = self._e.extract.extract_graph_with_llm_aliases(
            content,
            alias_nodes_str,
            alias_edges_str,
            extraction_schema_mode=extraction_schema_mode,
            offset_mismatch_policy=offset_mismatch_policy,
            offset_repair_scorer=offset_repair_scorer,
        )
        if error:
            raise ValueError(f"LLM parsing error: {error}")
        if not isinstance(parsed, LLMGraphExtraction):
            parsed = LLMGraphExtraction.model_validate(parsed)

        parsed = self._e.extract.de_alias_ids_in_result(doc_id, parsed)
        self._e.persist.preflight_validate(parsed, doc_id)
        res = self._e.persist.ingest_with_toposort(parsed, doc_id=doc_id)
        res["raw"] = raw

        if auto_adjudicate:
            data = run_awaitable_blocking(
                self._e.backend.node_get(where={"doc_id": doc_id}, include=["documents"])
            )
            buckets = {}
            for ndoc in data.get("documents") or []:
                n = Node.model_validate_json(ndoc)
                buckets.setdefault((n.type, n.label.strip().lower()), []).append(n)
            pairs = []
            for _, bucket in buckets.items():
                if len(bucket) > 1:
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            pairs.append((bucket[i], bucket[j]))
            if pairs:
                verdicts, _ = self._e.batch_adjudicate_merges(pairs)  # type: ignore
                verdicts = verdicts  # help type checker for getattr fallback below
                for (left, right), out in zip(pairs, verdicts):
                    verdict: AdjudicationVerdict = getattr(out, "verdict", out)
                    if verdict.same_entity:
                        self._e.commit_merge(left, right, verdict)
        return res

    def extract_graph_with_llm_internal(
        self,
        content: str,
        doc: Document,
    ) -> Tuple[Any, Optional[LLMGraphExtraction], Optional[str]]:
        result = self._e.llm_tasks.extract_graph(
            ExtractGraphTaskRequest(
                content=content,
                alias_nodes="[Empty]",
                alias_edges="[Empty]",
                doc_alias=str(doc.id),
                instruction="Extract entities and relationships as nodes and edges in a hypergraph.",
                prompt_rules=(
                    "Rules:\n"
                    "1) Each node/edge must include at least one grounding span.\n"
                    "2) Do not invent UUIDs or external URLs.\n"
                ),
                schema_mode="full",
            )
        )
        parsed_payload = result.parsed_payload
        parsed = None
        if parsed_payload is not None:
            parsed = LLMGraphExtraction.model_validate(parsed_payload)
        return result.raw, parsed, result.parsing_error

    def add_page(
        self,
        *,
        document_id: str,
        page_text: str | list[str] | dict[str, Any],
        page_number: int | None = None,
        auto_adjudicate: bool = True,
        extraction_schema_mode: ExtractionSchemaMode | None = None,
        offset_mismatch_policy: OffsetMismatchPolicy = "exact_fuzzy",
        offset_repair_scorer: OffsetRepairScorer | None = None,
    ):
        pages = coerce_pages(
            {"pages": [{"page_number": page_number, "text": page_text}]}
            if isinstance(page_text, str)
            else page_text
        )
        if not pages:
            return {"document_id": document_id, "nodes_added": 0, "edges_added": 0}

        total_nodes = total_edges = 0
        raw_by_page = []
        for pg in pages:
            # Keep legacy seam so tests/callers monkeypatching engine._ingest_text_with_llm still work.
            res = self._e._ingest_text_with_llm(
                doc_id=document_id,
                content=pg["text"],
                auto_adjudicate=auto_adjudicate,
                extraction_schema_mode=extraction_schema_mode,
                offset_mismatch_policy=offset_mismatch_policy,
                offset_repair_scorer=offset_repair_scorer,
            )
            total_nodes += res["nodes_added"]
            total_edges += res["edges_added"]
            raw_by_page.append({"page": pg["page_number"], "raw": res.get("raw")})

        return {
            "document_id": document_id,
            "nodes_added": total_nodes,
            "edges_added": total_edges,
            "raw_by_page": raw_by_page,
        }
