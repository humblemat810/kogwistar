from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.ci_full
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    AssocFlattenedLLMGraphExtraction,
    Document,
    LLMGraphExtraction,
)
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)


def _fake_task_set(*, provider: str) -> LLMTaskSet:
    hints = LLMTaskProviderHints(
        extract_graph_provider=provider,  # type: ignore[arg-type]
        adjudicate_pair_provider=provider,  # type: ignore[arg-type]
        adjudicate_batch_provider=provider,  # type: ignore[arg-type]
        filter_candidates_provider=provider,  # type: ignore[arg-type]
        summarize_context_provider=provider,  # type: ignore[arg-type]
        answer_with_citations_provider=provider,  # type: ignore[arg-type]
        repair_citations_provider=provider,  # type: ignore[arg-type]
    )

    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload={"nodes": [], "edges": []}, parsing_error=None
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload={"same_entity": False}, raw=None, parsing_error=None
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error=None
        ),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(
            node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None
        ),
        summarize_context=lambda _req: SummarizeContextTaskResult(text=""),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload={"text": "", "reasoning": "", "claims": []},
            raw=None,
            parsing_error=None,
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload={"text": "", "reasoning": "", "claims": []},
            raw=None,
            parsing_error=None,
        ),
        provider_hints=hints,
    )


@pytest.fixture
def engine():
    run_id = uuid.uuid4().hex
    base = Path(".tmp_pytest_schema_modes")
    base.mkdir(parents=True, exist_ok=True)
    persist_dir = base / run_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    eng = GraphKnowledgeEngine(
        persist_directory=str(persist_dir),
        embedding_cache_path=str(persist_dir / "emb_cache"),
        llm_tasks=_fake_task_set(provider="gemini"),
    )
    try:
        yield eng
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "mode, expected_schema, expects_json_schema_method",
    [
        ("full", LLMGraphExtraction["llm"], False),
        ("lean", LLMGraphExtraction["llm_in"], False),
        ("flattened_lean", AssocFlattenedLLMGraphExtraction["llm_in"], True),
        ("flattened_full", AssocFlattenedLLMGraphExtraction["llm"], True),
    ],
)
def test_structured_schema_for_mode(
    engine, mode, expected_schema, expects_json_schema_method
):
    schema, prefer_json_schema = engine._structured_schema_for_mode(mode)
    assert schema is expected_schema
    assert prefer_json_schema is expects_json_schema_method


def test_build_structured_output_for_mode_is_back_compat_schema_tuple(engine):
    schema, prefer_json_schema = engine._build_structured_output_for_mode(
        "flattened_lean"
    )
    assert schema is AssocFlattenedLLMGraphExtraction["llm_in"]
    assert prefer_json_schema is True


def test_resolve_schema_mode_uses_engine_default_when_no_override(engine):
    engine.extraction_schema_mode = "flattened_full"
    resolved = engine._resolve_extraction_schema_mode()
    assert resolved == "flattened_full"


def test_resolve_schema_mode_override_takes_precedence(engine):
    engine.extraction_schema_mode = "lean"
    resolved = engine._resolve_extraction_schema_mode("flattened_full")
    assert resolved == "flattened_full"


def test_resolve_schema_mode_auto_maps_gemini_to_lean(engine):
    engine.extraction_schema_mode = "auto"
    engine.llm_tasks = _fake_task_set(provider="gemini")
    resolved = engine._resolve_extraction_schema_mode()
    assert resolved == "lean"


def test_resolve_schema_mode_auto_maps_non_gemini_to_full(engine):
    engine.extraction_schema_mode = "auto"
    engine.llm_tasks = _fake_task_set(provider="openai")
    resolved = engine._resolve_extraction_schema_mode()
    assert resolved == "full"


def test_extract_graph_with_llm_forwards_schema_mode_override(engine, monkeypatch):
    seen: dict[str, Any] = {}
    parsed = LLMGraphExtraction.model_validate({"nodes": [], "edges": []})
    scorer = lambda candidate, excerpt: 42.0  # noqa: E731

    def _fake_extract(
        content: str,
        alias_nodes_str: str,
        alias_edges_str: str,
        instruction_for_node_edge_contents_parsing_inclusion=None,
        last_iteration_result=None,
        extraction_schema_mode=None,
        offset_mismatch_policy=None,
        offset_repair_scorer=None,
    ):
        _ = content, alias_nodes_str, alias_edges_str
        _ = instruction_for_node_edge_contents_parsing_inclusion, last_iteration_result
        seen["mode"] = extraction_schema_mode
        seen["offset_mismatch_policy"] = offset_mismatch_policy
        seen["offset_repair_scorer"] = offset_repair_scorer
        return None, parsed, None

    monkeypatch.setattr(engine, "_extract_graph_with_llm_aliases", _fake_extract)

    _ = engine.extract_graph_with_llm(
        content="stub content",
        doc_type="text",
        validate=False,
        extraction_schema_mode="lean",
        offset_mismatch_policy="strict",
        offset_repair_scorer=scorer,
    )

    assert seen["mode"] == "lean"
    assert seen["offset_mismatch_policy"] == "strict"
    assert seen["offset_repair_scorer"] is scorer


def test_ingest_document_with_llm_forwards_schema_mode_override(engine, monkeypatch):
    seen: dict[str, Any] = {}
    scorer = lambda candidate, excerpt: 37.0  # noqa: E731

    def _fake_extract_graph_with_llm(
        *,
        extraction_schema_mode=None,
        offset_mismatch_policy=None,
        offset_repair_scorer=None,
        **kwargs,
    ):
        _ = kwargs
        seen["mode"] = extraction_schema_mode
        seen["offset_mismatch_policy"] = offset_mismatch_policy
        seen["offset_repair_scorer"] = offset_repair_scorer
        return {
            "raw": None,
            "parsed": LLMGraphExtraction.model_validate({"nodes": [], "edges": []}),
            "error": None,
        }

    def _fake_preflight_validate(parsed, doc_id, alias_book=None):
        _ = parsed, doc_id, alias_book
        return None

    def _fake_persist_graph_extraction(*, document, parsed, mode):
        _ = parsed, mode
        return {
            "document_id": document.id,
            "node_ids": [],
            "edge_ids": [],
            "nodes_added": 0,
            "edges_added": 0,
        }

    monkeypatch.setattr(engine, "extract_graph_with_llm", _fake_extract_graph_with_llm)
    monkeypatch.setattr(engine, "_preflight_validate", _fake_preflight_validate)
    monkeypatch.setattr(
        engine, "persist_graph_extraction", _fake_persist_graph_extraction
    )
    monkeypatch.setattr(engine, "add_document", lambda document: document.id)

    document = Document(
        id="doc:test",
        content="stub content",
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    _ = engine.ingest_document_with_llm(
        document,
        extraction_schema_mode="flattened_full",
        offset_mismatch_policy="strict",
        offset_repair_scorer=scorer,
    )

    assert seen["mode"] == "flattened_full"
    assert seen["offset_mismatch_policy"] == "strict"
    assert seen["offset_repair_scorer"] is scorer


def test_add_page_forwards_schema_mode_override(engine, monkeypatch):
    seen_modes: list[str | None] = []
    seen_policies: list[str | None] = []
    seen_scorers: list[Any] = []
    scorer = lambda candidate, excerpt: 11.0  # noqa: E731

    def _fake_ingest_text_with_llm(
        *,
        doc_id,
        content,
        auto_adjudicate=False,
        extraction_schema_mode=None,
        offset_mismatch_policy=None,
        offset_repair_scorer=None,
    ):
        _ = doc_id, content, auto_adjudicate
        seen_modes.append(extraction_schema_mode)
        seen_policies.append(offset_mismatch_policy)
        seen_scorers.append(offset_repair_scorer)
        return {"nodes_added": 0, "edges_added": 0, "raw": None}

    monkeypatch.setattr(engine, "_ingest_text_with_llm", _fake_ingest_text_with_llm)

    _ = engine.add_page(
        document_id="doc:test",
        page_text="page one",
        extraction_schema_mode="flattened_lean",
        offset_mismatch_policy="strict",
        offset_repair_scorer=scorer,
    )

    assert seen_modes == ["flattened_lean"]
    assert seen_policies == ["strict"]
    assert seen_scorers == [scorer]
