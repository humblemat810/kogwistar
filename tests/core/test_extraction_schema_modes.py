from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    AssocFlattenedLLMGraphExtraction,
    Document,
    LLMGraphExtraction,
)


class _SchemaCaptureLLM:
    def __init__(self, *, reject_method: bool = False):
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.reject_method = reject_method

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, dict(kwargs)))
        if self.reject_method and "method" in kwargs:
            raise TypeError("method unsupported")
        return self


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
def test_build_structured_output_uses_expected_schema(engine, mode, expected_schema, expects_json_schema_method):
    capture = _SchemaCaptureLLM()
    engine.llm = capture

    _ = engine._build_structured_output_for_mode(mode)

    schema, kwargs = capture.calls[-1]
    assert schema is expected_schema
    assert kwargs.get("include_raw") is True
    if expects_json_schema_method:
        assert kwargs.get("method") == "json_schema"
    else:
        assert "method" not in kwargs


def test_build_structured_output_falls_back_when_method_kw_not_supported(engine):
    capture = _SchemaCaptureLLM(reject_method=True)
    engine.llm = capture

    _ = engine._build_structured_output_for_mode("flattened_lean")

    assert len(capture.calls) == 2
    first_schema, first_kwargs = capture.calls[0]
    second_schema, second_kwargs = capture.calls[1]
    assert first_schema is AssocFlattenedLLMGraphExtraction["llm_in"]
    assert first_kwargs.get("method") == "json_schema"
    assert first_kwargs.get("include_raw") is True
    assert second_schema is AssocFlattenedLLMGraphExtraction["llm_in"]
    assert second_kwargs.get("include_raw") is True
    assert "method" not in second_kwargs


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

    resolved = engine._resolve_extraction_schema_mode()

    assert resolved == "lean"


def test_resolve_schema_mode_auto_maps_non_gemini_to_full(engine):
    class _NonGeminiLLM:
        pass

    engine.extraction_schema_mode = "auto"
    engine.llm = _NonGeminiLLM()

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
        return {"raw": None, "parsed": LLMGraphExtraction.model_validate({"nodes": [], "edges": []}), "error": None}

    def _fake_preflight_validate(parsed, doc_id, alias_book=None):
        _ = parsed, doc_id, alias_book
        return None

    def _fake_persist_graph_extraction(*, document, parsed, mode):
        _ = parsed, mode
        return {"document_id": document.id, "node_ids": [], "edge_ids": [], "nodes_added": 0, "edges_added": 0}

    monkeypatch.setattr(engine, "extract_graph_with_llm", _fake_extract_graph_with_llm)
    monkeypatch.setattr(engine, "_preflight_validate", _fake_preflight_validate)
    monkeypatch.setattr(engine, "persist_graph_extraction", _fake_persist_graph_extraction)
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
