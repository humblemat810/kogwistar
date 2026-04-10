from __future__ import annotations

import contextlib
import io
import runpy
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rag_retrieval_comparison_kg_semantics.py"


def _load_module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="tutorial_module")


def test_kg_dataset_loads_and_builds_graph():
    ns = _load_module()
    docs = ns["load_dataset"]()
    demo = ns["KGSemanticsRetrievalTutorial"](docs, backend="memory")
    summary = demo.dataset_summary()
    assert summary["documents"] == 12
    assert summary["nodes_in_graph"] > 0
    assert summary["edges_in_graph"] > 0
    assert demo.records
    assert any(record.kind == "node" for record in demo.records)
    assert any(record.kind == "edge" for record in demo.records)
    assert demo.records[0].record_id in demo.records_by_id


def test_kg_tutorial_matches_raw_answers_for_all_cases():
    kg_ns = _load_module()
    raw_ns = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts" / "rag_retrieval_comparison_tutorial.py"),
        run_name="tutorial_module",
    )
    docs = kg_ns["load_dataset"]()
    kg_demo = kg_ns["KGSemanticsRetrievalTutorial"](docs, backend="memory")
    raw_demo = raw_ns["RetrievalTutorial"](docs)

    for query in kg_ns["QUERY_SET"]:
        kg_result = kg_demo.compare_query(query)
        raw_result = raw_demo.compare_query(query)
        for method in ("vector", "index", "graph", "hybrid"):
            assert kg_result[method]["answer"] == raw_result[method]["answer"]
            assert kg_result[method]


def test_cli_memory_backend_runs_end_to_end():
    ns = _load_module()
    stdout = io.StringIO()
    argv = sys.argv[:]
    try:
        sys.argv = [str(SCRIPT_PATH), "--backend", "memory", "--top-k", "2", "--json"]
        with contextlib.redirect_stdout(stdout):
            ns["main"]()
    finally:
        sys.argv = argv
    output = stdout.getvalue()
    assert '"backend": "memory"' in output
    assert '"all_parity_true": true' in output.lower()


def test_graph_query_can_traverse_memory_engine():
    ns = _load_module()
    docs = ns["load_dataset"]()
    demo = ns["KGSemanticsRetrievalTutorial"](docs, backend="memory")
    gq = demo.graph

    atlas_nodes = gq.search_nodes(label_contains="Atlas", limit=10)
    assert atlas_nodes
    path = gq.path_between_labels("Atlas", "Ben Ortiz")
    assert path


def test_kg_parity_is_true_in_batch_demo():
    ns = _load_module()
    payload = ns["run_demo"](backend="memory")
    for raw, kg in zip(payload["raw_results"], payload["results"]):
        for method in ("vector", "index", "graph", "hybrid"):
            assert raw[method]["answer"] == kg[method]["answer"]
    assert all(
        raw[method]["answer"] == kg[method]["answer"]
        for raw, kg in zip(payload["raw_results"], payload["results"])
        for method in ("vector", "index", "graph", "hybrid")
    )


def test_chroma_backend_is_optional_but_wired():
    ns = _load_module()
    if not ns["HAVE_REAL_GRAPH_KNOWLEDGE_ENGINE"]:
        pytest.skip("GraphKnowledgeEngine is not available in this environment")
    docs = ns["load_dataset"]()
    try:
        demo = ns["KGSemanticsRetrievalTutorial"](
            docs,
            backend="chroma",
            persist_directory=str(Path(ns["ROOT"]) / ".gke-data" / "pytest-kg-chroma"),
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Chroma backend unavailable: {exc}")
    assert demo.dataset_summary()["nodes_in_graph"] > 0
    result = demo.compare_query(ns["QUERY_SET"][0])
    assert result["vector"]["hits"]
