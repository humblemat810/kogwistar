from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "rag_retrieval_comparison_tutorial.py"


def _load_module():
    return runpy.run_path(str(SCRIPT_PATH), run_name="tutorial_module")


def test_dataset_loads_and_has_expected_shape():
    ns = _load_module()
    docs = ns["load_dataset"]()
    assert len(docs) == 12
    assert all("text" in doc for doc in docs)

    demo = ns["build_demo"]()
    summary = demo.dataset_summary()
    assert summary["documents"] == 12
    assert summary["chunks"] >= 12
    assert summary["entities"] >= 20


def test_retrieval_methods_cover_the_expected_cases():
    ns = _load_module()
    demo = ns["build_demo"]()

    direct = demo.compare_query("Who is the product lead for Atlas?")
    assert direct["graph"]["answer"] == "Ben Ortiz leads Atlas."
    assert direct["hybrid"]["answer"] == "Ben Ortiz leads Atlas."
    assert direct["index"]["hits"][0]["doc_id"] == "doc1"

    keyword = demo.compare_query("Which document mentions keyword index API and vector database?")
    assert keyword["index"]["hits"][0]["doc_id"] == "doc3"
    assert "Aurora Vector Database" in keyword["graph"]["answer"]

    synonym = demo.compare_query("Who heads the safety project for Atlas?")
    assert synonym["graph"]["answer"].startswith("Alice Chen leads Safety Project.")
    assert synonym["hybrid"]["answer"].startswith("Alice Chen leads Safety Project.")

    multi_hop = demo.compare_query("Which project depends on the Aurora vector database and who leads it?")
    assert "Atlas depends on Aurora Vector Database." in multi_hop["graph"]["answer"]
    assert "Ben Ortiz leads Atlas." in multi_hop["graph"]["answer"]

    ambiguous = demo.compare_query("Tell me about Aurora.")
    assert ambiguous["graph"]["answer"].startswith("Aurora is ambiguous")
    assert ambiguous["hybrid"]["answer"].startswith("Aurora is ambiguous")

    complex_case = demo.compare_query("Who prefers the graph view, and what does she prefer it over?")
    assert complex_case["graph"]["answer"] == "Alice Chen prefers Graph View over Keyword Index."
    assert complex_case["hybrid"]["answer"] == "Alice Chen prefers Graph View over Keyword Index."


def test_hybrid_expands_beyond_keyword_candidates():
    ns = _load_module()
    demo = ns["build_demo"]()
    result = demo.compare_query("Who prefers the graph view, and what does she prefer it over?")

    candidate_ids = {item["doc_id"] for item in result["hybrid"]["candidate_docs"]}
    expanded_ids = set(result["hybrid"]["expanded_doc_ids"])
    assert expanded_ids
    assert candidate_ids.issubset(expanded_ids)
    assert expanded_ids - candidate_ids


def test_demo_is_deterministic():
    ns = _load_module()
    demo1 = ns["build_demo"]()
    demo2 = ns["build_demo"]()
    queries = [
        "Who is the product lead for Atlas?",
        "Which document mentions keyword index API and vector database?",
        "Who heads the safety project for Atlas?",
        "Which project depends on the Aurora vector database and who leads it?",
        "Tell me about Aurora.",
        "Who prefers the graph view, and what does she prefer it over?",
    ]
    results1 = [demo1.compare_query(query) for query in queries]
    results2 = [demo2.compare_query(query) for query in queries]
    assert results1 == results2
