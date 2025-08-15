# tests/test_mcp_endpoints.py — end-to-end MCP tests with REAL engine/collections, offline LLMs
from __future__ import annotations
import json
import re
import typing as T

import graph_knowledge_engine.engine as engmod
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.mcp_endpoints import KnowledgeMCP


# --- Minimal chain wrapper ---
class _Chain:
    def __init__(self, fn):
        self._fn = fn
    def invoke(self, *a, **k):
        return self._fn(*a, **k)


# --- Offline Dummy LLM that satisfies engine/ingester expectations ---
class _DummyLLM:
    """
    Provides .with_structured_output(schema, include_raw=False) -> object with .invoke(...)
    Returns deterministic, schema-shaped payloads for summarization, grouping, and KG extraction.
    """
    def __init__(self, *_, **__):
        pass

    def with_structured_output(self, schema, include_raw: bool = False):
        name = getattr(schema, "__name__", str(schema))

        # SummarizeResponse (produce micro-chunks from text)
        if "SummarizeResponse" in name:
            def _summarize(prompt_text):
                text = prompt_text or ""
                parts = [p.strip() for p in text.split(".") if p.strip()]
                if not parts:
                    parts = [text.strip() or "(empty)"]
                items = []
                start = 0
                for i, s in enumerate(parts):
                    items.append({
                        "title": s[:30] or f"Chunk {i+1}",
                        "summary": s,
                        "start_page": 1,
                        "end_page": 1,
                        "start_char": start,
                        "end_char": start + len(s),
                    })
                    start += len(s) + 1
                return {"micro_chunks": items}
            return _Chain(lambda text, *_: _summarize(text))

        # GroupResponse (group indices)
        if "GroupResponse" in name:
            def _group(items_text, max_groups=1):
                idxs = [int(x) for x in re.findall(r"\[(\d+)\]", items_text or "")]
                if not idxs:
                    idxs = [0]
                return {"groups": [{
                    "title": "Group 1",
                    "summary": "Grouped",
                    "member_indices": sorted(set(idxs)),
                }]}
            return _Chain(lambda items_text, max_groups=1: _group(items_text, max_groups))

        # LLMGraphExtraction (very small pattern: X causes Y)
        if "LLMGraphExtraction" in name:
            def _extract(payload):
                content = (payload or {}).get("document") or (payload or {}).get("content") or ""
                src = "Smoking" if "smok" in content.lower() else "A"
                tgt = "Lung cancer" if "cancer" in content.lower() else "B"
                return {
                    "raw": "dummy",
                    "parsed": {
                        "nodes": [
                            {"id": None, "local_id": "nn:src", "label": src, "type": "entity", "summary": src,
                             "references": [{"doc_id": "_DOC_", "document_page_url": "document/_DOC_", "collection_page_url": "document_collection/_DOC_", "start_page": 1, "end_page": 1, "start_char": 0, "end_char": 1}]},
                            {"id": None, "local_id": "nn:tgt", "label": tgt, "type": "entity", "summary": tgt,
                             "references": [{"doc_id": "_DOC_", "document_page_url": "document/_DOC_", "collection_page_url": "document_collection/_DOC_", "start_page": 1, "end_page": 1, "start_char": 0, "end_char": 1}]},
                        ],
                        "edges": [
                            {"id": None, "local_id": "ne:rel", "label": "causal", "type": "relationship", "summary": "src→tgt", "relation": "causes",
                             "source_ids": ["nn:src"], "target_ids": ["nn:tgt"],
                             "references": [{"doc_id": "_DOC_", "document_page_url": "document/_DOC_", "collection_page_url": "document_collection/_DOC_", "start_page": 1, "end_page": 1, "start_char": 0, "end_char": 1}]}
                        ],
                    },
                    "parsing_error": None,
                }
            return _Chain(_extract)

        # Fallback
        return _Chain(lambda *a, **k: {"raw": "dummy", "parsed": None, "parsing_error": "noop"})


def _make_engine(tmp_path) -> GraphKnowledgeEngine:
    # Avoid real Azure client creation
    engmod.AzureChatOpenAI = _DummyLLM
    return GraphKnowledgeEngine(persist_directory=str(tmp_path))


def test_mcp_endpoints_end_to_end(tmp_path):
    engine = _make_engine(tmp_path)
    mcp = KnowledgeMCP(engine, ingester_llm=_DummyLLM())

    # 1) Parse document to a summary tree
    DOC = "D_SUMMARY"
    TEXT = "Smoking causes lung cancer. Symptoms include cough and chest pain. Treatment varies."
    out1 = mcp.call("doc.parse_summary_tree", {
        "doc_id": DOC,
        "content": TEXT,
        "split_max_chars": 200,
        "group_size": 3,
        "max_levels": 3,
        "force_concat_after_levels": 2,
    })
    assert out1.get("ok") is True
    assert out1.get("final_summary_node_id")

    # 2) Summary-tree helpers
    final_node = mcp.call("doc.query", {"doc_id": DOC, "what": "final_summary_node"})
    assert final_node.get("node")

    level0 = mcp.call("doc.query", {"doc_id": DOC, "what": "level_nodes", "level": 0})
    assert isinstance(level0.get("node_ids"), list)

    # 3) Extract KG facts from text, then query causal edges
    DOC2 = "D_KG"
    out2 = mcp.call("kg.extract_graph", {"doc_id": DOC2, "content": "Smoking causes lung cancer."})
    assert out2.get("ok") is True

    edges = mcp.call("kg.query", {"op": "find_edges", "relation": "causes", "doc_id": DOC2})
    assert isinstance(edges.get("result"), list)

    # 4) Semantic search (TEXT) — stub node_collection.query to be deterministic
    original_query = engine.node_collection.query
    try:
        all_ids = engine.node_collection.get(include=["ids"]).get("ids") or []
        engine.node_collection.query = lambda query_texts=None, query_embeddings=None, n_results=5: {
            "ids": [all_ids[: max(1, int(n_results))]]
        }
        sem = mcp.call("kg.semantic_search", {"text": "smoking cancer", "top_k": 3, "hops": 1})
        assert sem.get("ok") is True and sem.get("seeds")
        assert isinstance(sem.get("layers"), list)
    finally:
        engine.node_collection.query = original_query
