# tests/test_graph_query_real_engine.py
from __future__ import annotations
import os
import typing as T

import graph_knowledge_engine.engine_core.engineas engmod
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.engine_core.models import Document, Node, Edge, Span
# ---- Test helpers ----
class _DummyLLM:
    """Prevents AzureChatOpenAI from initializing/networking during tests."""
    def __init__(self, *_, **__):
        pass
    def with_structured_output(self, *_a, **_k):
        class _Chain:
            def invoke(self, *_x, **_y):
                return {"raw": "(disabled)", "parsed": None, "parsing_error": "LLM disabled in tests"}
        return _Chain()


def _ref(doc_id: str, excerpt: str = "") -> Span:
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id=doc_id,
        start_page=1,
        end_page=1,
        start_char=0,
        insertion_method='test-manual',
        end_char=max(0, len(excerpt)),
        excerpt=excerpt or None,
    )


def make_engine(tmp_path) -> GraphKnowledgeEngine:
    # Monkeypatch the engine to avoid AzureChatOpenAI at construction time
    engmod.AzureChatOpenAI = _DummyLLM
    e = GraphKnowledgeEngine(persist_directory=str(tmp_path))
    return e


def test_graph_query_structural_end_to_end(tmp_path):
    e = make_engine(tmp_path)

    # 1) Real document row
    doc = Document(id="D1", content="Smoking causes lung cancer.", type="text")
    e.add_document(doc)

    # 2) Real nodes (persisted into Chroma)
    n_smoke = Node(
        label="Smoking", type="entity", summary="habit",
        mentions=[_ref(doc.id, "Smoking")], doc_id=doc.id,
    )
    n_cancer = Node(
        label="Lung cancer", type="entity", summary="disease",
        mentions=[_ref(doc.id, "lung cancer")], doc_id=doc.id,
    )
    e.add_node(n_smoke, doc_id=doc.id)
    e.add_node(n_cancer, doc_id=doc.id)

    # 3) Real edge (engine will fan out edge_endpoints rows)
    e_causes = Edge(
        label="Smoking→Cancer", type="relationship", relation="causes",
        source_ids=[n_smoke.id], target_ids=[n_cancer.id], source_edge_ids=[], target_edge_ids = [], summary="causal claim",
        mentions=[_ref(doc.id, "causes")], doc_id=doc.id,
    )
    e.add_edge(e_causes, doc_id=doc.id)

    # 4) Graph queries against the *real* collections
    gq = GraphQuery(e)

    # neighbors(node) should include the edge + opposite node
    nbrs = gq.neighbors(n_smoke.id)
    assert e_causes.id in nbrs["edges"], f"Expected {e_causes.id} in edges, got {nbrs}"
    assert n_cancer.id in nbrs["nodes"], f"Expected {n_cancer.id} in nodes, got {nbrs}"
    nbrs = gq.neighbors(n_smoke.id, allow_jump_edge=False)
    assert len(nbrs["nodes"]) == 0, "node should not jump over edge to another jode in no jump mode"
    # k-hop expansion
    layers = gq.k_hop([n_smoke.id], k=2)
    assert e_causes.id in layers[0]["edges"]
    assert n_cancer.id in layers[1]["nodes"]

    # shortest path (unweighted)
    path = gq.shortest_path(n_smoke.id, n_cancer.id)
    assert path and path[0] == n_smoke.id and path[-1] == n_cancer.id
    assert e_causes.id in path

    # filtering edges by relation + endpoint labels + doc filter
    results = gq.find_edges(
        relation="causes",
        src_label_contains="smok",
        tgt_label_contains="cancer",
        doc_id=doc.id,
    )
    assert e_causes.id in results
    
def test_semantic_seed_then_expand_text(real_small_graph):
    e, doc_id = real_small_graph
    gq = GraphQuery(e)
    out = gq.semantic_seed_then_expand_text("smok", top_k=5, hops=1)
    assert out["seeds"]  # should find A or related
    assert isinstance(out["layers"], list) and out["layers"]