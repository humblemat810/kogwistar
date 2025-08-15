# tests/test_graph_query_real_engine.py
from __future__ import annotations
import os
import json
import typing as T

import graph_knowledge_engine.engine as engmod
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.graph_query import GraphQuery
from graph_knowledge_engine.models import Document, Node, Edge, ReferenceSession
import pytest
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


@pytest.fixture()
def small_graph():
    e = GraphKnowledgeEngine(persist_directory = "small_graph")
    doc_id = "D1"
    # nodes
    def add_node(nid, label):
        n = Node(id=nid, label=label, type="entity", summary=label, references=[ReferenceSession(
            collection_page_url=f"document_collection/{doc_id}", document_page_url=f"document/{doc_id}", doc_id=doc_id,
            start_page=1, end_page=1, start_char=0, end_char=1
        )], doc_id=doc_id)
        e.node_collection.add(ids=[nid], documents=[n.model_dump_json()], metadatas=[{"doc_id": doc_id, "label": n.label, "type": n.type}])
        # node_docs link
        ndid = f"{nid}::{doc_id}"
        row = {"id": ndid, "node_id": nid, "doc_id": doc_id}
        e.node_docs_collection.add(ids=[ndid], documents=[json.dumps(row)], metadatas=[row])
        return n

    A = add_node("A", "Smoking")
    B = add_node("B", "Lung Cancer")
    C = add_node("C", "Cough")

    # edge A -[causes]-> B
    e_id = "E1"
    edge = Edge(id=e_id, label="Smoking causes Lung Cancer", type="relationship", summary="causal", relation="causes",
                source_ids=["A"], target_ids=["B"], source_edge_ids=[], target_edge_ids=[],
                references=A.references, doc_id=doc_id)
    e.edge_collection.add(ids=[e_id], documents=[edge.model_dump_json()], metadatas=[{"doc_id": doc_id, "relation": "causes"}])
    # endpoints fan-out
    rows = [
        {"id": f"{e_id}::src::node::A", "edge_id": e_id, "endpoint_id": "A", "endpoint_type": "node", "role": "src", "relation": "causes", "doc_id": doc_id},
        {"id": f"{e_id}::tgt::node::B", "edge_id": e_id, "endpoint_id": "B", "endpoint_type": "node", "role": "tgt", "relation": "causes", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows], documents=[json.dumps(r) for r in rows], metadatas=rows)

    # final summary link S -> docnode:D1
    S = add_node("S", "Final Summary")
    e_id2 = "E2"
    e.edge_collection.add(ids=[e_id2], documents=[Edge(
        id=e_id2, label="summarizes_document", type="relationship", summary="S summarizes document", relation="summarizes_document",
        source_ids=["S"], target_ids=[f"docnode:{doc_id}"], source_edge_ids=[], target_edge_ids=[], references=S.references, doc_id=doc_id
    ).model_dump_json()], metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}])
    rows2 = [
        {"id": f"{e_id2}::src::node::S", "edge_id": e_id2, "endpoint_id": "S", "endpoint_type": "node", "role": "src", "relation": "summarizes_document", "doc_id": doc_id},
        {"id": f"{e_id2}::tgt::node::docnode:{doc_id}", "edge_id": e_id2, "endpoint_id": f"docnode:{doc_id}", "endpoint_type": "node", "role": "tgt", "relation": "summarizes_document", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows2], documents=[json.dumps(r) for r in rows2], metadatas=rows2)

    return e, doc_id
def _ref(doc_id: str, snippet: str = "") -> ReferenceSession:
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id=doc_id,
        start_page=1,
        end_page=1,
        start_char=0,
        end_char=max(0, len(snippet)),
        snippet=snippet or None,
    )


def make_engine(tmp_path) -> GraphKnowledgeEngine:
    # Monkeypatch the engine to avoid AzureChatOpenAI at construction time
    engmod.AzureChatOpenAI = _DummyLLM
    e = GraphKnowledgeEngine(persist_directory=str(tmp_path))
    return e


def test_graph_query_structural_end_to_end(tmp_path):
    e = make_engine(tmp_path)

    # 1) Real document row
    doc = Document(id="D1", content="Smoking causes lung cancer.", type="plain")
    e.add_document(doc)

    # 2) Real nodes (persisted into Chroma)
    n_smoke = Node(
        label="Smoking", type="entity", summary="habit",
        references=[_ref(doc.id, "Smoking")], doc_id=doc.id,
    )
    n_cancer = Node(
        label="Lung cancer", type="entity", summary="disease",
        references=[_ref(doc.id, "lung cancer")], doc_id=doc.id,
    )
    e.add_node(n_smoke, doc_id=doc.id)
    e.add_node(n_cancer, doc_id=doc.id)

    # 3) Real edge (engine will fan out edge_endpoints rows)
    e_causes = Edge(
        label="Smoking→Cancer", type="relationship", relation="causes",
        source_ids=[n_smoke.id], target_ids=[n_cancer.id], source_edge_ids=[], target_edge_ids = [], summary="causal claim",
        references=[_ref(doc.id, "causes")], doc_id=doc.id,
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
        src_label_contains="smoke",
        tgt_label_contains="cancer",
        doc_id=doc.id,
    )
    assert e_causes.id in results
    
def test_semantic_seed_then_expand_text(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    out = gq.semantic_seed_then_expand_text("smok", top_k=5, hops=1)
    assert out["seeds"]  # should find A or related
    assert isinstance(out["layers"], list) and out["layers"]