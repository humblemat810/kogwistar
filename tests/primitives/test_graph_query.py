# tests/test_graph_query.py — lightweight unit tests with fake in‑memory collections
import json
import pytest

# Minimal fake Chroma-like collection that supports get(where=...) and get(ids=...)
class FakeCollection:
    def __init__(self):
        self._docs = {}      # id -> document (json string)
        self._metas = {}     # id -> metadata (dict)

    def add(self, ids=None, documents=None, metadatas=None, embeddings=None):
        for i, rid in enumerate(ids or []):
            self._docs[rid] = (documents or [None])[i]
            self._metas[rid] = (metadatas or [None])[i]

    def update(self, ids=None, documents=None, metadatas=None):
        for i, rid in enumerate(ids or []):
            if documents:
                self._docs[rid] = documents[i]
            if metadatas:
                self._metas[rid] = metadatas[i]

    def delete(self, ids=None, where=None):
        if ids:
            for rid in ids:
                self._docs.pop(rid, None)
                self._metas.pop(rid, None)
        elif where:
            to_del = [rid for rid, meta in self._metas.items() if _match_where(meta, where)]
            for rid in to_del:
                self._docs.pop(rid, None)
                self._metas.pop(rid, None)

    def get(self, ids=None, where=None, include=None):
        ids = list(ids) if ids else None
        if ids is None and where is not None:
            ids = [rid for rid, meta in self._metas.items() if _match_where(meta, where)]
        if ids is None:
            ids = list(self._docs.keys())
        docs = [self._docs.get(rid) for rid in ids]
        metas = [self._metas.get(rid) for rid in ids]
        out = {"ids": ids}
        if include and "documents" in include:
            out["documents"] = docs
        if include and "metadatas" in include:
            out["metadatas"] = metas
        return out
    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        """Very small stub: if text provided, returns ids whose document JSON contains the text (case-insensitive)."""
        if query_texts:
            text = (query_texts[0] or "").lower()
            ranked = [rid for rid, doc in self._docs.items() if isinstance(doc, str) and text in doc.lower()]
        else:
            # if embeddings are provided, just return the first n ids deterministically
            ranked = list(self._docs.keys())
        return {"ids": [ranked[: max(1, int(n_results))]]}

# very small subset of Chroma's where filter: supports equality on flat keys, $and, and $in

def _match_where(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(meta, w) for w in where["$and"])
    for k, v in where.items():
        if isinstance(v, dict) and "$in" in v:
            if meta.get(k) not in set(v["$in"]):
                return False
        else:
            if meta.get(k) != v:
                return False
    return True

# Minimal Node/Edge JSON helpers (import models from package under test)
from graph_knowledge_engine.engine_core.models import Node, Edge, Span
from graph_knowledge_engine.graph_query import GraphQuery

class FakeEngine:
    def __init__(self):
        self.node_collection = FakeCollection()
        self.edge_collection = FakeCollection()
        self.edge_endpoints_collection = FakeCollection()
        self.node_docs_collection = FakeCollection()

    # replicate engine helpers used by GraphQuery
    def _nodes_by_doc(self, doc_id: str):
        got = self.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = got.get("metadatas") or []
        return [m["node_id"] for m in metas if m]

    def _edge_ids_by_doc(self, doc_id: str):
        got = self.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metas = got.get("metadatas") or []
        return sorted({m["edge_id"] for m in metas if m})

    def get_nodes(self, ids):
        got = self.node_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Node.model_validate_json(d) for d in (got.get("documents") or []) if d]

    def get_edges(self, ids):
        got = self.edge_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Edge.model_validate_json(d) for d in (got.get("documents") or []) if d]

@pytest.fixture()
def small_graph():
    e = FakeEngine()
    doc_id = "D1"
    # nodes
    def add_node(nid, label):
        n = Node(id=nid, label=label, type="entity", summary=label, 
                    mentions=[Span(
                                        collection_page_url=f"document_collection/{doc_id}", 
                                        document_page_url=f"document/{doc_id}", 
                                        doc_id=doc_id, 
                                        insertion_method = "manual", # document insertion
                                        start_page=1, end_page=1, start_char=0, end_char=1
                    )], doc_id=doc_id)
        e.node_collection.add(ids=[nid], documents=[n.model_dump_json(field_mode="backend")], metadatas=[{"doc_id": doc_id, "label": n.label, 
                                        "type": n.type}])
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
                mentions=A.mentions, doc_id=doc_id)
    e.edge_collection.add(ids=[e_id], documents=[edge.model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "causes"}])
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
        source_ids=["S"], target_ids=[f"docnode:{doc_id}"], source_edge_ids=[], target_edge_ids=[], mentions=S.mentions, doc_id=doc_id
    ).model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}])
    rows2 = [
        {"id": f"{e_id2}::src::node::S", "edge_id": e_id2, "endpoint_id": "S", "endpoint_type": "node", "role": "src", "relation": "summarizes_document", "doc_id": doc_id},
        {"id": f"{e_id2}::tgt::node::docnode:{doc_id}", "edge_id": e_id2, "endpoint_id": f"docnode:{doc_id}", "endpoint_type": "node", "role": "tgt", "relation": "summarizes_document", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows2], documents=[json.dumps(r) for r in rows2], metadatas=rows2)

    return e, doc_id


def test_nodes_edges_in_doc(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    nodes = gq.nodes_in_doc(doc_id)
    edges = gq.edges_in_doc(doc_id)
    assert {n.id for n in nodes} >= {"A", "B", "C", "S"}
    assert {e.relation for e in edges} >= {"causes", "summarizes_document"}


def test_neighbors_and_khop(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    nbrs = gq.neighbors("A", doc_id=doc_id)
    assert "E1" in nbrs["edges"]
    assert "B" in nbrs["nodes"]
    layers = gq.k_hop(["A"], k=2, doc_id=doc_id)
    assert layers and "B" in layers[1]["nodes"]


def test_shortest_path_and_find_edges(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    path = gq.shortest_path("A", "B", doc_id=doc_id)
    # Path should include A -> E1 -> B
    assert path == ["A", "E1", "B"]
    eids = gq.find_edges(relation="causes", src_label_contains="Smok", tgt_label_contains="Lung", doc_id=doc_id)
    assert "E1" in eids


def test_final_summary_helpers(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    sid = gq.final_summary_node_id(doc_id)
    assert sid == "S"
    snode = gq.final_summary_node(doc_id)
    assert snode and snode.id == "S"


def test_document_subgraph(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    sg = gq.document_subgraph(doc_id, center_ids=["A"], hops=2)
    assert "A" in sg["seed_ids"]
    # should pull B via E1
    assert any(n.id == "B" for n in sg["nodes"])

def test_semantic_seed_then_expand_text(small_graph):
    e, doc_id = small_graph
    gq = GraphQuery(e)
    out = gq.semantic_seed_then_expand_text("smok", top_k=5, hops=1)
    assert out["seeds"]  # should find A or related
    assert isinstance(out["layers"], list) and out["layers"]
