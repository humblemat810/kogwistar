import json

import pytest

from graph_knowledge_engine.engine_core.models import Edge, Node
from graph_knowledge_engine.graph_query import GraphQuery
from tests._kg_factories import kg_grounding


class FakeCollection:
    def __init__(self):
        self._docs = {}
        self._metas = {}

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
        if include is None or "documents" in include:
            out["documents"] = docs
        if include is None or "metadatas" in include:
            out["metadatas"] = metas
        return out

    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        if query_texts:
            text = (query_texts[0] or "").lower()
            ranked = [rid for rid, doc in self._docs.items() if isinstance(doc, str) and text in doc.lower()]
        else:
            ranked = list(self._docs.keys())
        return {"ids": [ranked[: max(1, int(n_results))]]}


def _match_where(meta, where):
    if not where:
        return True
    if "$and" in where:
        return all(_match_where(meta, item) for item in where["$and"])
    for key, value in where.items():
        if isinstance(value, dict) and "$in" in value:
            if meta.get(key) not in set(value["$in"]):
                return False
        elif meta.get(key) != value:
            return False
    return True


class _BackendShim:
    def __init__(self, engine):
        self._e = engine

    def node_get(self, **kwargs):
        return self._e.node_collection.get(**kwargs)

    def edge_get(self, **kwargs):
        return self._e.edge_collection.get(**kwargs)

    def edge_endpoints_get(self, **kwargs):
        return self._e.edge_endpoints_collection.get(**kwargs)

    def node_query(self, *, query_texts=None, query_embeddings=None, n_results, include=None, where=None, limit=None, offset=None):
        _ = include, where, limit, offset
        result = self._e.node_collection.query(query_texts=query_texts, query_embeddings=query_embeddings, n_results=n_results)
        return {
            "ids": result["ids"],
            "documents": [[self._e.node_collection._docs.get(rid) for rid in row] for row in result["ids"]],
        }

    def edge_query(self, *, query_texts=None, query_embeddings=None, n_results, include=None, where=None, limit=None, offset=None):
        _ = include, where, limit, offset
        result = self._e.edge_collection.query(query_texts=query_texts, query_embeddings=query_embeddings, n_results=n_results)
        return {
            "ids": result["ids"],
            "documents": [[self._e.edge_collection._docs.get(rid) for rid in row] for row in result["ids"]],
        }


class _ReadShim:
    def __init__(self, engine):
        self._e = engine

    def node_ids_by_doc(self, doc_id: str):
        got = self._e.node_docs_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return [meta["node_id"] for meta in got.get("metadatas") or [] if meta]

    def edge_ids_by_doc(self, doc_id: str):
        got = self._e.edge_endpoints_collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        return sorted({meta["edge_id"] for meta in got.get("metadatas") or [] if meta})

    def get_nodes(self, ids):
        got = self._e.node_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Node.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]

    def get_edges(self, ids):
        got = self._e.edge_collection.get(ids=list(ids), include=["documents"]) if ids else {"documents": []}
        return [Edge.model_validate_json(doc) for doc in (got.get("documents") or []) if doc]


class FakeEngine:
    def __init__(self):
        self.node_collection = FakeCollection()
        self.edge_collection = FakeCollection()
        self.edge_endpoints_collection = FakeCollection()
        self.node_docs_collection = FakeCollection()
        self.backend = _BackendShim(self)
        self.read = _ReadShim(self)


@pytest.fixture()
def small_graph():
    engine = FakeEngine()
    doc_id = "D1"

    def add_node(node_id, label):
        node = Node(
            id=node_id,
            label=label,
            type="entity",
            summary=label,
            mentions=[kg_grounding(doc_id, collection_page_url=f"document_collection/{doc_id}")],
            doc_id=doc_id,
        )
        engine.node_collection.add(
            ids=[node_id],
            documents=[node.model_dump_json(field_mode="backend")],
            metadatas=[{"doc_id": doc_id, "label": node.label, "type": node.type}],
        )
        row_id = f"{node_id}::{doc_id}"
        row = {"id": row_id, "node_id": node_id, "doc_id": doc_id}
        engine.node_docs_collection.add(ids=[row_id], documents=[json.dumps(row)], metadatas=[row])
        return node

    a_node = add_node("A", "Smoking")
    add_node("B", "Lung Cancer")
    add_node("C", "Cough")

    edge_id = "E1"
    edge = Edge(
        id=edge_id,
        label="Smoking causes Lung Cancer",
        type="relationship",
        summary="causal",
        relation="causes",
        source_ids=["A"],
        target_ids=["B"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=a_node.mentions,
        doc_id=doc_id,
    )
    engine.edge_collection.add(
        ids=[edge_id],
        documents=[edge.model_dump_json(field_mode="backend")],
        metadatas=[{"doc_id": doc_id, "relation": "causes"}],
    )
    rows = [
        {"id": f"{edge_id}::src::node::A", "edge_id": edge_id, "endpoint_id": "A", "endpoint_type": "node", "role": "src", "relation": "causes", "doc_id": doc_id},
        {"id": f"{edge_id}::tgt::node::B", "edge_id": edge_id, "endpoint_id": "B", "endpoint_type": "node", "role": "tgt", "relation": "causes", "doc_id": doc_id},
    ]
    engine.edge_endpoints_collection.add(ids=[row["id"] for row in rows], documents=[json.dumps(row) for row in rows], metadatas=rows)

    summary_node = add_node("S", "Final Summary")
    edge_id2 = "E2"
    edge2 = Edge(
        id=edge_id2,
        label="summarizes_document",
        type="relationship",
        summary="S summarizes document",
        relation="summarizes_document",
        source_ids=["S"],
        target_ids=[f"docnode:{doc_id}"],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=summary_node.mentions,
        doc_id=doc_id,
    )
    engine.edge_collection.add(
        ids=[edge_id2],
        documents=[edge2.model_dump_json(field_mode="backend")],
        metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}],
    )
    rows2 = [
        {"id": f"{edge_id2}::src::node::S", "edge_id": edge_id2, "endpoint_id": "S", "endpoint_type": "node", "role": "src", "relation": "summarizes_document", "doc_id": doc_id},
        {"id": f"{edge_id2}::tgt::node::docnode:{doc_id}", "edge_id": edge_id2, "endpoint_id": f"docnode:{doc_id}", "endpoint_type": "node", "role": "tgt", "relation": "summarizes_document", "doc_id": doc_id},
    ]
    engine.edge_endpoints_collection.add(ids=[row["id"] for row in rows2], documents=[json.dumps(row) for row in rows2], metadatas=rows2)

    return engine, doc_id


def test_nodes_edges_in_doc(small_graph):
    engine, doc_id = small_graph
    gq = GraphQuery(engine)
    nodes = gq.nodes_in_doc(doc_id)
    edges = gq.edges_in_doc(doc_id)
    assert {node.id for node in nodes} >= {"A", "B", "C", "S"}
    assert {edge.relation for edge in edges} >= {"causes", "summarizes_document"}


def test_neighbors_and_khop(small_graph):
    engine, doc_id = small_graph
    gq = GraphQuery(engine)
    nbrs = gq.neighbors("A", doc_id=doc_id)
    assert "E1" in nbrs["edges"]
    assert "B" in nbrs["nodes"]
    layers = gq.k_hop(["A"], k=2, doc_id=doc_id)
    assert layers and "B" in layers[1]["nodes"]


def test_shortest_path_and_find_edges(small_graph):
    engine, doc_id = small_graph
    gq = GraphQuery(engine)
    path = gq.shortest_path("A", "B", doc_id=doc_id)
    assert path == ["A", "E1", "B"]
    edge_ids = gq.find_edges(relation="causes", src_label_contains="Smok", tgt_label_contains="Lung", doc_id=doc_id)
    assert "E1" in edge_ids


def test_final_summary_helpers(small_graph):
    engine, doc_id = small_graph
    gq = GraphQuery(engine)
    summary_id = gq.final_summary_node_id(doc_id)
    assert summary_id == "S"
    summary_node = gq.final_summary_node(doc_id)
    assert summary_node and summary_node.id == "S"


def test_document_subgraph(small_graph):
    engine, doc_id = small_graph
    gq = GraphQuery(engine)
    subgraph = gq.document_subgraph(doc_id, center_ids=["A"], hops=2)
    assert "A" in subgraph["seed_ids"]
    assert any(node.id == "B" for node in subgraph["nodes"])


def test_semantic_seed_then_expand_text(small_graph):
    engine, _ = small_graph
    gq = GraphQuery(engine)
    out = gq.semantic_seed_then_expand_text("smok", top_k=5, hops=1)
    assert out["seeds"]
    assert isinstance(out["layers"], list) and out["layers"]
