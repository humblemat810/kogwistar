import uuid
import json
import pytest

from graph_knowledge_engine.engine import (
    GraphKnowledgeEngine,
    AliasBook,
    uuid_to_base62,
    base62_to_uuid,
)

from graph_knowledge_engine.models import (
    LLMGraphExtraction,
    LLMNode,
    LLMEdge,
    Span,
    Document,
    Node,
    Edge,
    MentionVerification,
)

# --------------------------
# Base62 helpers
# --------------------------

def test_base62_roundtrip_multiple_ids():
    for _ in range(10):
        u = str(uuid.uuid4())
        s = uuid_to_base62(u)
        back = base62_to_uuid(s)
        assert back.lower() == u.lower()
        assert len(s) < len(u)  # compressed compared to 36-char uuid


# --------------------------
# Session alias book
# --------------------------

def test_alias_book_is_stable_and_delta_minimal():
    book = AliasBook()

    node_ids = [str(uuid.uuid4()) for _ in range(3)]
    edge_ids = [str(uuid.uuid4()) for _ in range(2)]

    # First call assigns aliases, delta includes all
    new_nodes, new_edges = book.legend_delta(node_ids, edge_ids)
    assert len(new_nodes) == 3
    assert len(new_edges) == 2

    # Second call with same sets -> delta is empty (cache-friendly)
    new_nodes2, new_edges2 = book.legend_delta(node_ids, edge_ids)
    assert new_nodes2 == []
    assert new_edges2 == []

    # Aliases are stable
    a1_node0 = book.real_to_alias[node_ids[0]]
    a1_edge0 = book.real_to_alias[edge_ids[0]]

    # Ask again (no reassign) -> same aliases
    _ = book.legend_delta([node_ids[0]], [edge_ids[0]])
    a2_node0 = book.real_to_alias[node_ids[0]]
    a2_edge0 = book.real_to_alias[edge_ids[0]]

    assert a1_node0 == a2_node0
    assert a1_edge0 == a2_edge0
    assert a1_node0.startswith("N")
    assert a1_edge0.startswith("E")


# --------------------------
# De-aliasing back to real IDs (session_alias strategy)
# --------------------------

def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        start_page=1, end_page=1, start_char=0, end_char=1,
        verification=MentionVerification(method="heuristic", is_verified=False), 
        insertion_method="pytest-manual",
        doc_id = doc_id
        
    )

def test_de_alias_ids_in_result_session_alias(monkeypatch):
    eng = GraphKnowledgeEngine()
    doc = Document(content="x", type="test")
    eng.add_document(doc)

    # create tiny context: two nodes + an edge
    n1 = Node(label="A", type="entity", summary="a", mentions=[_span_for(doc.id)])
    n2 = Node(label="B", type="entity", summary="b", mentions=[_span_for(doc.id)])
    eng.add_node(n1, doc_id=doc.id)
    eng.add_node(n2, doc_id=doc.id)
    e = Edge(label="A->B", type="relationship", summary="ab", relation="rel",
             source_ids=[n1.id], target_ids=[n2.id], source_edge_ids = [], target_edge_ids = [],
             mentions=[_span_for(doc.id)])
    eng.add_edge(e, doc_id=doc.id)

    # allocate session aliases for the context
    book = eng._alias_book(doc.id)
    book.assign_for_sets([n1.id, n2.id], [e.id])

    # simulate an LLM structured output USING ALIASES
    parsed = LLMGraphExtraction(
        nodes=[
            LLMNode(
                id=book.real_to_alias[n1.id],
                label="A", type="entity", summary="a",
                mentions=[_span_for(doc.id)]
            )
        ],
        edges=[
            LLMEdge(
                id=book.real_to_alias[e.id],
                label="A->B", type="relationship", summary="ab",
                relation="rel",
                source_ids=[book.real_to_alias[n1.id]],
                target_ids=[book.real_to_alias[n2.id]],
                source_edge_ids = [], target_edge_ids = [],
                mentions=[_span_for(doc.id)]
            )
        ]
    )

    # de-alias back to real UUIDs
    out = eng._de_alias_ids_in_result(doc.id, parsed)
    assert out.nodes[0].id == n1.id
    assert out.edges[0].id == e.id
    assert out.edges[0].source_ids == [n1.id]
    assert out.edges[0].target_ids == [n2.id]


# --------------------------
# add_page wiring sanity (auto_adjudicate flag)
# --------------------------

def test_add_page_calls_common_ingest_with_auto_adjudicate(monkeypatch):
    eng = GraphKnowledgeEngine()
    doc = Document(content="ignored here", type="test")
    eng.add_document(doc)

    called = {"args": None, "kwargs": None}

    def fake_ingest(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        # simulate adding something
        return {"nodes_added": 1, "edges_added": 0, "raw": None}

    # Patch the shared path to avoid LLM calls and to capture args
    monkeypatch.setattr(eng, "_ingest_text_with_llm", fake_ingest)

    res = eng.add_page(document_id=doc.id, page_text="Page 1 content", page_number=1, auto_adjudicate=True)
    assert res["nodes_added"] == 1

    # Ensure our shared path was called with the expected flags
    assert called["kwargs"]["doc_id"] == doc.id
    assert called["kwargs"]["content"] == "Page 1 content"
    assert called["kwargs"]["auto_adjudicate"] is True