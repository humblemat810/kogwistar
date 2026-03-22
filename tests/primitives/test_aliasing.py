import pytest
import uuid

from kogwistar.engine_core.engine import (
    AliasBook,
    GraphKnowledgeEngine,
    base62_to_uuid,
    uuid_to_base62,
)
from kogwistar.engine_core.models import LLMGraphExtraction, Edge, Node
from tests._kg_factories import kg_document, kg_grounding, kg_llm_grounding_payload


@pytest.mark.ci_full
def test_base62_roundtrip_multiple_ids():
    for _ in range(10):
        value = str(uuid.uuid4())
        alias = uuid_to_base62(value)
        assert base62_to_uuid(alias).lower() == value.lower()
        assert len(alias) < len(value)


@pytest.mark.ci_full
def test_alias_book_is_stable_and_delta_minimal():
    book = AliasBook()

    node_ids = [str(uuid.uuid4()) for _ in range(3)]
    edge_ids = [str(uuid.uuid4()) for _ in range(2)]

    new_nodes, new_edges = book.legend_delta(node_ids, edge_ids)
    assert len(new_nodes) == 3
    assert len(new_edges) == 2

    new_nodes2, new_edges2 = book.legend_delta(node_ids, edge_ids)
    assert new_nodes2 == []
    assert new_edges2 == []

    a1_node0 = book.real_to_alias[node_ids[0]]
    a1_edge0 = book.real_to_alias[edge_ids[0]]

    _ = book.legend_delta([node_ids[0]], [edge_ids[0]])
    a2_node0 = book.real_to_alias[node_ids[0]]
    a2_edge0 = book.real_to_alias[edge_ids[0]]

    assert a1_node0 == a2_node0
    assert a1_edge0 == a2_edge0
    assert a1_node0.startswith("N")
    assert a1_edge0.startswith("E")


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_de_alias_ids_in_result_session_alias(monkeypatch, engine):
    _ = monkeypatch
    eng = engine
    doc = kg_document(
        doc_id="doc::test_de_alias_ids_in_result_session_alias",
        content="x",
        source="test_de_alias_ids_in_result_session_alias",
        doc_type="test",
    )
    eng.write.add_document(doc)

    n1 = Node(
        label="A",
        type="entity",
        summary="a",
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    n2 = Node(
        label="B",
        type="entity",
        summary="b",
        mentions=[kg_grounding(doc.id, start_char=1, end_char=2, excerpt="x")],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    eng.write.add_node(n1, doc_id=doc.id)
    eng.write.add_node(n2, doc_id=doc.id)

    e = Edge(
        label="A->B",
        type="relationship",
        summary="ab",
        relation="rel",
        source_ids=[n1.id],
        target_ids=[n2.id],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[kg_grounding(doc.id)],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
    )
    eng.write.add_edge(e, doc_id=doc.id)

    book = eng._alias_book(doc.id)
    book.assign_for_sets([n1.id, n2.id], [e.id])

    parsed = LLMGraphExtraction["llm"].model_validate(
        {
            "nodes": [
                {
                    "id": book.real_to_alias[n1.id],
                    "label": "A",
                    "type": "entity",
                    "summary": "a",
                    "mentions": [kg_llm_grounding_payload(doc.id)],
                }
            ],
            "edges": [
                {
                    "id": book.real_to_alias[e.id],
                    "label": "A->B",
                    "type": "relationship",
                    "summary": "ab",
                    "relation": "rel",
                    "source_ids": [book.real_to_alias[n1.id]],
                    "target_ids": [book.real_to_alias[n2.id]],
                    "source_edge_ids": [],
                    "target_edge_ids": [],
                    "mentions": [kg_llm_grounding_payload(doc.id)],
                }
            ],
        },
        context={"insertion_method": "pytest-graph_extractor"},
    )
    parsed = LLMGraphExtraction.FromLLMSlice(
        parsed, insertion_method="pytest-graph_extractor"
    )

    out = eng._de_alias_ids_in_result(doc.id, parsed)
    assert out.nodes[0].id == n1.id
    assert out.edges[0].id == e.id
    assert out.edges[0].source_ids == [n1.id]
    assert out.edges[0].target_ids == [n2.id]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_add_page_calls_common_ingest_with_auto_adjudicate(monkeypatch, engine):
    eng = engine
    doc = kg_document(
        doc_id="doc::test_add_page_calls_common_ingest_with_auto_adjudicate",
        content="ignored here",
        source="test_add_page_calls_common_ingest_with_auto_adjudicate",
        doc_type="test",
    )
    eng.write.add_document(doc)

    called = {"args": None, "kwargs": None}

    def fake_ingest(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return {"nodes_added": 1, "edges_added": 0, "raw": None}

    monkeypatch.setattr(eng, "_ingest_text_with_llm", fake_ingest)

    res = eng.add_page(
        document_id=doc.id,
        page_text="Page 1 content",
        page_number=1,
        auto_adjudicate=True,
    )
    assert res["nodes_added"] == 1
    assert called["kwargs"]["doc_id"] == doc.id
    assert called["kwargs"]["content"] == "Page 1 content"
    assert called["kwargs"]["auto_adjudicate"] is True
