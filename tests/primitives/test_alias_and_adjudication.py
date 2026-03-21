import uuid

from kogwistar.engine_core.engine import (
    AliasBook,
    GraphKnowledgeEngine,
    base62_to_uuid,
    uuid_to_base62,
)
from kogwistar.engine_core.models import (
    AdjudicationVerdict,
    Document,
    Edge,
    Grounding,
    LLMGraphExtraction,
    MentionVerification,
    Node,
    Span,
)


def _doc(*, doc_id: str, content: str, source: str) -> Document:
    return Document(
        id=doc_id,
        content=content,
        type="text",
        metadata={"source": source},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )


def _span_for(
    doc_id: str,
    start_page: int = 1,
    end_page: int = 1,
    start_char: int = 0,
    end_char: int = 1,
    *,
    collection_page_url: str = "c",
    document_page_url: str | None = None,
) -> Span:
    _ = end_page
    if document_page_url is None:
        document_page_url = f"document/{doc_id}"
    return Span(
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        doc_id=doc_id,
        insertion_method="pytest-manual",
        page_number=start_page,
        start_char=start_char,
        end_char=end_char,
        excerpt=f"{start_char}:{end_char}",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="heuristic",
            is_verified=False,
            notes=None,
            score=0.9,
        ),
    )


def _grounding_for(doc_id: str, **span_kwargs) -> Grounding:
    return Grounding(spans=[_span_for(doc_id, **span_kwargs)])


def _llm_grounding_payload(doc_id: str, **span_kwargs) -> dict:
    span = _span_for(doc_id, **span_kwargs)
    return {
        "spans": [
            {
                "collection_page_url": span.collection_page_url,
                "document_page_url": span.document_page_url,
                "doc_id": span.doc_id,
                "page_number": span.page_number,
                "start_char": span.start_char,
                "end_char": span.end_char,
                "excerpt": span.excerpt,
                "context_before": span.context_before,
                "context_after": span.context_after,
                "chunk_id": span.chunk_id,
                "source_cluster_id": span.source_cluster_id,
            }
        ]
    }


def test_base62_roundtrip():
    u = str(uuid.uuid4())
    s = uuid_to_base62(u)
    back = base62_to_uuid(s)
    assert back.lower() == u.lower()


def test_alias_book_stability_and_delta():
    book = AliasBook()
    nodes = [f"{uuid.uuid4()}" for _ in range(3)]
    edges = [f"{uuid.uuid4()}" for _ in range(2)]

    new_nodes, new_edges = book.legend_delta(nodes, edges)
    assert len(new_nodes) == 3 and len(new_edges) == 2

    new_nodes2, new_edges2 = book.legend_delta(nodes, edges)
    assert new_nodes2 == [] and new_edges2 == []

    a1 = [book.real_to_alias[nodes[0]], book.real_to_alias[edges[0]]]
    book.legend_delta([nodes[0]], [edges[0]])
    a2 = [book.real_to_alias[nodes[0]], book.real_to_alias[edges[0]]]
    assert a1 == a2


def test_de_alias_ids_in_result_session_alias(monkeypatch):
    _ = monkeypatch
    eng = GraphKnowledgeEngine()
    doc = _doc(
        doc_id="doc::test_de_alias_ids_in_result_session_alias",
        content="x",
        source="test_de_alias_ids_in_result_session_alias",
    )
    eng.write.add_document(doc)

    n1 = Node(
        label="A",
        type="entity",
        summary="a",
        metadata={"source": "test_de_alias_ids_in_result_session_alias"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        mentions=[_grounding_for(doc.id)],
    )
    n2 = Node(
        label="B",
        type="entity",
        summary="b",
        metadata={},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        mentions=[_grounding_for(doc.id, start_char=2, end_char=3)],
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
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        metadata={},
        mentions=[_grounding_for(doc.id, start_char=0, end_char=3)],
    )
    eng.write.add_edge(e, doc_id=doc.id)

    book = eng._alias_book(doc.id)
    book.assign_for_sets([n1.id, n2.id], [e.id])

    dumped = {
        "nodes": [
            {
                "id": book.real_to_alias[n1.id],
                "label": "A",
                "type": "entity",
                "summary": "a",
                "domain_id": None,
                "canonical_entity_id": None,
                "properties": None,
                "local_id": "nn:1",
                "mentions": [_llm_grounding_payload(doc.id)],
            }
        ],
        "edges": [
            {
                "id": book.real_to_alias[e.id],
                "label": "A->B",
                "type": "relationship",
                "summary": "ab",
                "domain_id": None,
                "canonical_entity_id": None,
                "properties": None,
                "local_id": "ne:1",
                "relation": "rel",
                "source_ids": [book.real_to_alias[n1.id]],
                "target_ids": [book.real_to_alias[n2.id]],
                "source_edge_ids": [],
                "target_edge_ids": [],
                "mentions": [_llm_grounding_payload(doc.id, start_char=2, end_char=3)],
            }
        ],
    }
    parsed = LLMGraphExtraction["llm"].model_validate(
        dumped,
        context={"insertion_method": "pytest-graph_extractor"},
    )
    parsed = LLMGraphExtraction.FromLLMSlice(
        parsed,
        insertion_method="pytest-graph_extractor",
    )
    out = eng._de_alias_ids_in_result(doc.id, parsed)
    assert out.nodes[0].id == n1.id
    assert out.edges[0].id == e.id
    assert out.edges[0].source_ids == [n1.id]
    assert out.edges[0].target_ids == [n2.id]


def test_commit_merge_creates_same_as_and_endpoints():
    eng = GraphKnowledgeEngine()
    doc = _doc(
        doc_id="doc::test_commit_merge_creates_same_as_and_endpoints",
        content="y",
        source="test_commit_merge_creates_same_as_and_endpoints",
    )
    eng.write.add_document(doc)
    mentions = [_grounding_for(doc.id, end_char=5, collection_page_url="c")]

    a = Node(
        label="Einstein",
        type="entity",
        summary="person",
        mentions=mentions,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        metadata={},
    )
    b = Node(
        label="Einstein",
        type="entity",
        summary="person (dup)",
        mentions=mentions,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        metadata={},
    )
    eng.write.add_node(a, doc_id=doc.id)
    eng.write.add_node(b, doc_id=doc.id)

    verdict = AdjudicationVerdict.model_validate(
        {
            "same_entity": True,
            "confidence": 0.9,
            "reason": "dup",
            "canonical_entity_id": None,
        }
    )

    eng.commit_merge(a, b, verdict, method="pytest_commit_merge")

    edges = eng.backend.edge_get(include=["metadatas", "documents"])
    found = False
    for meta in edges["metadatas"]:
        if (meta or {}).get("relation") == "same_as":
            found = True
            break
    assert found, "same_as edge not created by commit_merge"
