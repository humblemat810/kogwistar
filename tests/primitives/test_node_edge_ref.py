from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Document,
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)


def _mk_span(doc_id: str, method: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        insertion_method=method,
        page_number=1,
        start_char=0,
        end_char=1,
        doc_id=doc_id,
        excerpt="x",
        context_before="",
        context_after="",
        verification=MentionVerification(
            method="human" if method == "pytest-manual" else "llm",
            is_verified=True,
            score=1.0,
            notes=None,
        ),
        source_cluster_id=None,
        chunk_id=None,
    )


def test_node_refs_indexing(engine: GraphKnowledgeEngine):
    doc = Document(
        id="doc-node-refs",
        content="x",
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    engine.add_document(doc)

    ref_llm = _mk_span(doc.id, "pytest-llm")
    ref_manual = _mk_span(doc.id, "pytest-manual")
    n = Node(
        label="A",
        type="entity",
        summary="s",
        mentions=[Grounding(spans=[ref_llm, ref_manual])],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        metadata={},
    )
    engine.add_node(n, doc_id=doc.id)
    engine._index_node_refs(n)

    assert n.id in engine.nodes_by_doc(doc.id, where={"insertion_method": "pytest-llm"})
    assert n.id in engine.nodes_by_doc(
        doc.id, where={"insertion_method": "pytest-manual"}
    )


def iter_span(n_or_e: Node | Edge):
    for g in n_or_e.mentions:
        for sp in g.spans:
            yield sp


def test_edge_refs_indexing(engine):
    doc = Document(
        id="doc-edge-refs",
        content="x",
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    engine.add_document(doc)
    ref_llm = _mk_span(doc.id, "pytest-llm")
    ref_manual = _mk_span(doc.id, "pytest-manual")

    a = Node(
        label="A",
        type="entity",
        summary="s",
        mentions=[Grounding(spans=[ref_llm])],
        metadata={},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )
    engine.add_node(a, doc_id=doc.id)
    b = Node(
        label="B",
        type="entity",
        summary="s",
        mentions=[Grounding(spans=[ref_manual])],
        metadata={},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )
    engine.add_node(b, doc_id=doc.id)

    e = Edge(
        label="A->B",
        type="relationship",
        summary="rel",
        relation="related",
        source_ids=[a.id],
        target_ids=[b.id],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[Grounding(spans=[ref_llm, ref_manual])],
        metadata={},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=doc.id,
    )
    engine.add_edge(e, doc_id=doc.id)
    engine._index_edge_refs(e)
    doc_eids = engine.edges_by_doc(doc.id)
    assert e.id in doc_eids  # , where={"insertion_method": "pytest-llm"}
    assert list(
        [
            sp.insertion_method
            for e in engine.get_edges()
            for sp in e.iter_span()
            if sp.insertion_method == "pytest-llm"
        ]
    )
    assert e.id in engine.edges_by_doc(
        doc.id
    )  # , where={"insertion_method": "pytest-manual"}
    assert list(
        [
            sp.insertion_method
            for e in engine.get_edges()
            for sp in e.iter_span()
            if sp.insertion_method == "pytest-manual"
        ]
    )
