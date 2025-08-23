from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Edge, ReferenceSession, MentionVerification
def test_node_refs_indexing(engine):
    doc = Document(content="x", type="text"); engine.add_document(doc)
    ref_llm = ReferenceSession(
        collection_page_url="c", document_page_url=f"document/{doc.id}",
        insertion_method="pytest-llm",
        start_page=1, end_page=1, start_char=0, end_char=5, doc_id=doc.id,
        verification=MentionVerification(method="llm", is_verified=True, score=0.9),
    )
    ref_manual = ReferenceSession(
        collection_page_url="c", document_page_url=f"document/{doc.id}",
        insertion_method="pytest-manual",
        start_page=2, end_page=2, start_char=0, end_char=5, doc_id=doc.id,
        verification=MentionVerification(method="human", is_verified=True, score=1.0),
    )
    n = Node(label="A", type="entity", summary="s", references=[ref_llm, ref_manual])
    engine.add_node(n, doc_id=doc.id)
    engine._index_node_refs(n)  # if you didn’t wire _maybe_ in add_node yet

    assert n.id in engine.nodes_by_doc(doc.id, where = dict(insertion_method="pytest-llm"))
    assert n.id in engine.nodes_by_doc(doc.id, where = dict(insertion_method="pytest-manual"))
    
def test_edge_refs_indexing(engine):
    doc = Document(content="x", type="text"); engine.add_document(doc)
    ref_llm = ReferenceSession(
        collection_page_url="c", document_page_url=f"document/{doc.id}",
        start_page=1, end_page=1, start_char=0, end_char=5, doc_id=doc.id,
        insertion_method="pytest-llm",
        verification=MentionVerification(method="llm", is_verified=True, score=0.9),
    )
    ref_manual = ReferenceSession(
        collection_page_url="c", document_page_url=f"document/{doc.id}",
        start_page=2, end_page=2, start_char=0, end_char=5, doc_id=doc.id,
        insertion_method="pytest-manual",
        verification=MentionVerification(method="human", is_verified=True, score=1.0),
    )
    # two nodes to connect
    a = Node(label="A", type="entity", summary="s", references=[ref_llm]); engine.add_node(a, doc_id=doc.id)
    b = Node(label="B", type="entity", summary="s", references=[ref_manual]); engine.add_node(b, doc_id=doc.id)

    e = Edge(
        label="A->B", type="relationship", summary="rel", relation="related",
        source_ids=[a.id], target_ids=[b.id], source_edge_ids=[], target_edge_ids=[],
        references=[ref_llm, ref_manual],
    )
    engine.add_edge(e, doc_id=doc.id)
    engine._index_edge_refs(e)  # if not yet wired to add_edge

    assert e.id in engine.edges_by_doc(doc.id, where = dict(insertion_method="pytest-llm"))
    assert e.id in engine.edges_by_doc(doc.id, where = dict(insertion_method="pytest-manual"))