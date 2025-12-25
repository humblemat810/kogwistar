import json
import uuid
import shutil
import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Node, Edge, Document

@pytest.fixture(scope="function")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db_ep")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture(scope="function")
def engine(tmp_chroma_dir):
    # Use a fresh engine with the edge_endpoints collection enabled
    eng = GraphKnowledgeEngine(persist_directory=tmp_chroma_dir)
    # Sanity: the helper collection exists
    assert getattr(eng, "edge_endpoints_collection", None) is not None
    return eng

def _count_ids(get_result):
    return 0 if not get_result or "ids" not in get_result else len(get_result["ids"] or [])
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Edge, Span

def test_rollback_single_document():
    engine = GraphKnowledgeEngine()

    # Create a document
    doc = Document(content="Test doc", type="text")
    engine.add_document(doc)

    # Create two nodes belonging to the doc
    ref1 = Span(collection_page_url="c", document_page_url=f"document/{doc.id}", start_page=13,end_page=14,start_char=24,end_char=17,
                            doc_id = doc.id, insertion_method = "pytest-manual")
    ref2 = Span(collection_page_url="c", document_page_url=f"document/{doc.id}", start_page=15,end_page=15,start_char=4,end_char=45,
                            doc_id = doc.id, insertion_method = "pytest-manual")
    node1 = Node(label="A", type="entity", summary="n1", mentions=[ref1], doc_id = doc.id)
    node2 = Node(label="B", type="entity", summary="n2", mentions=[ref2], doc_id = doc.id)
    engine.add_node(node1)
    engine.add_node(node2)

    # Create an edge between them
    
    edge = Edge(label="A->B", type="relationship", summary="edge",
                source_ids=[node1.id], target_ids=[node2.id], source_edge_ids= [], target_edge_ids=[], relation="related", 
                mentions=[ref1, ref2])
    engine.add_edge(edge)

    # Rollback document
    result = engine.rollback_document(doc.id)

    assert result["deleted_nodes"] == 2
    assert result["deleted_edges"] == 1
    assert result["updated_edges"] == 0

    # Ensure no nodes remain
    remaining_nodes = engine.node_collection.get()
    assert node1.id not in remaining_nodes["ids"]
    assert node2.id not in remaining_nodes["ids"]

def test_rollback_multiple_documents(tmp_path):
    engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

    docs = [Document(content=f"Doc {i}", type="text") for i in range(2)]
    for d in docs:
        engine.add_document(d)

    ref1 = Span(collection_page_url="c", document_page_url=f"document/{docs[0].id}",
                            start_page=1, end_page=1, start_char=0, end_char=1, doc_id=docs[0].id, insertion_method = "pytest-manual")
    ref2 = Span(collection_page_url="c", document_page_url=f"document/{docs[1].id}",
                            start_page=1, end_page=1, start_char=0, end_char=1, doc_id=docs[1].id, insertion_method = "pytest-manual")

    shared_node = Node(label="X", type="entity", summary="shared", mentions=[ref1, ref2])
    engine.add_node(shared_node)

    n1 = Node(label="Y", type="entity", summary="n1", mentions=[ref1])
    engine.add_node(n1)

    n2 = Node(label="Z", type="entity", summary="n2", mentions=[ref2])
    engine.add_node(n2)

    # Proper endpoints + pass doc_id so endpoints rows get doc_id
    e1 = Edge(label="shared->n1", type="relationship", summary="e1",
              source_ids=[shared_node.id], target_ids=[n1.id], source_edge_ids=[], target_edge_ids = [], 
              relation="related", mentions=[ref1])
    e2 = Edge(label="n2->shared", type="relationship", summary="e2",
              source_ids=[n2.id], target_ids=[shared_node.id], source_edge_ids=[], target_edge_ids = [], 
              relation="related", mentions=[ref2])
    engine.add_edge(e1, doc_id=docs[0].id)
    engine.add_edge(e2, doc_id=docs[1].id)

    # Rollback both documents
    result = engine.rollback_many_documents([docs[0].id, docs[1].id])

    assert result["deleted_nodes"] >= 3
    assert result["deleted_edges"] >= 2