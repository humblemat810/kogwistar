# tests/test_node_docs_rollback.py
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Span

def _ref_for(did: str, s=0, e=1):
    return Span(
        collection_page_url=f"document_collection/{did}",
        document_page_url=f"document/{did}",
        insertion_method="pytest-manual",
        start_page=1, end_page=1, start_char=s, end_char=e, doc_id=did
    )

def test_node_docs_partial_then_full_rollback(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))
    d1 = Document(content="D1", type="text"); eng.add_document(d1)
    d2 = Document(content="D2", type="text"); eng.add_document(d2)

    # One node with evidence in *both* documents (no single doc_id in node meta)
    n = Node(label="Shared", type="entity", summary="x",
             mentions=[_ref_for(d1.id), _ref_for(d2.id)])
    eng.add_node(n)  # no doc_id passed; relies on references + node_docs

    # Sanity: node_docs has two rows
    rows = eng.node_docs_collection.get(where={"node_id": n.id}, include=["metadatas"])
    assert {m["doc_id"] for m in rows.get("metadatas") or []} == {d1.id, d2.id}

    # Rollback d1 only: node remains, but loses d1 reference; node_docs loses (n,d1)
    res1 = eng.rollback_document(d1.id)
    assert isinstance(res1, dict)
    rows_after = eng.node_docs_collection.get(where={"node_id": n.id}, include=["metadatas"])
    assert {m["doc_id"] for m in rows_after.get("metadatas") or []} == {d2.id}

    n_got = eng.node_collection.get(ids=[n.id], include=["documents"])
    node_json = n_got["documents"][0]
    node = Node.model_validate_json(node_json)
    assert all(getattr(r, "doc_id", None) != d1.id for r in node.mentions or [])

    # Rollback d2: now the node has no references and is deleted
    res2 = eng.rollback_document(d2.id)
    n_gone = eng.node_collection.get(ids=[n.id])
    assert not (n_gone.get("ids") and n_gone["ids"][0] == n.id)