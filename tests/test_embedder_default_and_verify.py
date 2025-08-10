import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, ReferenceSession

def _ref(doc_id, start=0, end=20):
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        start_page=1, end_page=1, start_char=start, end_char=end
    )

def test_default_sentence_transformer_embedder(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

    # Add a doc
    text = "Chlorophyll is a pigment that absorbs light in plants."
    doc = Document(content=text, type="text")
    eng.add_document(doc)

    # Add a node with no explicit embeddings -> collection embedder should run
    n = Node(label="Chlorophyll", type="entity", summary="pigment that absorbs light", references=[_ref(doc.id, 0, 40)])
    eng.add_node(n, doc_id=doc.id)

    got = eng.node_collection.get(ids=[n.id], include=["embeddings", "documents"])
    assert got["embeddings"] and len(got["embeddings"][0]) > 0  # auto-embedded
    # Verify mention (embedding similarity included)
    out = eng.verify_mentions_for_doc(doc.id, min_ngram=4, threshold=0.3)
    assert out["updated_nodes"] >= 1
    n2 = eng.node_collection.get(ids=[n.id], include=["documents"])
    node = Node.model_validate_json(n2["documents"][0])
    assert node.references[0].verification is not None
    detail = json.loads(node.references[0].verification.notes)
    # embedding score may be present if model ran
    assert "coverage" in detail