import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Span

def _ref(doc_id, start=0, end=20):
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id = doc_id,
        start_page=1, end_page=1, start_char=start, end_char=end, insertion_method= 'pytest-manual'
    )

def test_default_sentence_transformer_embedder(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

    # Add a doc
    text = "Chlorophyll is a pigment that absorbs light in plants."
    doc = Document(content=text, type="text")
    eng.add_document(doc)

    # Add a node with no explicit embeddings -> collection embedder should run
    n = Node(label="Chlorophyll", type="entity", summary="pigment that absorbs light", mentions=[_ref(doc.id, 0, 40)])
    eng.add_node(n, doc_id=doc.id)

    got = eng.node_collection.get(ids=[n.id], include=["embeddings", "documents"])
    assert got["embeddings"].shape[0] and len(got["embeddings"][0]) > 0  # auto-embedded
    # Verify mention (embedding similarity included)
    out = eng.verify_mentions_for_doc(doc.id, min_ngram=4, threshold=0.3)
    assert out["updated_nodes"] >= 1
    n2 = eng.node_collection.get(ids=[n.id], include=["documents"])
    node = Node.model_validate_json(n2["documents"][0])
    assert node.mentions[0].verification is not None
    detail = json.loads(node.mentions[0].verification.notes)
    # embedding score may be present if model ran
    assert "coverage" in detail
    
def test_default_embedder_autoruns(tmp_path):
    from graph_knowledge_engine.engine import GraphKnowledgeEngine
    from graph_knowledge_engine.models import Document, Node, Span

    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))
    doc = Document(content="Chlorophyll absorbs light.", type="text")
    eng.add_document(doc)

    ref = Span(collection_page_url="c", document_page_url=f"document/{doc.id}", doc_id = doc.id,
                           start_page=1, end_page=1, start_char=0, end_char=10, insertion_method = 'pytest-manual')
    n = Node(label="Chlorophyll", type="entity", summary="absorbs light", mentions=[ref])
    eng.add_node(n, doc_id=doc.id)  # embeddings=None -> auto-embed via DefaultEmbeddingFunction

    got = eng.node_collection.get(ids=[n.id], include=["embeddings"])
    assert got["embeddings"] is not None and len(got["embeddings"][0]) > 0  # vector was created

    # embedding similarity is also available to verifier now
    out = eng.verify_mentions_for_doc(doc.id, min_ngram=4, threshold=0.3)
    assert out["updated_nodes"] >= 1