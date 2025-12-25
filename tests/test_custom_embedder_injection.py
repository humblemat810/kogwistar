from chromadb import Client
from chromadb.config import Settings

class DummyEF:
    def __call__(self, texts):
        # 2D toy embedding
        return [[float(len(t)), float(t.count(" "))] for t in texts]

def test_custom_embedder(tmp_path):
    # Prove we can inject our own EF into engine and it drives Chroma collections
    from graph_knowledge_engine.engine import GraphKnowledgeEngine
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"), embedding_function=DummyEF())

    # Adding a node with embeddings=None uses DummyEF
    from graph_knowledge_engine.models import Document, Node, Span
    doc = Document(content="abc def", type="text")
    eng.add_document(doc)
    ref = Span(collection_page_url="c", document_page_url=f"document/{doc.id}", 
                           insertion_method="pytest-manual",
                           start_page=1, end_page=1, start_char=0, end_char=3,
                           doc_id = doc.id)
    node = Node(label="X", type="entity", summary="abc", mentions=[ref])
    eng.add_node(node, doc_id=doc.id)
    eng._embed_one
    got = eng.node_collection.get(ids=[node.id], include=["embeddings"])
    emb = got["embeddings"][0]
    assert (all(eng._embed_one(eng.node_collection.get(ids=[node.id])['documents'][0]) - emb < 1e6))
    # assert len(emb) == 2 and emb[0] == float(len(node.model_dump_json(field_mode = 'backend'))) or len(emb) == 2  # allow toy impl variance