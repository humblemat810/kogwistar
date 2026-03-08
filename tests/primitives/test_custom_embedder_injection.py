class DummyEF:
    def name(self):
        return "DummyEF"
    def __call__(self, input):
        # 2D toy embedding
        return [[float(len(t)), float(t.count(" "))] for t in input]

def test_custom_embedder(tmp_path):
    # Prove we can inject our own EF into engine and it drives Chroma collections
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
    from tests._kg_factories import kg_document, kg_grounding

    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"), embedding_function=DummyEF())

    # Adding a node with embeddings=None uses DummyEF
    from graph_knowledge_engine.engine_core.models import Node

    doc = kg_document(
        doc_id="doc::test_custom_embedder",
        content="abc def",
        source="test_custom_embedder",
    )
    eng.write.add_document(doc)
    node = Node(label="X", type="entity", summary="abc", mentions=[kg_grounding(doc.id, end_char=3, excerpt="abc")])
    eng.write.add_node(node, doc_id=doc.id)
    got = eng.backend.node_get(ids=[node.id], include=["embeddings", "documents"])
    emb = got["embeddings"][0]
    expected = eng._embed_one(got["documents"][0])
    assert len(emb) == len(expected)
    assert all(abs(a - b) < 1e-9 for a, b in zip(expected, emb))
