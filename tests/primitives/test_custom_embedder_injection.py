import pytest

pytestmark = pytest.mark.core

from tests._helpers.fake_backend import build_fake_backend

class DummyEF:
    def name(self):
        return "DummyEF"

    def __call__(self, input):
        # 2D toy embedding
        return [[float(len(t)), float(t.count(" "))] for t in input]


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
def test_custom_embedder(tmp_path, backend_kind):
    # Prove we can inject our own EF into engine and it drives Chroma collections
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from tests._kg_factories import kg_document, kg_grounding

    kwargs = {
        "persist_directory": str(tmp_path / "chroma"),
        "embedding_function": DummyEF(),
    }
    if backend_kind == "fake":
        kwargs["backend_factory"] = build_fake_backend
    eng = GraphKnowledgeEngine(**kwargs)

    # Adding a node with embeddings=None uses DummyEF
    from kogwistar.engine_core.models import Node

    doc = kg_document(
        doc_id="doc::test_custom_embedder",
        content="abc def",
        source="test_custom_embedder",
    )
    eng.write.add_document(doc)
    node = Node(
        label="X",
        type="entity",
        summary="abc",
        mentions=[kg_grounding(doc.id, end_char=3, excerpt="abc")],
    )
    eng.write.add_node(node, doc_id=doc.id)
    got = eng.backend.node_get(ids=[node.id], include=["embeddings", "documents"])
    emb = got["embeddings"][0]
    expected = eng._embed_one(got["documents"][0])
    assert len(emb) == len(expected)
    assert all(abs(a - b) < 1e-9 for a, b in zip(expected, emb))
