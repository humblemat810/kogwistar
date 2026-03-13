import json
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Node

from tests._kg_factories import kg_document, kg_grounding


def test_default_sentence_transformer_embedder(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

    # Add a doc
    text = "Chlorophyll is a pigment that absorbs light in plants."
    doc = kg_document(
        doc_id="doc::test_default_sentence_transformer_embedder",
        content=text,
        source="test_default_sentence_transformer_embedder",
    )
    eng.write.add_document(doc)

    # Add a node with no explicit embeddings -> collection embedder should run
    n = Node(
        label="Chlorophyll",
        type="entity",
        summary="pigment that absorbs light",
        mentions=[
            kg_grounding(
                doc.id,
                start_char=0,
                end_char=40,
                excerpt=text[:40],
                context_after=text[40:60],
                collection_page_url=f"document_collection/{doc.id}",
            )
        ],
    )
    eng.write.add_node(n, doc_id=doc.id)

    got = eng.backend.node_get(ids=[n.id], include=["embeddings", "documents"])
    assert got["embeddings"].shape[0] and len(got["embeddings"][0]) > 0  # auto-embedded
    # Verify mention (embedding similarity included)
    out = eng.verify_mentions_for_doc(doc.id, min_ngram=4, threshold=0.3)
    assert out["updated_nodes"] >= 1
    n2 = eng.backend.node_get(ids=[n.id], include=["documents"])
    node = Node.model_validate_json(n2["documents"][0])
    assert node.mentions[0].spans[0].verification is not None
    detail = json.loads(node.mentions[0].spans[0].verification.notes)
    # embedding score may be present if model ran
    assert "coverage" in detail


def test_default_embedder_autoruns(tmp_path):
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))
    text = "Chlorophyll absorbs light."
    doc = kg_document(
        doc_id="doc::test_default_embedder_autoruns",
        content=text,
        source="test_default_embedder_autoruns",
    )
    eng.write.add_document(doc)

    n = Node(
        label="Chlorophyll",
        type="entity",
        summary="absorbs light",
        mentions=[
            kg_grounding(doc.id, start_char=0, end_char=10, excerpt="Chlorophyll")
        ],
    )
    eng.write.add_node(
        n, doc_id=doc.id
    )  # embeddings=None -> auto-embed via DefaultEmbeddingFunction

    got = eng.backend.node_get(ids=[n.id], include=["embeddings"])
    assert (
        got["embeddings"] is not None and len(got["embeddings"][0]) > 0
    )  # vector was created

    # embedding similarity is also available to verifier now
    out = eng.verify_mentions_for_doc(doc.id, min_ngram=4, threshold=0.3)
    assert out["updated_nodes"] >= 1
