import json
import pytest
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import MentionVerification, Node

from tests.conftest import FakeEmbeddingFunction
from tests._kg_factories import kg_document, kg_grounding


@pytest.fixture
def engine_tmp(tmp_path):
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=FakeEmbeddingFunction(dim=8),
    )


def test_verify_mentions_for_doc(engine_tmp):
    # 1) add a doc with known text
    text = "Photosynthesis lets plants convert light energy to chemical energy. Chlorophyll is the pigment that absorbs light."
    doc = kg_document(
        doc_id="doc::test_verify_mentions_for_doc",
        content=text,
        source="test_verify_mentions_for_doc",
    )
    engine_tmp.write.add_document(doc)

    # 2) add a node that claims something present in the text
    start = text.index("Chlorophyll")
    excerpt = "Chlorophyll is the pigment that absorbs light."
    n = Node(
        label="Chlorophyll",
        type="entity",
        summary="Chlorophyll is the pigment that absorbs light",
        mentions=[
            kg_grounding(
                doc.id,
                start_char=start,
                end_char=start + len(excerpt),
                excerpt=excerpt,
                context_before=text[max(0, start - 20) : start],
                context_after="",
                collection_page_url=f"document_collection/{doc.id}",
            )
        ],
        doc_id=doc.id,
    )
    engine_tmp.write.add_node(n, doc_id=doc.id)

    # 3) run verification against the current Grounding->Span contract.
    out = engine_tmp.verify_mentions_for_doc(
        doc.id,
        source_text=text,
        min_ngram=5,
        threshold=0.5,
    )
    assert out["updated_nodes"] >= 1

    got = engine_tmp.backend.node_get(ids=[n.id], include=["documents"])
    nn = Node.model_validate_json(got["documents"][0])
    assert nn.mentions and nn.mentions[0].spans[0].verification is not None
    mv: MentionVerification = nn.mentions[0].spans[0].verification
    assert mv.method == "ensemble"
    assert mv.score is not None
    # notes should be compact JSON with individual scores inside
    detail = json.loads(mv.notes)
    assert "coverage" in detail
