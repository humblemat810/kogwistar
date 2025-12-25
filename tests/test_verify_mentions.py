import json
import pytest
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Span, MentionVerification

@pytest.fixture
def engine_tmp(tmp_path):
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

def _ref(doc_id, start=0, end=40, snippet=None):
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}", doc_id = doc_id,
        insertion_method="pytest-manual",
        start_page=1, end_page=1, start_char=start, end_char=end,
        snippet=snippet or None
    )

def test_verify_mentions_for_doc(engine_tmp):
    # 1) add a doc with known text
    text = "Photosynthesis lets plants convert light energy to chemical energy. Chlorophyll is the pigment that absorbs light."
    doc = Document(content=text, type="text")
    engine_tmp.add_document(doc)

    # 2) add a node that claims something present in the text
    n = Node(
        label="Chlorophyll",
        type="entity",
        summary="Chlorophyll is the pigment that absorbs light",
        mentions=[_ref(doc.id, start=text.index("Chlorophyll"), end=text.index("light.")+6)]
    )
    engine_tmp.add_node(n, doc_id=doc.id)

    # 3) run verification (RapidFuzz optional; coverage alone is fine)
    out = engine_tmp.verify_mentions_for_doc(doc.id, min_ngram=5, threshold=0.5)
    assert out["updated_nodes"] >= 1

    got = engine_tmp.node_collection.get(ids=[n.id], include=["documents"])
    nn = Node.model_validate_json(got["documents"][0])
    assert nn.mentions and nn.mentions[0].verification is not None
    mv: MentionVerification = nn.mentions[0].verification
    assert mv.method == "ensemble"
    assert mv.score is not None
    # notes should be compact JSON with individual scores inside
    detail = json.loads(mv.notes)
    assert "coverage" in detail
