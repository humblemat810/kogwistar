# tests/test_metadata_sanitization.py
import json
from graph_knowledge_engine.models import Node, Span, MentionVerification
from chromadb import app

def _ref_for(doc_id: str) -> Span:
    return _span_for(doc_id)
def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        start_page=1, end_page=1, start_char=0, end_char=1,
        verification=MentionVerification(method="heuristic", is_verified=False, notes = None, score = 0.9), 
        insertion_method="pytest-manual",
        doc_id = doc_id,
        source_cluster_id = None,
        snippet = None
    )

def test_chroma_metadata_strips_none(engine):
    n = Node(
        label="Entity A",
        type="entity",
        summary="An entity without some metadata",
        domain_id=None,
        properties=None,         # should get stripped from metadata
        mentions=[_ref_for(f'test-doc-id1-{__file__}')],         # should get stripped from metadata
        embedding=None           # optional, should not be sent
    )
    engine.add_node(n)
    got = engine.node_collection.get(ids=[n.id])
    assert got["ids"] == [n.id]
    meta = got["metadatas"][0]
    # None fields should be absent
    assert "properties" not in meta
    assert "references" in meta
    # type/summary should be present
    assert meta["type"] == "entity"
    assert meta["summary"] == "An entity without some metadata"
