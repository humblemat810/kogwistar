# tests/test_metadata_sanitization.py
import json
from graph_knowledge_engine.models import Node, Span
from chromadb import app

def _ref_for(doc_id: str) -> Span:
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        start_page=1,
        end_page=1,
        start_char=0,
        insertion_method="pytest-manual",
        end_char=1,
        doc_id = doc_id
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
