# tests/test_ingest_document_mock.py
import json
from graph_knowledge_engine.models import Document

def test_ingest_document_with_llm_mock(engine):
    doc = Document(
        content="Plants convert light energy. Chlorophyll absorbs sunlight.",
        type="ocr",
        metadata={"source": "test"},
        processed=False
    )
    result = engine.ingest_document_with_llm(doc)
    assert result["document_id"] == doc.id
    assert result["nodes_added"] >= 1
    assert result["edges_added"] >= 1

    nodes = engine.node_collection.get()
    edges = engine.edge_collection.get()
    assert len(nodes["ids"]) >= 1
    assert len(edges["ids"]) >= 1

    # references present (fallback URLs ok)
    any_node_ref = False
    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        assert "id" in node and isinstance(node["id"], str)
        assert node["label"]
        assert node["type"] in ("entity", "relationship")
        assert node["summary"]
        # Check references exist (from fallback)
        if node.get("references"):
            r0 = node["references"][0]
            assert "collection_page_url" in r0 and "document_page_url" in r0
            any_node_ref = True
    assert any_node_ref, "Expected at least one node to have references"
