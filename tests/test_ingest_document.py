import os
import pytest
import json
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, ReferenceSession

@pytest.fixture(scope="module")
def engine():
    return GraphKnowledgeEngine()

def test_ingest_document_with_llm(engine):
    # Example document content
    doc_content = (
        "Photosynthesis is a process used by plants to convert light energy into chemical energy. "
        "Chlorophyll is the molecule that absorbs sunlight. "
        "Plants perform photosynthesis in their leaves."
    )
    document = Document(
        content=doc_content,
        type="ocr",
        metadata={"source": "test"},
        processed=False
    )

    # The .env file must be configured with Azure OpenAI credentials
    result = engine.ingest_document_with_llm(document)
    assert "document_id" in result
    assert result["nodes_added"] >= 1
    assert result["edges_added"] >= 1

    # Retrieve nodes and edges from ChromaDB
    nodes = engine.node_collection.get()
    edges = engine.edge_collection.get()

    # Check that at least one node and one edge has a non-empty references field
    node_found = False
    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        if node.get("references"):
            node_found = True
            # Check ReferenceSession structure
            ref = node["references"][0]
            assert "collection_page_url" in ref and "document_page_url" in ref
    assert node_found, "No node with references found"

    edge_found = False
    for edge_json in edges["documents"]:
        edge = json.loads(edge_json)
        if edge.get("references"):
            edge_found = True
            ref = edge["references"][0]
            assert "collection_page_url" in ref and "document_page_url" in ref
    assert edge_found, "No edge with references found"

    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        assert "id" in node and isinstance(node["id"], str)
        assert node["label"] and node["type"]

    for edge_json in edges["documents"]:
        edge = json.loads(edge_json)
        assert isinstance(edge.get("source_ids", []), list)
        assert isinstance(edge.get("target_ids", []), list)
        assert isinstance(edge.get("relation", ""), str)
    print("Ingested document:", result)

#{'document_id': 'cbd56c36-7664-11f0-9315-0456e5e49702', 'node_ids': ['668ad754-4a3f-4ba9-a6ab-9178daa45e8c', '21a81be9-4257-41c7-93fd-a3ef32273eec', '81f7100a-7ecb-4645-ab2c-00b9323dcf27', '7a8e0857-ae61-483f-84bc-e246b92b31d7'], 'edge_ids': ['2a3d53e2-06dc-4b41-9368-71d56216035a', 'dd5fe3b6-9e7b-4583-b6a5-2cc35db3725e', '49d92d0a-1ab3-42d3-91af-1bf80563d9e5'], 'nodes_added': 4, 'edges_added': 3}