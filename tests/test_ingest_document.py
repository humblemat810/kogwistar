import os
import pytest
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document

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
    print("Ingested document:", result)
