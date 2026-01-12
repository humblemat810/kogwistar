from graph_knowledge_engine.models import (
    Node,
    Edge,
    Document)
from joblib import Memory
import os
import pathlib

def test_document_rollback(engine):
    # Create and ingest a dummy document
    
    doc = Document(
        content="The moon orbits the Earth.",
        type="test",
        metadata={"source": "rollback_test"},
        domain_id = None, processed = False
    )
    doc.id = 'a882ec6b-75e1-11f0-87ad-0456e5e49702'
    engine.node_collection.delete(where={"doc_id": doc.id})
    location = os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "test_document_rollback")
    os.makedirs(location, exist_ok=True)
    # memory = Memory(location=location, verbose=0)
    # @memory.cache
    def ingest_with_doc_with_llm(docd):
        doc = Document.model_validate(docd)
        result = engine.ingest_document_with_llm(doc)
        return result
    # Ensure data exists
    result = ingest_with_doc_with_llm(doc.model_dump())
    nodes_before = engine.node_collection.get(where={"doc_id": doc.id})
    assert len(nodes_before["ids"]) > 0

    # Rollback
    engine.rollback_document(doc.id)

    # Ensure all related data is gone
    nodes_after = engine.node_collection.get(where={"doc_id": doc.id})
    edges_after = engine.edge_collection.get(where={"doc_id": doc.id})
    docs_after = engine.document_collection.get(where={"doc_id": doc.id})

    assert len(nodes_after["ids"]) == 0
    assert len(edges_after["ids"]) == 0
    assert len(docs_after["ids"]) == 0
    
def test_batch_document_rollback(engine):
    docs = [
        Document(content=f"title test Document {i}, Content : test content {i}, this is first sentence of test doc{i}", type="test",
                 metadata = {"source": "test_batch_document_rollback"}, domain_id = None, processed = False)
        for i in range(3)
    ]
    inserted = []
    for doc in docs:
        inserted.append(engine.ingest_document_with_llm(doc))

    # Ensure they are in DB
    for doc in docs:
        assert engine.node_collection.get(where={"doc_id": doc.id})["ids"]

    # Rollback in batch
    ids_to_remove = [doc.id for doc in docs]
    engine.rollback_many_documents(ids_to_remove)

    # Verify deletion
    for doc in docs:
        assert not engine.node_collection.get(where={"doc_id": doc.id})["ids"]
        assert not engine.edge_collection.get(where={"doc_id": doc.id})["ids"]
        assert not engine.document_collection.get(where={"doc_id": doc.id})["ids"]