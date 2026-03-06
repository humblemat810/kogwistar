# tests/test_ingest_document_with_llm_cache.py
import json
from graph_knowledge_engine.engine_core.models import Document
from joblib import Memory
import os, pathlib

def test_ingest_document_with_llm_cache(engine):
    doc = Document(content="Plants convert light energy. Chlorophyll absorbs sunlight.",
                   type="ocr", metadata={"source":"test"}, domain_id = None, processed = False)

    # cache ONLY the pure extraction on the doc content
    cache_dir = os.path.join(".cache","test",pathlib.Path(__file__).name,"extract")
    memory = Memory(location=cache_dir, verbose=0)

    location=os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "_extract_only")
    os.makedirs(location, exist_ok = True)
    memory = Memory(location=location, verbose=0)
    @memory.cache
    def _extract_only(content: str):
        return engine.extract_graph_with_llm(content=content)

    extracted = _extract_only(doc.content)
    parsed = extracted["parsed"]

    # persist deterministically (choose replace/append/skip-if-exists)
    out = engine.persist_graph_extraction(document=doc, parsed=parsed, mode="replace")

    assert out["document_id"] == doc.id
    assert len(out["node_ids"]) >= 1
    assert len(out["edge_ids"]) >= 1

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
