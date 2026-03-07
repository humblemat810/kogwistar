# tests/test_ingest_document_with_llm_cache.py
import json
from graph_knowledge_engine.engine_core.models import Document
from joblib import Memory
import os, pathlib

def test_ingest_document_with_llm_cache(engine):
    content = "Plants convert light energy. Chlorophyll absorbs sunlight."
    doc = Document(content=content,
                   type="text", metadata={"source":"test"}, domain_id = None, processed = False, 
                   id = "doc::test_ingest_document_with_llm_cache", source_map = None,
                   embeddings = engine.embed.iterative_defensive_emb(content))

    # cache ONLY the pure extraction on the doc content
    cache_dir = os.path.join(".cache","test",pathlib.Path(__file__).name,"extract")
    memory = Memory(location=cache_dir, verbose=0)

    location=os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "_extract_only")
    os.makedirs(location, exist_ok = True)
    memory = Memory(location=location, verbose=0)
    @memory.cache
    def _extract_only(content: str):
        return engine.extract_graph_with_llm(content=content, doc_type='text')

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
    for node_json in nodes["documents"]:
        node = json.loads(node_json)
        assert "id" in node and isinstance(node["id"], str)
        assert node["label"]
        assert node["type"] in ("entity", "relationship")
        assert node["summary"]
        # Check references exist (from fallback)
        assert node['mentions']
        for mention in node['mentions']:
            assert mention['spans']
            for sp in mention['spans']:
                assert sp['collection_page_url'], 'missing collection page url'
                assert sp['document_page_url'], 'missing document page url'
    
    for n in parsed.nodes:
        spans = [sp for sp in n.iter_span()]
        assert spans, f"node {n} has no grounding"
        for sp in spans:
            assert [sp for sp in n.iter_span()][0].collection_page_url, 'missing collection page url'
            assert [sp for sp in n.iter_span()][0].document_page_url, 'missing document page url'
