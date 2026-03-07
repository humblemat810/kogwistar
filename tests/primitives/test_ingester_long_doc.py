import os
import uuid
import pathlib
import pytest
import requests
from joblib import Memory

# Project imports (adjust if your package name/layout differs)
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Document
from graph_knowledge_engine.ingester import PagewiseSummaryIngestor  # your side-car ingester

# ----------------------------
# Joblib cache for downloads
# ----------------------------
_CACHE_DIR = os.getenv("INGESTER_TEST_CACHE", ".ingester_test_cache")
memory = Memory(_CACHE_DIR, verbose=0)

@memory.cache
def _download_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # Try to keep it large; trim obvious headers/footers for PG texts
    text = r.text
    # A tiny normalization so repeated calls hash-stably:
    return text.replace("\r\n", "\n")

# ----------------------------
# Env & LLM availability
# ----------------------------
def _has_azure_openai():
    # Minimal check; adjust if you use different names
    return all([
        os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
        os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
        os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
        os.getenv("OPENAI_API_KEY_GPT4_1"),
    ])

@pytest.mark.skipif(not _has_azure_openai(), reason="Azure OpenAI env vars not set; skipping long LLM test.")
def test_sidecar_ingester_on_long_document(tmp_path: pathlib.Path):
    """
    Integration test:
      - download & cache a long doc
      - run side-car ingester (split -> summarize -> group -> persist)
      - verify asymmetric edges and node counts
    """
    # 1) Fetch a long public domain text (cached)
    #    You can change the URL if you prefer another document.
    url = "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"  # Moby-Dick
    full_text = _download_text(url)
    assert len(full_text) > 200_000, "Downloaded document seems too short; pick a longer one."

    # 2) Engine in a temporary, isolated Chroma directory
    persist_dir = tmp_path / "db"
    persist_dir = os.path.join('.', 'doc_chroma')
    os.makedirs(persist_dir, exist_ok= True)
    engine = GraphKnowledgeEngine(persist_directory=str(persist_dir))

    # 3) Side-car ingester uses its own explicit model instance.
    from langchain_openai import AzureChatOpenAI

    ingester_llm = AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
        model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
        azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
        openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
        api_version="2024-08-01-preview",
        openai_api_type="azure",
        temperature=0.1,
    )
    ingester = PagewiseSummaryIngestor(engine=engine, llm=ingester_llm, cache_dir=str(os.path.join(".",".llm_cache")))
    partial_doc = full_text[:50000]
    # 4) Create the document row
    doc_id = f"doc-test_sidecar"
    doc = Document(
        id=doc_id,
        content=partial_doc,
        type="text/plain",
        metadata={"source_url": url},
        domain_id=None,
        processed=False,
    )
    engine.add_document(doc)  # explicit (ingester will not call engine's ingest paths)

    # 5) Run the side-car pipeline
    result = ingester.ingest_document(
        document=doc,
        split_max_chars=2500,          # ensures many chunks
        group_size=5,                  # default grouping heuristic
        max_levels=6,
        force_summarise_after_levels=3,   # speed up convergence for tests
    )

    # 6) Basic persistence checks
    node_ids = engine.node_ids_by_doc(doc_id)
    edge_ids = engine.edge_ids_by_doc(doc_id)

    assert len(node_ids) > 0, "No nodes were persisted by the side-car ingester."
    assert len(edge_ids) > 0, "No edges were persisted by the side-car ingester."

    nodes = engine.get_nodes(node_ids)
    edges = engine.get_edges(edge_ids)

    # 7) Relationship checks (asymmetric)
    rels = {e.relation for e in edges}
    assert "summarizes" in rels, "Missing 'summarizes' edges (parent -> child)."
    assert "details" in rels, "Missing 'details' edges (child -> parent)."
    assert "precedes" in rels, "Missing 'precedes' sibling edges (forward order)."
    assert "after" in rels, "Missing 'after' sibling edges (reverse order)."

    # 8) Optional: look for a final summary node (pipeline may force-concat)
    final_nodes = [n for n in nodes if (n.label or "").lower().startswith("final summary")]
    # It's okay if grouping converged to a single non-"Final Summary" label; we won't assert hard here.
    # But do assert at least one summary_chunk exists.
    summary_chunks = [n for n in nodes if (n.label or "").lower().startswith("summary:")]
    assert len(summary_chunks) > 0, "No summary_chunk nodes found."

    # 9) Print a compact note to help debug locally (pytest -s)
    print(
        f"[ingester] doc={doc_id} nodes={len(nodes)} edges={len(edges)} "
        f"final={final_nodes[0].id if final_nodes else 'n/a'}"
    )
