import os
import pathlib
import importlib.util
import pytest
pytestmark = pytest.mark.ci_full
import requests
from joblib import Memory
from tests._helpers.embeddings import build_test_embedding_function

# Project imports (adjust if your package name/layout differs)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Document
from kogwistar.ingester import (
    PagewiseSummaryIngestor,
)  # your side-car ingester

# ----------------------------
# Joblib cache for downloads
# ----------------------------
_CACHE_DIR = os.getenv("INGESTER_TEST_CACHE", ".ingester_test_cache")
memory = Memory(_CACHE_DIR, verbose=0)


def _test_embedding_function():
    dim = int(os.getenv("KOGWISTAR_TEST_EMBEDDING_DIM", "384"))
    kind = str(os.getenv("KOGWISTAR_TEST_EMBEDDING_KIND", "lexical_hash")).strip().lower()
    ef = build_test_embedding_function(kind, dim=dim)
    return ef


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
    return all(
        [
            os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            os.getenv("OPENAI_API_KEY_GPT4_1"),
        ]
    )


def _has_ollama():
    return importlib.util.find_spec("langchain_ollama") is not None and bool(
        os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL")
    )


def _build_ollama_ingester_llm():
    if not _has_ollama():
        pytest.skip("Need local Ollama for default long-doc ingester test.")
    from langchain_ollama import ChatOllama

    model_name = os.getenv("INGESTER_OLLAMA_MODEL", "gemma4:e2b")
    return ChatOllama(
        model=model_name,
        temperature=0.1,
    )


def _build_azure_ingester_llm():
    if not _has_azure_openai():
        pytest.skip("Azure OpenAI env vars not set for manual long-doc ingester test.")
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
        model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
        azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
        openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
        api_version="2024-08-01-preview",
        openai_api_type="azure",
        temperature=0.1,
    )


@pytest.mark.parametrize(
    "provider_kind",
    [
        pytest.param("ollama", id="ollama", marks=pytest.mark.slow),
        pytest.param("azure", id="azure", marks=pytest.mark.manual),
    ],
)
def test_sidecar_ingester_on_long_document(
    tmp_path: pathlib.Path, provider_kind: str, monkeypatch: pytest.MonkeyPatch
):
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
    assert len(full_text) > 200_000, (
        "Downloaded document seems too short; pick a longer one."
    )

    # 2) Engine in a temporary, isolated Chroma directory
    persist_dir = tmp_path / "db"
    os.makedirs(persist_dir, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(persist_dir),
        embedding_function=_test_embedding_function(),
    )

    # 3) Side-car ingester uses local Ollama by default; Azure is manual-only.
    if provider_kind == "ollama":
        monkeypatch.setenv("OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        monkeypatch.setenv(
            "INGESTER_OLLAMA_MODEL", os.getenv("INGESTER_OLLAMA_MODEL", "gemma4:e2b")
        )
        ingester_llm = _build_ollama_ingester_llm()
    elif provider_kind == "azure":
        ingester_llm = _build_azure_ingester_llm()
    else:
        raise AssertionError(f"unknown provider_kind={provider_kind!r}")
    ingester = PagewiseSummaryIngestor(
        engine=engine, llm=ingester_llm, cache_dir=str(os.path.join(".", ".llm_cache"))
    )
    partial_doc = full_text[:20000] if provider_kind == "ollama" else full_text[:50000]
    # 4) Create the document row
    doc_id = "doc-test_sidecar"
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
        split_max_chars=2500,  # ensures many chunks
        group_size=5,  # default grouping heuristic
        max_levels=6,
        force_summarise_after_levels=3,  # speed up convergence for tests
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
    final_nodes = [
        n for n in nodes if (n.label or "").lower().startswith("final summary")
    ]
    # It's okay if grouping converged to a single non-"Final Summary" label; we won't assert hard here.
    # But do assert at least one summary_chunk exists.
    summary_chunks = [
        n for n in nodes if (n.label or "").lower().startswith("summary:")
    ]
    assert len(summary_chunks) > 0, "No summary_chunk nodes found."

    # 9) Print a compact note to help debug locally (pytest -s)
    print(
        f"[ingester] provider={provider_kind} doc={doc_id} nodes={len(nodes)} edges={len(edges)} "
        f"final={final_nodes[0].id if final_nodes else 'n/a'}"
    )
