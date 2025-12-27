import os, json, pathlib, uuid
import pytest
from joblib import Memory
from typing import List

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Node,
    Edge,
    Document,
    Span,
    LLMMergeAdjudication,
    AdjudicationVerdict,
    AdjudicationQuestionCode,
    QUESTION_KEY,
    MentionVerification
)

# --- Skip if Azure OpenAI env is not configured (prevents CI failures) ---
_required_env = (
    "OPENAI_DEPLOYMENT_NAME_GPT4_1",
    "OPENAI_MODEL_NAME_GPT4_1",
    "OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1",
    "OPENAI_API_KEY_GPT4_1",
)
_skip = not all(os.getenv(k) for k in _required_env)


@pytest.fixture(scope="function")
def engine(tmp_path):
    # fresh Chroma dir per test
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

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
from typing import List
from pydantic import BaseModel
from graph_knowledge_engine.models import LLMMergeAdjudication

class BatchAdjudications(BaseModel):
    items: List[LLMMergeAdjudication]

@pytest.mark.skipif(_skip, reason="Azure OpenAI env not set for real LLM adjudication")
def test_batch_adjudication_with_llm_cache(engine):
    """
    Real LLM adjudication, but joblib-cached so re-runs are fast.
    We cache purely on the *input payload* (pairs payload + mapping table),
    not on the engine instance.
    """
    # 1) Seed a doc + nodes
    doc = Document(content="dummy", type="text", metadata = {}, domain_id = None, processed = False)
    engine.add_document(doc)

    ref = _ref_for(doc.id)
    a = Node(label="Chlorophyll a", type="entity", summary="Pigment in plants", mentions=[ref])
    b = Node(label="Chlorophyll b", type="entity", summary="Another chlorophyll pigment", mentions=[ref])
    c = Node(label="Hemoglobin",   type="entity", summary="Protein in red blood cells", mentions=[ref])

    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)
    engine.add_node(c, doc_id=doc.id)

    # Pairs as pure-JSON payload (so joblib can hash/pickle deterministically)
    pairs_payload = [
        {"left": a.model_dump(), "right": b.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
        {"left": a.model_dump(), "right": c.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
    ]
    mapping_table = [
        {"code": int(code), "key": QUESTION_KEY[code]}
        for code in AdjudicationQuestionCode
    ]

    # 2) Prepare per-test cache dir
    location = os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "batch_adjudicate")
    os.makedirs(location, exist_ok=True)
    memory = Memory(location=location, verbose=0)

    # 3) Cached real-LLM adjudication (rebuilds LLM inside for cache-ability)
    @memory.cache
    def _adjudicate_with_llm_cached(mapping, payload):
        from langchain_openai import AzureChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from graph_knowledge_engine.models import LLMMergeAdjudication

        llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=2000,
            openai_api_type="azure",
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You adjudicate candidate pairs. Use the mapping table to interpret question_code. "
                       "Return only the structured JSON per schema."),
            ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}")
        ])
        chain = prompt | llm.with_structured_output(BatchAdjudications)
        results:BatchAdjudications  = chain.invoke({"mapping": mapping, "pairs": payload})
        # Return a JSON-serializable form for caching
        return results

    # 4) Run (first time hits network, subsequent runs are cached)
    results = _adjudicate_with_llm_cached(mapping_table, pairs_payload)

    # Rehydrate for assertions / commit
    
    assert len(results.items) == 2
    assert all(hasattr(r, "verdict") for r in results.items)

    v1 = results.items[0].verdict
    v2 = results.items[1].verdict
    assert isinstance(v1, AdjudicationVerdict) and isinstance(v2, AdjudicationVerdict)
    # we don't assert True/False — actual model output may vary.
    assert 0.0 <= (v1.confidence or 0.0) <= 1.0
    assert 0.0 <= (v2.confidence or 0.0) <= 1.0

    # 5) Optionally commit positive merges and ensure “same_as” edges appear
    # (committing only those that are positive)
    for (pair, res) in zip([(a,b), (a,c)], results.items):
        if res.verdict.same_entity:
            canonical = engine.commit_merge(pair[0], pair[1], res.verdict)
            assert canonical

    edges = engine.edge_collection.get(include=["metadatas"])
    # If at least one positive, a same_as relation should exist
    if any(r.verdict.same_entity for r in results.items):
        assert any((m or {}).get("relation") == "same_as" for m in (edges.get("metadatas") or []))