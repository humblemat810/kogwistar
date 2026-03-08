import os
import pathlib
from typing import List, cast

import pytest
from joblib import Memory
from pydantic import BaseModel

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    LLMMergeAdjudication,
    Node,
    QUESTION_KEY,
)
from tests._kg_factories import kg_document, kg_grounding


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
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))


class BatchAdjudications(BaseModel):
    items: List[LLMMergeAdjudication]


@pytest.mark.skipif(_skip, reason="Azure OpenAI env not set for real LLM adjudication")
def test_batch_adjudication_with_llm_cache(engine):
    """
    Real LLM adjudication, but joblib-cached so re-runs are fast.
    We cache purely on the input payload, not on the engine instance.
    """
    doc = kg_document(
        doc_id="doc::test_batch_adjudication_with_llm_cache",
        content="dummy",
        source="test_batch_adjudication_with_llm_cache",
    )
    engine.write.add_document(doc)

    a = Node(
        label="Chlorophyll a",
        type="entity",
        summary="Pigment in plants",
        mentions=[kg_grounding(doc.id)],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
    )
    b = Node(
        label="Chlorophyll b",
        type="entity",
        summary="Another chlorophyll pigment",
        mentions=[kg_grounding(doc.id)],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
    )
    c = Node(
        label="Hemoglobin",
        type="entity",
        summary="Protein in red blood cells",
        mentions=[kg_grounding(doc.id)],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
    )

    engine.write.add_node(a, doc_id=doc.id)
    engine.write.add_node(b, doc_id=doc.id)
    engine.write.add_node(c, doc_id=doc.id)

    pairs_payload = [
        {"left": a.model_dump(), "right": b.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
        {"left": a.model_dump(), "right": c.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
    ]
    mapping_table = [
        {"code": int(code), "key": QUESTION_KEY[code]}
        for code in AdjudicationQuestionCode
    ]

    location = os.path.join(".cache", "test", pathlib.Path(__file__).parts[-1], "batch_adjudicate")
    os.makedirs(location, exist_ok=True)
    memory = Memory(location=location, verbose=0)

    @memory.cache
    def _adjudicate_with_llm_cached(mapping, payload):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),  # type: ignore[arg-type]
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),  # type: ignore[arg-type]
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),  # type: ignore[arg-type]
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=2000,
            openai_api_type="azure",
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You adjudicate candidate pairs. Use the mapping table to interpret question_code. "
                "Return only the structured JSON per schema.",
            ),
            ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}"),
        ])
        chain = prompt | llm.with_structured_output(BatchAdjudications)
        results: BatchAdjudications = cast(BatchAdjudications, chain.invoke({"mapping": mapping, "pairs": payload}))
        return results

    results = _adjudicate_with_llm_cached(mapping_table, pairs_payload)

    assert len(results.items) == 2
    assert all(hasattr(r, "verdict") for r in results.items)

    v1 = results.items[0].verdict
    v2 = results.items[1].verdict
    assert isinstance(v1, AdjudicationVerdict) and isinstance(v2, AdjudicationVerdict)
    assert 0.0 <= (v1.confidence or 0.0) <= 1.0
    assert 0.0 <= (v2.confidence or 0.0) <= 1.0

    for pair, res in zip([(a, b), (a, c)], results.items):
        if res.verdict.same_entity:
            canonical = engine.commit_merge(pair[0], pair[1], res.verdict, method="pytest_batch_llm_cache")
            assert canonical

    edges = engine.backend.edge_get(include=["metadatas"])
    if any(r.verdict.same_entity for r in results.items):
        assert any((m or {}).get("relation") == "same_as" for m in (edges.get("metadatas") or []))
