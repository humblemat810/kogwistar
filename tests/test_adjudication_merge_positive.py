# tests/test_adjudication_merge_positive.py
import os
import json
import uuid
import shutil
import pytest
from typing import List

from langchain_core.runnables import Runnable
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Document,
    Node,
    ReferenceSession,
    AdjudicationVerdict,
    LLMMergeAdjudication,
    AdjudicationQuestionCode,
    QUESTION_KEY,
)

def _ref_for(doc_id: str) -> ReferenceSession:
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        start_page=1,
        end_page=1,
        start_char=0,
        end_char=1,
    )

# A deterministic, Runnable-compatible fake that mimics:
# llm.with_structured_output(BatchAdjudications).invoke(...)
class _DeterministicBatchLLM(Runnable):
    def __init__(self):
        self._schema = None
        self._many = True  # we expect a list-like structured output

    # langchain pattern: return self configured for structured output
    def with_structured_output(self, schema, include_raw: bool = False, many: bool = False):
        self._schema = schema
        self._many = many
        return self

    # input: {"mapping": [...], "pairs": [{"left":..., "right":..., "question_code":...}, ...]}
    def invoke(self, input, config=None, **kwargs):
        pairs = input["pairs"]
        items: List[LLMMergeAdjudication] = []

        # Rule: if both labels start with "Chlorophyll", mark same_entity=True (high confidence),
        # else False (low confidence)
        for item in pairs:
            l = item["left"]["label"]
            r = item["right"]["label"]
            if l.startswith("Chlorophyll") and r.startswith("Chlorophyll"):
                ver = AdjudicationVerdict(
                    same_entity=True,
                    confidence=0.93,
                    reason="Deterministic test: both start with 'Chlorophyll'",
                    canonical_entity_id=str(uuid.uuid4()),
                )
            else:
                ver = AdjudicationVerdict(
                    same_entity=False,
                    confidence=0.20,
                    reason="Deterministic test: different families",
                    canonical_entity_id=None,
                )
            items.append(LLMMergeAdjudication(verdict=ver))

        # The schema we pass from the test will be a pydantic model with field `items`
        # that holds List[LLMMergeAdjudication]. Construct it.
        return self._schema(items=items)

@pytest.fixture(scope="function")
def engine(tmp_path):
    # fresh Chroma dir per test
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

def test_deterministic_batch_merge(engine, monkeypatch):
    # Create a document so nodes carry doc_id & refs
    doc = Document(content="dummy", type="text")
    engine.add_document(doc)

    ref = _ref_for(doc.id)
    a = Node(label="Chlorophyll a", type="entity", summary="Pigment in plants", references=[ref])
    b = Node(label="Chlorophyll b", type="entity", summary="Another chlorophyll pigment", references=[ref])
    c = Node(label="Hemoglobin",   type="entity", summary="Protein in red blood cells", references=[ref])

    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)
    engine.add_node(c, doc_id=doc.id)

    # Prepare payload like your cached test (pure JSON)
    pairs = [(a, b), (a, c)]
    pairs_payload = [
        {"left": a.model_dump(), "right": b.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
        {"left": a.model_dump(), "right": c.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
    ]
    mapping_table = [{"code": int(code), "key": QUESTION_KEY[code]} for code in AdjudicationQuestionCode]

    # Define the schema that wraps a list of LLMMergeAdjudication
    from pydantic import BaseModel
    class BatchAdjudications(BaseModel):
        items: List[LLMMergeAdjudication]

    # Patch engine.llm to our deterministic batch adjudicator
    engine.llm = _DeterministicBatchLLM()

    # Build the prompt chain like your real test does
    from langchain.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You adjudicate candidate pairs. Use the mapping table to interpret question_code. "
                   "Return only the structured JSON per schema."),
        ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}"),
    ])
    chain = prompt | engine.llm.with_structured_output(BatchAdjudications)
    results: BatchAdjudications = chain.invoke({"mapping": mapping_table, "pairs": pairs_payload})

    # We deterministically expect two items
    assert len(results.items) == 2
    v1, v2 = results.items[0].verdict, results.items[1].verdict
    assert v1.same_entity is True and v1.confidence > 0.5
    assert v2.same_entity is False and v2.confidence <= 0.5

    # Commit the positive merge (a, b) and verify
    canonical = engine.commit_merge(a, b, v1)
    assert canonical

    # Verify canonical_entity_id persisted on a & b
    a_got = engine.node_collection.get(ids=[a.id])
    b_got = engine.node_collection.get(ids=[b.id])
    a_doc = json.loads(a_got["documents"][0])
    b_doc = json.loads(b_got["documents"][0])
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    # Verify a same_as edge exists
    edges = engine.edge_collection.get(include=["metadatas"])
    assert any((m or {}).get("relation") == "same_as" for m in edges.get("metadatas") or [])
