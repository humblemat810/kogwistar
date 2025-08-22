# tests/test_adjudication_merge_positive.py
import os
import json
import uuid
import shutil
import pytest
import ast
from typing import List
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
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
        insertion_method="pytest-manual"
        end_char=1,
        doc_id = doc_id
    )

# A deterministic, Runnable-compatible fake that mimics:
# llm.with_structured_output(BatchAdjudications).invoke(...)
class BatchAdjudications(BaseModel):
    items: List[LLMMergeAdjudication]

class _DeterministicBatchLLM(Runnable):
    """Runnable fake that supports prompt|llm composition and dict inputs."""
    def with_structured_output(self, schema, **_):
        # Keep parity with LangChain’s API; we’ll just remember the schema.
        self._schema = schema
        return self

    def _extract_pairs_from_prompt(self, cpv: ChatPromptValue):
        # Find the last HumanMessage and parse the JSON (or Python-literal) after "Pairs:\n"
        msgs = cpv.to_messages()
        human = next((m for m in msgs[::-1] if isinstance(m, HumanMessage)), None)
        if not human:
            raise ValueError("No HumanMessage found in ChatPromptValue")
        text = human.content
        marker = "Pairs:\n"
        i = text.find(marker)
        if i < 0:
            raise ValueError("Could not find 'Pairs:' in the HumanMessage")
        payload = text[i + len(marker):].strip()
        # Be robust to models that emit Python-ish literals
        try:
            pairs = json.loads(payload)
        except Exception:
            pairs = ast.literal_eval(payload)
        return pairs

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, ChatPromptValue):
            pairs = self._extract_pairs_from_prompt(input)
        elif isinstance(input, dict):
            pairs = input["pairs"]
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

        items = []
        for item in pairs:
            left = item["left"]; right = item["right"]
            # Simple deterministic rule: first token of labels equal => same
            ltok = (left.get("label") or "").split()[:1]
            rtok = (right.get("label") or "").split()[:1]
            same = bool(ltok and rtok and ltok[0].lower() == rtok[0].lower())
            ver = AdjudicationVerdict(
                same_entity=same,
                confidence=0.95 if same else 0.20,
                reason="first-token match" if same else "first-token differs",
                canonical_entity_id=None,
            )
            items.append(LLMMergeAdjudication(verdict=ver))

        # Return the Pydantic wrapper the test expects
        return BatchAdjudications(items=items)

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
