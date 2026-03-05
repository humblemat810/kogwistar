import ast
import json
from typing import List

import pytest
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    Document,
    Grounding,
    LLMMergeAdjudication,
    MentionVerification,
    Node,
    QUESTION_KEY,
    Span,
)


def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        page_number=1,
        start_char=0,
        end_char=1,
        verification=MentionVerification(method="heuristic", is_verified=False, notes=None, score=0.9),
        insertion_method="pytest-manual",
        doc_id=doc_id,
        source_cluster_id=None,
        chunk_id=None,
        excerpt="d",
        context_before="",
        context_after="ummy",
    )


class BatchAdjudications(BaseModel):
    items: List[LLMMergeAdjudication]


class _DeterministicBatchLLM(Runnable):
    """Runnable fake that supports prompt|llm composition and dict inputs."""

    def with_structured_output(self, schema, **_):
        self._schema = schema
        return self

    def _extract_pairs_from_prompt(self, cpv: ChatPromptValue):
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
            left = item["left"]
            right = item["right"]
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

        return BatchAdjudications(items=items)


@pytest.fixture(scope="function")
def engine(tmp_path):
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))


def test_deterministic_batch_merge(engine, monkeypatch):
    doc = Document(
        id="doc-deterministic-batch-merge",
        content="dummy",
        type="text",
        metadata={"source": "test_deterministic_batch_merge"},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )
    engine.add_document(doc)

    ref = _span_for(doc.id)
    a = Node(label="Chlorophyll a", type="entity", summary="Pigment in plants", mentions=[Grounding(spans=[ref])])
    b = Node(label="Chlorophyll b", type="entity", summary="Another chlorophyll pigment", mentions=[Grounding(spans=[ref])])
    c = Node(label="Hemoglobin", type="entity", summary="Protein in red blood cells", mentions=[Grounding(spans=[ref])])

    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)
    engine.add_node(c, doc_id=doc.id)

    pairs_payload = [
        {"left": a.model_dump(), "right": b.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
        {"left": a.model_dump(), "right": c.model_dump(), "question_code": int(AdjudicationQuestionCode.SAME_ENTITY)},
    ]
    mapping_table = [{"code": int(code), "key": QUESTION_KEY[code]} for code in AdjudicationQuestionCode]

    from langchain_core.prompts import ChatPromptTemplate

    engine.llm = _DeterministicBatchLLM()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You adjudicate candidate pairs. Use the mapping table to interpret question_code. "
                "Return only the structured JSON per schema.",
            ),
            ("human", "Mapping table:\n{mapping}\n\nPairs:\n{pairs}"),
        ]
    )
    chain = prompt | engine.llm.with_structured_output(BatchAdjudications)
    # for embeddings to jsonable
    for i in pairs_payload:
        if type(i['left']['embedding']) is not list:
            i['left']['embedding'] = i['left']['embedding'].tolist()
        if type(i['right']['embedding']) is not list:
            i['right']['embedding'] = i['right']['embedding'].tolist()
    results: BatchAdjudications = chain.invoke({"mapping": json.dumps(mapping_table), "pairs": json.dumps(pairs_payload)})

    assert len(results.items) == 2
    v1, v2 = results.items[0].verdict, results.items[1].verdict
    assert v1.same_entity is True and v1.confidence > 0.5
    assert v2.same_entity is False and v2.confidence <= 0.5

    canonical = engine.commit_merge(a, b, v1, method='llm')
    assert canonical

    a_got = engine.backend.node_get(ids=[a.id], include=["documents"])
    b_got = engine.backend.node_get(ids=[b.id], include=["documents"])
    a_doc = json.loads(a_got["documents"][0])
    b_doc = json.loads(b_got["documents"][0])
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    edges = engine.backend.edge_get(include=["metadatas"])
    assert any((m or {}).get("relation") == "same_as" for m in edges.get("metadatas") or [])
