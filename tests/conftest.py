# tests/conftest.py
import os, shutil, uuid, json
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parents))
import pytest
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Edge, LLMGraphExtraction, LLMNode, LLMEdge,
    LLMMergeAdjudication, AdjudicationVerdict, Node, ReferenceSession
)
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable
from graph_knowledge_engine.models import LLMMergeAdjudication, AdjudicationVerdict

class FakeStructuredRunnable(Runnable):
    """A minimal Runnable that returns a fixed structured result."""
    def __init__(self, parsed: Any, include_raw: bool = False):
        self._parsed = parsed
        self._include_raw = include_raw

    # sync single
    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        if self._include_raw:
            return {"raw": None, "parsed": self._parsed, "parsing_error": None}
        return self._parsed

    # async single
    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        return self.invoke(input, config=config, **kwargs)

    # sync batch
    def batch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    # async batch
    async def abatch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]


class FakeLLMForAdjudication:
    """
    Test double for your LLM. Mimics `.with_structured_output(...)` by returning
    a Runnable that yields a fixed LLMMergeAdjudication.
    """
    def __init__(self, verdict: AdjudicationVerdict, include_raw: bool = False):
        self._verdict = verdict
        self._include_raw = include_raw

    def with_structured_output(self, schema, include_raw: bool = False, many: bool = False):
        # Build a deterministic structured reply; ignore schema/many in this simple fake
        parsed = LLMMergeAdjudication(verdict=self._verdict)
        return FakeStructuredRunnable(parsed, include_raw=include_raw or self._include_raw)
class _FakeLLMForExtraction:
    """Mocks .with_structured_output(..., include_raw=True) → .invoke(...) for extraction."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        self._include_raw = include_raw
        self._schema = schema
        self._many = many
        return self

    def invoke(self, variables):
        # Deterministic graph from any document
        parsed = LLMGraphExtraction(
            nodes=[
                LLMNode(label="Photosynthesis", type="entity", summary="Process converting light to chemical energy"),
                LLMNode(label="Chlorophyll", type="entity", summary="Molecule absorbing sunlight"),
            ],
            edges=[
                LLMEdge(
                    label="causes",
                    type="relationship",
                    summary="Chlorophyll absorption enables photosynthesis",
                    source_ids=["Chlorophyll"],  # will be mapped later in your pipeline
                    target_ids=["Photosynthesis"],
                    relation="enables"
                )
            ],
        )
        if self._include_raw:
            return {"raw": "fake_raw", "parsed": parsed, "parsing_error": None}
        return parsed

class _FakeLLMForAdjudication:
    """Mocks .with_structured_output(LLMMergeAdjudication) for adjudication."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        self._schema = schema
        self._include_raw = include_raw
        self._many = many
        return self

    def invoke(self, variables):
        # Always say "same entity" with high confidence for test simplicity
        ver = AdjudicationVerdict(
            same_entity=True,
            confidence=0.97,
            reason="Labels and summaries strongly match.",
            canonical_entity_id=str(uuid.uuid4()),
        )
        return LLMMergeAdjudication(verdict=ver)

class _CompositeFakeLLM:
    """Single fake that behaves for both extraction and adjudication chains."""
    def with_structured_output(self, schema, include_raw=False, many=False):
        # route by schema class name
        if getattr(schema, "__name__", "") == "LLMGraphExtraction":
            self._impl = _FakeLLMForExtraction()
        else:
            self._impl = _FakeLLMForAdjudication()
        return self._impl

@pytest.fixture(scope="module")
def tmp_chroma_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chroma_db")
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture(scope="function")
def engine(tmp_chroma_dir, monkeypatch):
    eng = GraphKnowledgeEngine(persist_directory=tmp_chroma_dir)
    # Patch the real LLM with a deterministic fake
    #eng.llm = _CompositeFakeLLM()
    return eng


@pytest.fixture()
def real_small_graph():
    e = GraphKnowledgeEngine(persist_directory = "small_graph")
    doc_id = "D1"
    # nodes
    def add_node(nid, label):
        n = Node(id=nid, label=label, type="entity", summary=label, references=[ReferenceSession(
            collection_page_url=f"document_collection/{doc_id}", document_page_url=f"document/{doc_id}", doc_id=doc_id,
            start_page=1, end_page=1, start_char=0, end_char=1
        )], doc_id=doc_id)
        e.node_collection.add(ids=[nid], documents=[n.model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "label": n.label, "type": n.type}])
        # node_docs link
        ndid = f"{nid}::{doc_id}"
        row = {"id": ndid, "node_id": nid, "doc_id": doc_id}
        e.node_docs_collection.add(ids=[ndid], documents=[json.dumps(row)], metadatas=[row])
        return n

    A = add_node("A", "Smoking")
    B = add_node("B", "Lung Cancer")
    C = add_node("C", "Cough")

    # edge A -[causes]-> B
    e_id = "E1"
    edge = Edge(id=e_id, label="Smoking causes Lung Cancer", type="relationship", summary="causal", relation="causes",
                source_ids=["A"], target_ids=["B"], source_edge_ids=[], target_edge_ids=[],
                references=A.references, doc_id=doc_id)
    e.edge_collection.add(ids=[e_id], documents=[edge.model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "causes"}])
    # endpoints fan-out
    rows = [
        {"id": f"{e_id}::src::node::A", "edge_id": e_id, "endpoint_id": "A", "endpoint_type": "node", "role": "src", "relation": "causes", "doc_id": doc_id},
        {"id": f"{e_id}::tgt::node::B", "edge_id": e_id, "endpoint_id": "B", "endpoint_type": "node", "role": "tgt", "relation": "causes", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows], documents=[json.dumps(r) for r in rows], metadatas=rows)

    # final summary link S -> docnode:D1
    S = add_node("S", "Final Summary")
    e_id2 = "E2"
    e.edge_collection.add(ids=[e_id2], documents=[Edge(
        id=e_id2, label="summarizes_document", type="relationship", summary="S summarizes document", relation="summarizes_document",
        source_ids=["S"], target_ids=[f"docnode:{doc_id}"], source_edge_ids=[], target_edge_ids=[], references=S.references, doc_id=doc_id
    ).model_dump_json(field_mode = 'backend')], metadatas=[{"doc_id": doc_id, "relation": "summarizes_document"}])
    rows2 = [
        {"id": f"{e_id2}::src::node::S", "edge_id": e_id2, "endpoint_id": "S", "endpoint_type": "node", "role": "src", "relation": "summarizes_document", "doc_id": doc_id},
        {"id": f"{e_id2}::tgt::node::docnode:{doc_id}", "edge_id": e_id2, "endpoint_id": f"docnode:{doc_id}", "endpoint_type": "node", "role": "tgt", "relation": "summarizes_document", "doc_id": doc_id},
    ]
    e.edge_endpoints_collection.add(ids=[r["id"] for r in rows2], documents=[json.dumps(r) for r in rows2], metadatas=rows2)

    return e, doc_id
