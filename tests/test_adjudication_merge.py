import json
from graph_knowledge_engine.models import (
    Document, Node, ReferenceSession,
    AdjudicationVerdict,
)
from conftest import FakeLLMForAdjudication  # ensure this is the Runnable version

def _ref_for(doc_id: str) -> ReferenceSession:
    return ReferenceSession(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}", doc_id = doc_id,
        start_page=1, 
        end_page=1, 
        start_char=0, 
        end_char=0,
        insertion_method="pytest-manual",
    )

def _load_node(engine, node_id: str) -> dict:
    got = engine.node_collection.get(ids=[node_id], include=["documents"])
    assert got["documents"], "node not found"
    return json.loads(got["documents"][0])

def test_adjudication_and_commit(engine):
    # create a doc so nodes can carry a doc_id/ref
    doc = Document(content="dummy", type="text")
    engine.add_document(doc)

    a = Node(
        label="Chlorophyll",
        type="entity",
        summary="Pigment in plants",
        references=[_ref_for(doc.id)],
    )
    b = Node(
        label="Chlorophyll (pigment)",
        type="entity",
        summary="Plant pigment; absorbs light",
        references=[_ref_for(doc.id)],
    )

    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)

    # Fake LLM returns a positive merge verdict
    verdict = AdjudicationVerdict(same_entity=True, confidence=0.9, reason="dup", canonical_entity_id=None)
    assert verdict.same_entity is True
    
    engine.llm = FakeLLMForAdjudication(verdict)  # Runnable-compatible test double

    res = engine.adjudicate_merge(a, b)
    verdict = res.verdict if hasattr(res, "verdict") else res
    

    assert verdict.same_entity is True
    assert verdict.confidence >= 0.5

    canonical = engine.commit_merge(a, b, verdict)
    assert canonical

    # Verify canonical_entity_id persisted
    a_doc = _load_node(engine, a.id)
    b_doc = _load_node(engine, b.id)
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    # A same_as edge should exist
    edges = engine.edge_collection.get(include=["metadatas"])
    found_same_as = any((m or {}).get("relation") == "same_as" for m in (edges.get("metadatas") or []))
    assert found_same_as
    
    
    
import pytest
import uuid
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    Node,
    Edge,
    Document,
    ReferenceSession,
    AdjudicationVerdict
)

def _ref_for(doc_id: str) -> ReferenceSession:
    return ReferenceSession(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        start_page=1,
        end_page=1,
        start_char=0,
        end_char=5,
        insertion_method="pytest-manual"
        snippet="dummy snippet", doc_id = doc_id
    )

def test_commit_cross_kind_creates_reifies(engine):
    doc = Document(content="dummy", type="text")
    engine.add_document(doc)
    ref = _ref_for(doc.id)

    # real source/target nodes
    src = Node(label="S", type="entity", summary="src", references=[ref])
    tgt = Node(label="T", type="entity", summary="tgt", references=[ref])
    engine.add_node(src, doc_id=doc.id)
    engine.add_node(tgt, doc_id=doc.id)

    node_a = Node(label="Special Concept", type="entity", summary="as node", references=[ref])
    engine.add_node(node_a, doc_id=doc.id)

    edge_b = Edge(
        label="Special Concept as Relation",
        type="relationship",
        summary="as edge",
        source_ids=[src.id],
        target_ids=[tgt.id],
        source_edge_ids= [],
        target_edge_ids= [],
        relation="has_concept",
        references=[ref],
    )
    engine.add_edge(edge_b, doc_id=doc.id)

    verdict = AdjudicationVerdict(same_entity=True, confidence=0.95, reason="same idea", canonical_entity_id = 'pctt')
    engine.commit_any_kind(
        engine._target_from_node(node_a),
        engine._target_from_edge(edge_b),
        verdict,
    )

    edges = engine.edge_collection.get(include=["metadatas"])
    assert any((m or {}).get("relation") == "reifies" for m in (edges.get("metadatas") or []))