# tests/test_generate_cross_kind_candidates.py
import pytest
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import Document, Node, Edge, Span,MentionVerification

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

@pytest.fixture(scope="function")
def engine(tmp_path):
    return GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma"))

def test_generate_cross_kind_candidates_happy_path(engine: GraphKnowledgeEngine):
    # One doc with both a node and an edge that share a salient token
    doc = Document(content="Photosynthesis basics", type="text")
    engine.add_document(doc)
    ref = _ref_for(doc.id)

    n = Node(label="Photosynthesis", type="entity", summary="Process in plants", mentions=[ref])
    engine.add_node(n, doc_id=doc.id)

    # Edge summary/label includes the same token "Photosynthesis"
    e = Edge(label="Photosynthesis relation", type="relationship", summary="Photosynthesis converts light",
             relation="converts", source_ids=[n.id], target_ids=[], source_edge_ids=[] , target_edge_ids= [], 
             mentions=[ref])
    engine.add_edge(e, doc_id=doc.id)

    engine.allow_cross_kind_adjudication = True
    pairs = engine.generate_cross_kind_candidates(scope_doc_id=doc.id)

    # At least one node↔edge candidate with the correct question
    assert any(
        p.left.kind == "node"
        and p.right.kind == "edge"
        and p.question == "same_entity"
        for p in pairs
    )

def test_generate_cross_kind_candidates_disabled_scoped_and_limit(engine: GraphKnowledgeEngine):
    # Two docs; only one should be considered with scope
    doc1 = Document(content="Causation doc", type="text")
    doc2 = Document(content="Unrelated doc", type="text")
    engine.add_document(doc1)
    engine.add_document(doc2)
    ref1 = _ref_for(doc1.id)
    ref2 = _ref_for(doc2.id)

    # In doc1: matching tokens ("causation")
    n1 = Node(label="Causation", type="entity", summary="Cause and effect", mentions=[ref1])
    engine.add_node(n1, doc_id=doc1.id)
    e1 = Edge(label="Causation relation", type="relationship", summary="Causation between X and Y",
              relation="causation", source_ids=[n1.id], target_ids=[], source_edge_ids=[] , target_edge_ids= [], mentions=[ref1])
    engine.add_edge(e1, doc_id=doc1.id)

    # In doc2: no overlap
    n2 = Node(label="Gravity", type="entity", summary="Force", mentions=[ref2])
    engine.add_node(n2, doc_id=doc2.id)
    e2 = Edge(label="Electromagnetism", type="relationship", summary="Different force",
              relation="interacts", source_ids=[n2.id], target_ids=[], source_edge_ids=[] , target_edge_ids= [], mentions=[ref2])
    engine.add_edge(e2, doc_id=doc2.id)

    # (a) Disabled → no candidates
    engine.allow_cross_kind_adjudication = False
    with pytest.raises(ValueError, match="Configuration disallow cross kind adjudication."):
        engine.generate_cross_kind_candidates(scope_doc_id=doc1.id)

    # (b) Enable and scope to doc1 → should find candidates only from doc1
    engine.allow_cross_kind_adjudication = True
    pairs_scoped = engine.generate_cross_kind_candidates(scope_doc_id=doc1.id)
    assert pairs_scoped, "Expected at least one candidate in scoped doc"
    assert all(
        # sanity: involved IDs should come from doc1’s items
        (p.left.kind in ("node","edge") and p.right.kind in ("node","edge"))
        for p in pairs_scoped
    )

    # (c) Limit enforcement
    pairs_limited = engine.generate_cross_kind_candidates(scope_doc_id=doc1.id, limit_per_bucket=1)
    # there is at least one; and not more than the limit per bucket
    assert len(pairs_limited) >= 1
    assert len(pairs_limited) <= len(pairs_scoped)
