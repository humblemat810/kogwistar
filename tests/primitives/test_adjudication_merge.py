import json
from graph_knowledge_engine.engine_core.models import (
    Document, Node, Span,
    AdjudicationVerdict,MentionVerification
)
from graph_knowledge_engine.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)


def _task_set_for_verdict(verdict: AdjudicationVerdict) -> LLMTaskSet:
    pair_payload = {"verdict": verdict.model_dump(mode="python")}
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(raw=None, parsed_payload=None, parsing_error="unused"),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(verdict_payload=pair_payload, raw=None, parsing_error=None),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(verdict_payloads=(), raw=None, parsing_error="unused"),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None),
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(answer_payload=None, raw=None, parsing_error="unused"),
        repair_citations=lambda _req: RepairCitationsTaskResult(answer_payload=None, raw=None, parsing_error="unused"),
        provider_hints=LLMTaskProviderHints(adjudicate_pair_provider="custom"),
    )

def _ref_for(doc_id: str) -> Span:
    return _span_for(doc_id)
def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        start_page=1, end_page=1, start_char=0, end_char=1, page_number = 1,
        excerpt = "test text", context_before = "", context_after = "",
        verification=MentionVerification(method="heuristic", is_verified=False, notes = None, score = 0.9), 
        insertion_method="pytest-manual",
        doc_id = doc_id,
        source_cluster_id = None,
    )

def _load_node(engine, node_id: str) -> dict:
    got = engine.node_collection.get(ids=[node_id], include=["documents"])
    assert got["documents"], "node not found"
    return json.loads(got["documents"][0])

def test_adjudication_and_commit(engine):
    # create a doc so nodes can carry a doc_id/ref
    doc = Document(id='doc::test_adjunication_merge_with_llm_cache', 
                   content="dummy", type="text", metadata = {}, domain_id = None, processed = False, source_map = None,
                   embeddings = engine.embed.iterative_defensive_emb("dummy"))
    engine.add_document(doc)

    a = Node(
        label="Chlorophyll",
        type="entity",
        summary="Pigment in plants",
        mentions = [Grounding(spans = [_ref_for(doc.id)])], metadata = {"source": "test_commit_cross_kind_creates_reifies"}, 
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, level_from_root = 0,
    )
    b = Node(
        label="Chlorophyll (pigment)",
        type="entity",
        summary="Plant pigment; absorbs light",
        mentions = [Grounding(spans = [_ref_for(doc.id)])], metadata = {"source": "test_commit_cross_kind_creates_reifies"}, 
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, level_from_root = 0,
    )

    engine.add_node(a, doc_id=doc.id)
    engine.add_node(b, doc_id=doc.id)

    # Fake LLM returns a positive merge verdict
    verdict = AdjudicationVerdict(same_entity=True, confidence=0.9, reason="dup", canonical_entity_id=None)
    assert verdict.same_entity is True
    
    engine.llm_tasks = _task_set_for_verdict(verdict)

    res = engine.adjudicate_merge(a, b)
    verdict = res.verdict if hasattr(res, "verdict") else res
    

    assert verdict.same_entity is True
    assert verdict.confidence >= 0.5

    canonical = engine.commit_merge(a, b, verdict, method = "test_adj")
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
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import (
    Node,
    Edge,
    Document,
    Span,
    Grounding,
    AdjudicationVerdict
)


def test_commit_cross_kind_creates_reifies(engine):
    doc = Document(id="doc::test_commit_cross_kind_creates_reifies", content="dummy", type="text", metadata = {"source":"test_commit_cross_kind_creates_reifies"}, 
                   domain_id = None, processed = False, source_map = None, embeddings = engine.embed.iterative_defensive_emb("dummy"))
    engine.add_document(doc)
    ref = _ref_for(doc.id)

    # real source/target nodes
    src = Node(label="S", type="entity", summary="src", 
               mentions = [Grounding(spans=[ref])], metadata = {"source": "test_commit_cross_kind_creates_reifies"}, 
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, level_from_root = 0)
    tgt = Node(label="T", type="entity", summary="tgt", 
               mentions = [Grounding(spans=[ref])], metadata = {"source": "test_commit_cross_kind_creates_reifies"},
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, level_from_root = 0)
    engine.add_node(src, doc_id=doc.id)
    engine.add_node(tgt, doc_id=doc.id)

    node_a = Node(label="Special Concept", type="entity", summary="as node", 
                  mentions = [Grounding(spans=[ref])], metadata = {"source": "test_commit_cross_kind_creates_reifies"}, 
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None, level_from_root = 0)
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
        mentions = [Grounding(spans=[ref])], metadata = {"source": "test_commit_cross_kind_creates_reifies"}, 
               domain_id=None, canonical_entity_id=None, properties=None, embedding=None, doc_id=None,
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
