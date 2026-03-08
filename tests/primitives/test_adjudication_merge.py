import json

from graph_knowledge_engine.engine_core.models import AdjudicationVerdict, Edge, Node
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
from tests._kg_factories import kg_document, kg_grounding


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


def _load_node(engine, node_id: str) -> dict:
    got = engine.backend.node_get(ids=[node_id], include=["documents"])
    assert got["documents"], "node not found"
    return json.loads(got["documents"][0])


def test_adjudication_and_commit(engine):
    doc = kg_document(
        doc_id="doc::test_adjunication_merge_with_llm_cache",
        content="dummy",
        source="test_adjunication_merge_with_llm_cache",
        embeddings=engine.embed.iterative_defensive_emb("dummy"),
    )
    engine.write.add_document(doc)

    a = Node(
        label="Chlorophyll",
        type="entity",
        summary="Pigment in plants",
        mentions=[kg_grounding(doc.id, excerpt="d")],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    b = Node(
        label="Chlorophyll (pigment)",
        type="entity",
        summary="Plant pigment; absorbs light",
        mentions=[kg_grounding(doc.id, excerpt="d")],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )

    engine.write.add_node(a, doc_id=doc.id)
    engine.write.add_node(b, doc_id=doc.id)

    verdict = AdjudicationVerdict(same_entity=True, confidence=0.9, reason="dup", canonical_entity_id=None)
    engine.llm_tasks = _task_set_for_verdict(verdict)

    res = engine.adjudicate_merge(a, b)
    verdict = res.verdict if hasattr(res, "verdict") else res

    assert verdict.same_entity is True
    assert verdict.confidence >= 0.5

    canonical = engine.commit_merge(a, b, verdict, method="test_adj")
    assert canonical

    a_doc = _load_node(engine, a.id)
    b_doc = _load_node(engine, b.id)
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    edges = engine.backend.edge_get(include=["metadatas"])
    assert any((m or {}).get("relation") == "same_as" for m in (edges.get("metadatas") or []))


def test_commit_cross_kind_creates_reifies(engine):
    doc = kg_document(
        doc_id="doc::test_commit_cross_kind_creates_reifies",
        content="dummy",
        source="test_commit_cross_kind_creates_reifies",
        embeddings=engine.embed.iterative_defensive_emb("dummy"),
    )
    engine.write.add_document(doc)
    ref = kg_grounding(doc.id, excerpt="d")

    src = Node(
        label="S",
        type="entity",
        summary="src",
        mentions=[ref],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    tgt = Node(
        label="T",
        type="entity",
        summary="tgt",
        mentions=[ref],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    engine.write.add_node(src, doc_id=doc.id)
    engine.write.add_node(tgt, doc_id=doc.id)

    node_a = Node(
        label="Special Concept",
        type="entity",
        summary="as node",
        mentions=[ref],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
        level_from_root=0,
    )
    engine.write.add_node(node_a, doc_id=doc.id)

    edge_b = Edge(
        label="Special Concept as Relation",
        type="relationship",
        summary="as edge",
        source_ids=[src.id],
        target_ids=[tgt.id],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="has_concept",
        mentions=[ref],
        metadata={"source": "test_commit_cross_kind_creates_reifies"},
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        doc_id=None,
    )
    engine.write.add_edge(edge_b, doc_id=doc.id)

    verdict = AdjudicationVerdict(same_entity=True, confidence=0.95, reason="same idea", canonical_entity_id="pctt")
    engine.commit_any_kind(
        engine._target_from_node(node_a),
        engine._target_from_edge(edge_b),
        verdict,
    )

    edges = engine.backend.edge_get(include=["metadatas"])
    assert any((m or {}).get("relation") == "reifies" for m in (edges.get("metadatas") or []))
