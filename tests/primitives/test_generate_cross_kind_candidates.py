import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    AdjudicationCandidate,
    AdjudicationVerdict,
    Document,
    Edge,
    Grounding,
    MentionVerification,
    Node,
    Span,
)
from kogwistar.llm_tasks import (
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
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend


_required_env = (
    "OPENAI_DEPLOYMENT_NAME_GPT4_1",
    "OPENAI_MODEL_NAME_GPT4_1",
    "OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1",
    "OPENAI_API_KEY_GPT4_1",
)
_skip_real_llm = not all(os.getenv(key) for key in _required_env)


def _doc(*, doc_id: str, content: str, source: str) -> Document:
    return Document(
        id=doc_id,
        content=content,
        type="text",
        metadata={"source": source},
        domain_id=None,
        processed=False,
        embeddings=None,
        source_map=None,
    )


def _grounding_for(doc_id: str) -> Grounding:
    return Grounding(
        spans=[
            Span(
                collection_page_url="c",
                document_page_url=f"document/{doc_id}",
                doc_id=doc_id,
                insertion_method="pytest-manual",
                page_number=1,
                start_char=0,
                end_char=1,
                excerpt="x",
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="heuristic",
                    is_verified=False,
                    notes=None,
                    score=0.9,
                ),
            )
        ]
    )


def _task_set_for_verdict(verdict: AdjudicationVerdict) -> LLMTaskSet:
    payload = {"verdict": verdict.model_dump(mode="python")}
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error="unused"
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=payload,
            raw={"provider": "pytest-stub", "reason": verdict.reason},
            parsing_error=None,
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error="unused"
        ),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(
            node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None
        ),
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        provider_hints=LLMTaskProviderHints(adjudicate_pair_provider="custom"),
    )


def _candidate_dump(candidates: list[AdjudicationCandidate]) -> str:
    return json.dumps(
        [
            {
                "left_kind": candidate.left.kind,
                "left_id": candidate.left.id,
                "left_label": candidate.left.label,
                "right_kind": candidate.right.kind,
                "right_id": candidate.right.id,
                "right_label": candidate.right.label,
                "question": candidate.question,
            }
            for candidate in candidates
        ],
        indent=2,
        sort_keys=True,
    )


def _trace_failure(candidate: AdjudicationCandidate, trace) -> str:
    verdict = trace.adjudication.verdict if trace.adjudication is not None else None
    return "\n".join(
        [
            "Cross-kind adjudication did not return a positive equivalence verdict.",
            f"question={candidate.question}",
            f"left={json.dumps(candidate.left.model_dump(mode='python'), sort_keys=True)}",
            f"right={json.dumps(candidate.right.model_dump(mode='python'), sort_keys=True)}",
            f"same_entity={None if verdict is None else verdict.same_entity}",
            f"confidence={None if verdict is None else verdict.confidence}",
            f"reason={None if verdict is None else verdict.reason}",
            f"parsing_error={trace.parsing_error}",
            f"raw={trace.raw}",
        ]
    )


def _assert_positive_trace(candidate: AdjudicationCandidate, trace) -> None:
    if trace.adjudication is None:
        pytest.fail(_trace_failure(candidate, trace))
    verdict = trace.adjudication.verdict
    if verdict.same_entity is not True:
        pytest.fail(_trace_failure(candidate, trace))


def _seed_reified_relation_fixture(
    engine: GraphKnowledgeEngine,
    *,
    doc_id: str,
    source: str,
) -> tuple[Document, Node, Edge]:
    doc = _doc(
        doc_id=doc_id,
        content=(
            "Photosynthesis includes a named light-to-chemical-energy conversion step. "
            "This document describes that relation instance explicitly."
        ),
        source=source,
    )
    engine.write.add_document(doc)
    ref = _grounding_for(doc.id)

    source_node = Node(
        label="Light energy",
        type="entity",
        summary="Energy captured from sunlight.",
        mentions=[ref],
    )
    target_node = Node(
        label="Chemical energy",
        type="entity",
        summary="Energy stored in glucose and related molecules.",
        mentions=[ref],
    )
    engine.write.add_node(source_node, doc_id=doc.id)
    engine.write.add_node(target_node, doc_id=doc.id)

    signature_text = "photosynthesis converts light energy into chemical energy"
    relation_node = Node(
        label="Photosynthesis light-to-chemical-energy conversion",
        type="entity",
        summary="Named relation instance for the photosynthesis step that converts light energy into chemical energy.",
        mentions=[ref],
        properties={"signature_text": signature_text},
    )
    engine.write.add_node(relation_node, doc_id=doc.id)

    relation_edge = Edge(
        label="Photosynthesis light-to-chemical-energy conversion",
        type="relationship",
        summary="Specific relation instance where photosynthesis converts light energy into chemical energy.",
        relation="converts_to",
        source_ids=[source_node.id],
        target_ids=[target_node.id],
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[ref],
        properties={"signature_text": signature_text},
    )
    engine.write.add_edge(relation_edge, doc_id=doc.id)
    return doc, relation_node, relation_edge


def _find_relation_candidate(
    candidates: list[AdjudicationCandidate],
    *,
    node_id: str,
    edge_id: str,
) -> AdjudicationCandidate | None:
    for candidate in candidates:
        if (
            candidate.left.kind == "node"
            and candidate.left.id == node_id
            and candidate.right.kind == "edge"
            and candidate.right.id == edge_id
            and candidate.question == "node_edge_equivalence"
        ):
            return candidate
    return None


@pytest.fixture(scope="function")
def engine(backend_kind):
    root = Path.cwd() / ".tmp_pytest"
    root.mkdir(exist_ok=True)
    persist_root = Path(tempfile.mkdtemp(prefix="test_cross_kind_", dir=root))
    try:
        kwargs = {
            "persist_directory": str(persist_root / "chroma"),
            "embedding_function": build_test_embedding_function("constant", dim=384),
        }
        if backend_kind == "fake":
            kwargs["backend_factory"] = build_fake_backend
        eng = GraphKnowledgeEngine(**kwargs)
        eng._test_cache_dir = persist_root / "llm_cache"  # type: ignore[attr-defined]
        yield eng
    finally:
        shutil.rmtree(persist_root, ignore_errors=True)


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
def test_generate_cross_kind_candidates_happy_path(engine: GraphKnowledgeEngine):
    doc, relation_node, relation_edge = _seed_reified_relation_fixture(
        engine,
        doc_id="doc::test_generate_cross_kind_candidates_happy_path",
        source="test_generate_cross_kind_candidates_happy_path",
    )

    engine.allow_cross_kind_adjudication = True
    candidates = engine.generate_cross_kind_candidates(scope_doc_id=doc.id)

    candidate = _find_relation_candidate(
        candidates,
        node_id=relation_node.id,
        edge_id=relation_edge.id,
    )
    assert candidate is not None, (
        "Expected a node_edge_equivalence candidate for the reified relation.\n"
        f"candidates={_candidate_dump(candidates)}"
    )
    assert isinstance(candidate, AdjudicationCandidate)

    engine.llm_tasks = _task_set_for_verdict(
        AdjudicationVerdict(
            same_entity=True,
            confidence=0.97,
            reason="The node is an explicit reification of the same conversion relation instance as the edge.",
            canonical_entity_id=None,
        )
    )
    trace = engine.adjudicate_pair_trace(
        candidate.left,
        candidate.right,
        candidate.question,
        cache_dir=getattr(engine, "_test_cache_dir"),
    )
    _assert_positive_trace(candidate, trace)


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
def test_generate_cross_kind_candidates_disabled_scoped_and_limit(
    engine: GraphKnowledgeEngine,
):
    doc1, _, _ = _seed_reified_relation_fixture(
        engine,
        doc_id="doc::test_generate_cross_kind_candidates_disabled_scoped_and_limit::1",
        source="test_generate_cross_kind_candidates_disabled_scoped_and_limit",
    )
    _seed_reified_relation_fixture(
        engine,
        doc_id="doc::test_generate_cross_kind_candidates_disabled_scoped_and_limit::2",
        source="test_generate_cross_kind_candidates_disabled_scoped_and_limit",
    )

    engine.allow_cross_kind_adjudication = False
    with pytest.raises(
        ValueError, match="Configuration disallow cross kind adjudication."
    ):
        engine.generate_cross_kind_candidates(scope_doc_id=doc1.id)

    engine.allow_cross_kind_adjudication = True
    candidates_scoped = engine.generate_cross_kind_candidates(scope_doc_id=doc1.id)
    assert candidates_scoped, "Expected at least one candidate in scoped doc"
    assert all(
        isinstance(candidate, AdjudicationCandidate)
        and candidate.left.kind == "node"
        and candidate.right.kind == "edge"
        and candidate.question == "node_edge_equivalence"
        for candidate in candidates_scoped
    )

    candidates_limited = engine.generate_cross_kind_candidates(
        scope_doc_id=doc1.id, limit_per_bucket=1
    )
    assert len(candidates_limited) >= 1
    assert len(candidates_limited) <= len(candidates_scoped)


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci_full),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.skipif(
    _skip_real_llm, reason="Azure OpenAI env not set for real LLM adjudication"
)
def test_generate_cross_kind_candidates_happy_path_real_llm_reason(
    engine: GraphKnowledgeEngine,
):
    doc, relation_node, relation_edge = _seed_reified_relation_fixture(
        engine,
        doc_id="doc::test_generate_cross_kind_candidates_happy_path_real_llm_reason",
        source="test_generate_cross_kind_candidates_happy_path_real_llm_reason",
    )

    engine.allow_cross_kind_adjudication = True
    candidates = engine.generate_cross_kind_candidates(scope_doc_id=doc.id)
    candidate = _find_relation_candidate(
        candidates,
        node_id=relation_node.id,
        edge_id=relation_edge.id,
    )
    assert candidate is not None, (
        "Expected a node_edge_equivalence candidate for the reified relation.\n"
        f"candidates={_candidate_dump(candidates)}"
    )

    trace = engine.adjudicate_pair_trace(
        candidate.left,
        candidate.right,
        candidate.question,
        cache_dir=getattr(engine, "_test_cache_dir") / "real_provider",
    )
    _assert_positive_trace(candidate, trace)
