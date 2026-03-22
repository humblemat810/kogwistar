import json
from typing import Mapping

import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationVerdict,
    Document,
    Grounding,
    MentionVerification,
    Node,
    QUESTION_KEY,
    Span,
)
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskRequest,
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


def _span_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        page_number=1,
        start_char=0,
        end_char=1,
        verification=MentionVerification(
            method="heuristic", is_verified=False, notes=None, score=0.9
        ),
        insertion_method="pytest-manual",
        doc_id=doc_id,
        source_cluster_id=None,
        chunk_id=None,
        excerpt="d",
        context_before="",
        context_after="ummy",
    )


def _first_token_batch_task_set() -> LLMTaskSet:
    def _adjudicate_batch(
        request: AdjudicateBatchTaskRequest,
    ) -> AdjudicateBatchTaskResult:
        items: list[Mapping[str, object]] = []
        for item in request.pairs:
            left = item.get("left") if isinstance(item, Mapping) else {}
            right = item.get("right") if isinstance(item, Mapping) else {}
            left_name = left.get("name") if isinstance(left, Mapping) else ""
            right_name = right.get("name") if isinstance(right, Mapping) else ""
            ltok = str(left_name or "").split()[:1]
            rtok = str(right_name or "").split()[:1]
            same = bool(ltok and rtok and ltok[0].lower() == rtok[0].lower())
            ver = AdjudicationVerdict(
                same_entity=same,
                confidence=0.95 if same else 0.20,
                reason="first-token match" if same else "first-token differs",
                canonical_entity_id=None,
            )
            items.append({"verdict": ver.model_dump(mode="python")})
        return AdjudicateBatchTaskResult(
            verdict_payloads=items, raw=None, parsing_error=None
        )

    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error="unused"
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error="unused"
        ),
        adjudicate_batch=_adjudicate_batch,
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
        provider_hints=LLMTaskProviderHints(adjudicate_batch_provider="custom"),
    )


@pytest.fixture(scope="function")
def engine(tmp_path, backend_kind):
    kwargs = {
        "persist_directory": str(tmp_path / "chroma"),
        "embedding_function": build_test_embedding_function("constant", dim=384),
    }
    if backend_kind == "fake":
        kwargs["backend_factory"] = build_fake_backend
    return GraphKnowledgeEngine(**kwargs)


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
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
    a = Node(
        label="Chlorophyll a",
        type="entity",
        summary="Pigment in plants",
        mentions=[Grounding(spans=[ref])],
    )
    b = Node(
        label="Chlorophyll b",
        type="entity",
        summary="Another chlorophyll pigment",
        mentions=[Grounding(spans=[ref])],
    )
    c = Node(
        label="Hemoglobin",
        type="entity",
        summary="Protein in red blood cells",
        mentions=[Grounding(spans=[ref])],
    )

    engine.write.add_node(a, doc_id=doc.id)
    engine.write.add_node(b, doc_id=doc.id)
    engine.write.add_node(c, doc_id=doc.id)

    pairs_payload = [
        {
            "left": a.model_dump(),
            "right": b.model_dump(),
            "question_code": int(AdjudicationQuestionCode.SAME_ENTITY),
        },
        {
            "left": a.model_dump(),
            "right": c.model_dump(),
            "question_code": int(AdjudicationQuestionCode.SAME_ENTITY),
        },
    ]
    mapping_table = [
        {"code": int(code), "key": QUESTION_KEY[code]}
        for code in AdjudicationQuestionCode
    ]
    _ = (pairs_payload, mapping_table)
    engine.llm_tasks = _first_token_batch_task_set()

    results, _qkey = engine.batch_adjudicate_merges(
        [(a, b), (a, c)],
        question_code=AdjudicationQuestionCode.SAME_ENTITY,
    )

    assert len(results) == 2
    v1, v2 = results[0].verdict, results[1].verdict
    assert v1.same_entity is True and v1.confidence > 0.5
    assert v2.same_entity is False and v2.confidence <= 0.5

    canonical = engine.commit_merge(a, b, v1, method="llm")
    assert canonical

    a_got = engine.backend.node_get(ids=[a.id], include=["documents"])
    b_got = engine.backend.node_get(ids=[b.id], include=["documents"])
    a_doc = json.loads(a_got["documents"][0])
    b_doc = json.loads(b_got["documents"][0])
    assert a_doc.get("canonical_entity_id") == canonical
    assert b_doc.get("canonical_entity_id") == canonical

    edges = engine.backend.edge_get(include=["metadatas"])
    assert any(
        (m or {}).get("relation") == "same_as" for m in edges.get("metadatas") or []
    )
