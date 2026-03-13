import json
import shutil
import tempfile
from pathlib import Path

import pytest

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


@pytest.fixture(scope="function")
def engine():
    root = Path.cwd() / ".tmp_pytest"
    root.mkdir(exist_ok=True)
    persist_root = Path(tempfile.mkdtemp(prefix="test_batch_adj_", dir=root))
    try:
        yield GraphKnowledgeEngine(persist_directory=str(persist_root / "chroma"))
    finally:
        shutil.rmtree(persist_root, ignore_errors=True)


def _doc() -> Document:
    return Document(
        id="doc::test_batch_adjudication_and_commit",
        content="dummy",
        type="text",
        metadata={"source": "test_batch_adjudication_and_commit"},
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
                excerpt="d",
                context_before="",
                context_after="ummy",
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


def test_batch_adjudication_and_commit(engine, monkeypatch):
    doc = _doc()
    engine.write.add_document(doc)

    ref = _grounding_for(doc.id)
    a = Node(
        label="Chlorophyll a",
        type="entity",
        summary="Pigment in plants",
        mentions=[ref],
    )
    b = Node(
        label="Chlorophyll b",
        type="entity",
        summary="Another chlorophyll pigment",
        mentions=[ref],
    )
    c = Node(
        label="Hemoglobin",
        type="entity",
        summary="Protein in red blood cells",
        mentions=[ref],
    )

    engine.write.add_node(a, doc_id=doc.id)
    engine.write.add_node(b, doc_id=doc.id)
    engine.write.add_node(c, doc_id=doc.id)

    pairs = [(a, b), (a, c)]

    def fake_batch_adjudicate_merges(
        pairs, question_code=AdjudicationQuestionCode.SAME_ENTITY
    ):
        outs = []
        for left, right in pairs:
            if "Chlorophyll a" in left.label and "Chlorophyll b" in right.label:
                verdict = AdjudicationVerdict(
                    same_entity=True,
                    confidence=0.9,
                    reason="similar pigments",
                    canonical_entity_id=None,
                )
            else:
                verdict = AdjudicationVerdict(
                    same_entity=False,
                    confidence=0.2,
                    reason="different concepts",
                    canonical_entity_id=None,
                )
            outs.append(LLMMergeAdjudication(verdict=verdict))
        qkey = QUESTION_KEY[AdjudicationQuestionCode(question_code)]
        return outs, qkey

    monkeypatch.setattr(engine, "batch_adjudicate_merges", fake_batch_adjudicate_merges)

    results, qkey = engine.batch_adjudicate_merges(
        pairs,
        question_code=AdjudicationQuestionCode.SAME_ENTITY,
    )

    assert qkey == QUESTION_KEY[AdjudicationQuestionCode.SAME_ENTITY]
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(hasattr(r, "verdict") for r in results)

    v1 = results[0].verdict
    v2 = results[1].verdict

    assert v1.same_entity is True and v1.confidence > 0.5
    assert v2.same_entity is False and v2.confidence <= 0.5

    canonical = engine.commit_merge(a, b, v1, method="pytest_batch_adjudication")
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
