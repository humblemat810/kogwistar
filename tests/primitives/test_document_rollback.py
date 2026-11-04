
import pytest

from kogwistar.engine_core.models import (
    Document,
    LLMGraphExtraction,
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
from tests._kg_factories import kg_document, kg_grounding
from tests._helpers.graph_builders import build_relationship_edge

pytestmark = pytest.mark.core


def _fake_llm_tasks() -> LLMTaskSet:
    def _extract_graph(request):
        _ = request
        doc_alias = getattr(request, "doc_alias", "doc::fake")
        content = str(getattr(request, "content", "") or "")
        excerpt = content[: min(5, len(content))] or "x"

        # The extraction pipeline validates the LLM slice using insertion_method
        # context, then rehydrates node/edge mentions from those span payloads.
        span = Span.model_validate(
            {
                "collection_page_url": f"document_collection/{doc_alias}",
                "document_page_url": f"document/{doc_alias}",
                "doc_id": doc_alias,
                "page_number": 1,
                "start_char": 0,
                "end_char": len(excerpt),
                "excerpt": excerpt,
                "context_before": "",
                "context_after": content[len(excerpt) :],
                "chunk_id": None,
                "source_cluster_id": None,
            },
            context={"insertion_method": "llm"},
        )
        parsed = LLMGraphExtraction.model_validate(
            {
                "nodes": [
                    {
                        "label": "Photosynthesis",
                        "type": "entity",
                        "summary": "Process converting light to chemical energy",
                        "mentions": [span],
                    },
                    {
                        "label": "Chlorophyll",
                        "type": "entity",
                        "summary": "Molecule absorbing sunlight",
                        "mentions": [span],
                    },
                ],
                "edges": [
                    {
                        "label": "causes",
                        "type": "relationship",
                        "summary": "Chlorophyll absorption enables photosynthesis",
                        "source_ids": ["Chlorophyll"],
                        "target_ids": ["Photosynthesis"],
                        "relation": "enables",
                        "source_edge_ids": [],
                        "target_edge_ids": [],
                        "mentions": [span],
                    }
                ],
            },
            context={"insertion_method": "llm"},
        )
        return ExtractGraphTaskResult(
            raw="fake_raw",
            parsed_payload=parsed,
            parsing_error=None,
        )

    return LLMTaskSet(
        extract_graph=_extract_graph,
        adjudicate_pair=lambda _request: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error="unused"
        ),
        adjudicate_batch=lambda _request: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error="unused"
        ),
        filter_candidates=lambda _request: FilterCandidatesTaskResult(
            node_ids=(), edge_ids=(), reasoning="", raw=None, parsing_error=None
        ),
        summarize_context=lambda request: SummarizeContextTaskResult(
            text=getattr(request, "full_text", "")
        ),
        answer_with_citations=lambda _request: AnswerWithCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        repair_citations=lambda _request: RepairCitationsTaskResult(
            answer_payload=None, raw=None, parsing_error="unused"
        ),
        provider_hints=LLMTaskProviderHints(extract_graph_provider="custom"),
    )


def _document_rollback(engine, *, fake_backend: bool) -> None:
    doc = kg_document(
        doc_id="a882ec6b-75e1-11f0-87ad-0456e5e49702",
        content="The moon orbits the Earth.",
        source="rollback_test",
        doc_type="text",
    )
    engine.node_collection.delete(where={"doc_id": doc.id})

    # Rollback behavior should be deterministic; keep extraction local for both backends.
    engine.llm_tasks = _fake_llm_tasks()

    def ingest_with_doc_with_llm(docd):
        doc_obj = Document.model_validate(docd)
        return engine.ingest_document_with_llm(doc_obj)

    result = ingest_with_doc_with_llm(doc.model_dump())
    assert result is not None

    nodes_before = engine.node_collection.get(where={"doc_id": doc.id})
    assert len(nodes_before["ids"]) > 0

    engine.rollback_document(doc.id)

    nodes_after = engine.backend.node_get(
        where={"$and": [{"doc_id": doc.id}, {"lifecycle_status": "active"}]}
    )
    edges_after = engine.backend.edge_get(
        where={"$and": [{"doc_id": doc.id}, {"lifecycle_status": "active"}]}
    )
    docs_after = engine.document_collection.get(where={"doc_id": doc.id})

    assert len(nodes_after["ids"]) == 0
    assert len(edges_after["ids"]) == 0
    assert len(docs_after["ids"]) == 0


def _batch_document_rollback(engine, *, fake_backend: bool) -> None:
    docs = [
        kg_document(
            doc_id=f"doc::test_batch_document_rollback::{i}",
            content=f"title test Document {i}, Content : test content {i}, this is first sentence of test doc{i}",
            source="test_batch_document_rollback",
            doc_type="text",
        )
        for i in range(3)
    ]

    # Rollback behavior should be deterministic; keep extraction local for both backends.
    engine.llm_tasks = _fake_llm_tasks()

    for doc in docs:
        assert engine.ingest_document_with_llm(doc) is not None

    for doc in docs:
        assert engine.node_collection.get(where={"doc_id": doc.id})["ids"]

    ids_to_remove = [doc.id for doc in docs]
    engine.rollback_many_documents(ids_to_remove)

    for doc in docs:
        assert not engine.backend.node_get(
            where={"$and": [{"doc_id": doc.id}, {"lifecycle_status": "active"}]}
        )["ids"]
        assert not engine.backend.edge_get(
            where={"$and": [{"doc_id": doc.id}, {"lifecycle_status": "active"}]}
        )["ids"]
        assert not engine.document_collection.get(where={"doc_id": doc.id})["ids"]


@pytest.mark.parametrize("backend_kind", ["fake"], indirect=True)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_document_rollback_replacement_ids_are_stable(engine):
    removed_doc = "doc:rollback-idempotency:removed"
    surviving_doc = "doc:rollback-idempotency:surviving"
    mentions = [kg_grounding(removed_doc), kg_grounding(surviving_doc)]
    node = Node(
        id="node:rollback-idempotency",
        label="Shared entity",
        type="entity",
        summary="Shared entity",
        mentions=mentions,
        doc_id=None,
        metadata={},
    )
    edge = build_relationship_edge(
        edge_id="edge:rollback-idempotency",
        src=node.id,
        tgt="node:rollback-target",
        doc_id=removed_doc,
        mentions=mentions,
    )

    first_node = engine.rollback._replacement_node(
        node, [mentions[1]], removed_doc
    )
    retried_node = engine.rollback._replacement_node(
        node, [mentions[1]], removed_doc
    )
    first_edge = engine.rollback._replacement_edge(
        edge,
        mentions=[mentions[1]],
        source_ids=["node:replacement"],
        target_ids=["node:rollback-target"],
        source_edge_ids=[],
        target_edge_ids=[],
        document_id=removed_doc,
    )
    retried_edge = engine.rollback._replacement_edge(
        edge,
        mentions=[mentions[1]],
        source_ids=["node:replacement"],
        target_ids=["node:rollback-target"],
        source_edge_ids=[],
        target_edge_ids=[],
        document_id=removed_doc,
    )

    assert first_node.id == retried_node.id != node.id
    assert first_edge.id == retried_edge.id != edge.id
    assert first_node.id != engine.rollback._replacement_node(
        node, [mentions[1]], "doc:rollback-idempotency:other"
    ).id


@pytest.mark.parametrize("backend_kind", ["fake"], indirect=True)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_document_rollback_retry_after_replacement_write_reuses_id(
    engine, monkeypatch
):
    removed_doc = kg_document(
        doc_id="doc:rollback-retry:removed",
        content="removed source",
        source="rollback-retry-test",
    )
    surviving_doc = kg_document(
        doc_id="doc:rollback-retry:surviving",
        content="surviving source",
        source="rollback-retry-test",
    )
    engine.write.add_document(removed_doc)
    engine.write.add_document(surviving_doc)
    node = Node(
        label="Shared retry entity",
        type="entity",
        summary="Shared retry entity",
        mentions=[kg_grounding(removed_doc.id), kg_grounding(surviving_doc.id)],
    )
    engine.write.add_node(node)

    redirect_node = engine.lifecycle.redirect_node
    attempts = 0

    def fail_first_redirect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated interruption after replacement write")
        return redirect_node(*args, **kwargs)

    monkeypatch.setattr(engine.lifecycle, "redirect_node", fail_first_redirect)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        engine.rollback.rollback_document(removed_doc.id)

    first_replacement = engine.backend.node_get(
        where={"doc_id": surviving_doc.id}, include=["metadatas"]
    )["ids"]
    assert len(first_replacement) == 1

    result = engine.rollback.rollback_document(removed_doc.id)

    assert result["node_redirects"][node.id] == first_replacement[0]
    assert engine.backend.node_get(
        where={"doc_id": surviving_doc.id}, include=["metadatas"]
    )["ids"] == first_replacement


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_document_rollback(engine, backend_kind):
    _document_rollback(engine, fake_backend=backend_kind == "fake")


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", marks=pytest.mark.ci),
        pytest.param("chroma", marks=pytest.mark.ci_full),
    ],
    indirect=True,
)
@pytest.mark.parametrize("embedding_kind", ["constant"], indirect=True)
def test_batch_document_rollback(engine, backend_kind):
    _batch_document_rollback(engine, fake_backend=backend_kind == "fake")
