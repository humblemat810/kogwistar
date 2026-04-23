
import pytest

from kogwistar.engine_core.models import Document
from kogwistar.engine_core.models import (
    LLMGraphExtraction,
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

from tests._kg_factories import kg_document

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
