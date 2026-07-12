import json
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Document, MentionVerification, Span
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.span_consistent_seed import build_span_consistent_debug_rag_seed
from tests.conftest import FakeEmbeddingFunction


pytestmark = pytest.mark.ci


@pytest.fixture
def engine_tmp(tmp_path):
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "persist_document_graph_extraction"),
        embedding_function=FakeEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )


def test_persist_document_graph_extraction_accepts_consistent_spans(engine_tmp):
    seed = build_span_consistent_debug_rag_seed(doc_id="doc::persist_ok")
    engine_tmp.write.add_document(seed.document)

    result = engine_tmp.persist.persist_document_graph_extraction(
        doc_id=seed.document.id,
        parsed=seed.as_graph_extraction(),
    )

    assert result["document_id"] == seed.document.id
    assert result["nodes_added"] == len(seed.nodes)
    assert result["edges_added"] == len(seed.edges)
    got_nodes = engine_tmp.backend.node_get(
        ids=[node.id for node in seed.nodes], include=["documents"]
    )
    assert len(got_nodes["documents"]) == len(seed.nodes)
    got_edges = engine_tmp.backend.edge_get(
        ids=[edge.id for edge in seed.edges], include=["documents"]
    )
    assert len(got_edges["documents"]) == len(seed.edges)


def test_persist_document_graph_extraction_rejects_incorrect_span(engine_tmp):
    seed = build_span_consistent_debug_rag_seed(doc_id="doc::persist_bad")
    bad_graph = seed.as_graph_extraction()
    bad_graph.nodes[0].mentions[0].spans[0].excerpt = "incorrect excerpt"
    engine_tmp.write.add_document(seed.document)

    with pytest.raises(Exception, match="Incorrect span occur"):
        engine_tmp.persist.persist_document_graph_extraction(
            doc_id=seed.document.id,
            parsed=bad_graph,
        )


def test_span_repair_recovers_unique_boundary_offset(engine_tmp):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "boundary_span_failure_payload.json"
    ).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    excerpt = str(fixture["span"]["excerpt"])
    document = Document.from_text(
        str(fixture["source_prefix_character"]) * int(fixture["source_prefix_length"])
        + excerpt
        + str(fixture["source_suffix"]),
        id=str(fixture["source_document_id"]),
    )
    engine_tmp.write.add_document(document)
    start = document.content.index(excerpt)
    validator = engine_tmp.get_span_validator_of_doc_type(document=document)
    span = Span(
        collection_page_url="doc://boundary_repair",
        document_page_url="doc://boundary_repair",
        doc_id=document.id,
        insertion_method="workflow_ingest",
        page_number=1,
        start_char=int(fixture["span"]["start_char"]),
        end_char=int(fixture["span"]["end_char"]),
        excerpt=excerpt,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=str(fixture["span"]["source_cluster_id"]),
        verification=MentionVerification(
            method="heuristic", is_verified=False, score=None, notes="fixture"
        ),
    )

    repaired, diagnostics = validator.repair_span(span, doc=document)

    assert diagnostics["repaired"] is True
    assert repaired.start_char == start
    assert repaired.end_char == start + len(excerpt)
    assert validator.validate_span(repaired, doc=document)["correctness"] is True


def test_span_repair_rejects_ambiguous_excerpt(engine_tmp):
    document = Document.from_text("same\nother\nsame\n", id="doc::ambiguous_repair")
    engine_tmp.write.add_document(document)
    validator = engine_tmp.get_span_validator_of_doc_type(document=document)
    span = Span(
        collection_page_url="doc://ambiguous_repair",
        document_page_url="doc://ambiguous_repair",
        doc_id=document.id,
        insertion_method="workflow_ingest",
        page_number=1,
        start_char=1,
        end_char=5,
        excerpt="same",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id="doc::ambiguous_repair|p1_t0",
        verification=None,
    )

    repaired, diagnostics = validator.repair_span(span, doc=document)

    assert repaired == span
    assert diagnostics["match_mode"] == "ambiguous_exact"
