import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
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
