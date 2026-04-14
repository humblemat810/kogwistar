import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Document, GraphExtractionWithIDs, Grounding, Node, Span
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


def test_persist_document_graph_extraction_coerces_lossless_ocr_pointers_to_mentions(engine_tmp):
    doc = Document.from_ocr(
        id="doc::persist_ocr_pointer_coercion",
        ocr_content={
            "sample.pdf": [
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [
                        {
                            "text": "Hello world",
                            "bb_x_min": 0.0,
                            "bb_x_max": 10.0,
                            "bb_y_min": 0.0,
                            "bb_y_max": 10.0,
                            "cluster_number": 0,
                        }
                    ],
                    "non_text_objects": [],
                    "is_empty_page": False,
                    "printed_page_number": "1",
                    "meaningful_ordering": [0],
                    "page_x_min": 0.0,
                    "page_x_max": 100.0,
                    "page_y_min": 0.0,
                    "page_y_max": 100.0,
                    "estimated_rotation_degrees": 0.0,
                    "incomplete_words_on_edge": False,
                    "incomplete_text": False,
                    "data_loss_likelihood": 0.0,
                    "scan_quality": "high",
                    "contains_table": False,
                }
            ]
        },
        type="ocr",
    )
    engine_tmp.write.add_document(doc)

    span = Span(
        collection_page_url=f"document_collection/{doc.id}",
        document_page_url=f"document/{doc.id}#p1_c0",
        doc_id=doc.id,
        insertion_method="pytest-manual",
        page_number=1,
        start_char=0,
        end_char=5,
        excerpt="Hello",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id="p1_c0",
    )
    node = Node(
        id="11111111-1111-1111-1111-111111111111",
        label="Greeting",
        type="entity",
        summary="Opening word",
        mentions=[Grounding(spans=[span])],
        doc_id=doc.id,
        embedding=None,
        metadata={
            "pointers": [
                {
                    "source_cluster_id": "p1_c0",
                    "start_char": 0,
                    "end_char": 5,
                    "verbatim_text": "Hello",
                }
            ]
        },
        level_from_root=0,
    )
    parsed = GraphExtractionWithIDs(nodes=[node], edges=[])

    result = engine_tmp.persist.persist_document_graph_extraction(
        doc_id=doc.id,
        parsed=parsed,
    )

    assert result["nodes_added"] == 1
    got = engine_tmp.backend.node_get(ids=[node.id], include=["metadatas", "documents"])
    assert got["ids"] == [node.id]
    meta = got["metadatas"][0]
    assert "pointers" not in meta
    assert "mentions" in meta
