import json

from graph_knowledge_engine.engine_core.models import Grounding, MentionVerification, Node, Span


def _ref_for(doc_id: str) -> Span:
    return Span(
        collection_page_url="c",
        document_page_url=f"document/{doc_id}",
        page_number=1,
        start_char=0,
        end_char=1,
        verification=MentionVerification(method="heuristic", is_verified=False, notes=None, score=0.9),
        insertion_method="pytest-manual",
        doc_id=doc_id,
        source_cluster_id=None,
        chunk_id=None,
        excerpt="A",
        context_before="",
        context_after="n entity without some metadata",
    )


def test_chroma_metadata_strips_none(engine):
    n = Node(
        label="Entity A",
        type="entity",
        summary="An entity without some metadata",
        domain_id=None,
        properties=None,
        mentions=[Grounding(spans=[_ref_for(f"test-doc-id1-{__file__}")])],
        embedding=None,
        canonical_entity_id=None,
        doc_id=None,
        metadata={},
        level_from_root=0,
    )
    engine.add_node(n)
    got = engine.backend.node_get(ids=[n.id], include=["metadatas"])
    assert got["ids"] == [n.id]
    meta = got["metadatas"][0]
    assert "properties" not in meta
    assert "mentions" in meta
    assert meta["type"] == "entity"
    assert meta["summary"] == "An entity without some metadata"
