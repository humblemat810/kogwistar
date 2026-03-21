from kogwistar.engine_core.models import Node

from tests._kg_factories import kg_grounding


def test_embeddings_optional_insert(engine):
    doc_id = "test-did"
    n = Node(
        label="NoEmbed",
        type="entity",
        summary="Inserted without embedding",
        embedding=None,
        mentions=[
            kg_grounding(doc_id, collection_page_url=f"document_collection/{doc_id}")
        ],
    )
    engine.write.add_node(n)
    got = engine.backend.node_get(ids=[n.id])
    assert got["ids"] == [n.id]
