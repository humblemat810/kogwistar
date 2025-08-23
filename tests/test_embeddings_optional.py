# tests/test_embeddings_optional.py
from graph_knowledge_engine.models import Node

def test_embeddings_optional_insert(engine):
    n = Node(
        label="NoEmbed",
        type="entity",
        summary="Inserted without embedding",
        embedding=None,
    )
    # Should not raise; Chroma accepts missing embeddings
    engine.add_node(n)
    got = engine.node_collection.get(ids=[n.id])
    assert got["ids"] == [n.id]
