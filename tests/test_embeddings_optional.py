# tests/test_embeddings_optional.py
from graph_knowledge_engine.models import Node,ReferenceSession

def test_embeddings_optional_insert(engine):
    doc_id = 'test-did'
    n = Node(
        label="NoEmbed",
        type="entity",
        summary="Inserted without embedding",
        embedding=None,
        references=[ReferenceSession(
                        collection_page_url=f"document_collection/{doc_id}",
                        document_page_url=f"document/{doc_id}",
                        start_page=1,
                        end_page=1,
                        start_char=0,
                        insertion_method="pytest-manual",
                        end_char=1,
                        doc_id = doc_id
                    )]
    )
    # Should not raise; Chroma accepts missing embeddings
    engine.add_node(n)
    got = engine.node_collection.get(ids=[n.id])
    assert got["ids"] == [n.id]
