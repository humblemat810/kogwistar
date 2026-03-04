import pytest

from graph_knowledge_engine.engine_core.engine_postgres import PgVectorBackend


@pytest.mark.integration
def test_pgvector_similarity_ordering_is_correct(sa_engine, pg_schema):
    be = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    be.ensure_schema()

    # Clean slate
    be.node_delete(ids=["a", "b", "c"])

    # Three points in 3D; query is [1,0,0] so expected order a, b, c
    be.node_add(
        ids=["a", "b", "c"],
        documents=["a", "b", "c"],
        metadatas=[
            {"lifecycle_status": "active"},
            {"lifecycle_status": "active"},
            {"lifecycle_status": "active"},
        ],
        embeddings=[
            [1.0, 0.0, 0.0],  # closest
            [0.9, 0.0, 0.0],  # second
            [0.0, 1.0, 0.0],  # far
        ],
    )

    got = be.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=3,
        where={"lifecycle_status": "active"},
        include=["documents", "metadatas", "distances"],
    )

    assert got["ids"][0] == ["a", "b", "c"]


@pytest.mark.integration
def test_pgvector_ties_break_by_id(sa_engine, pg_schema):
    be = PgVectorBackend(engine=sa_engine, embedding_dim=2, schema=pg_schema)
    be.ensure_schema()

    be.node_delete(ids=["id_1", "id_2"])

    # Identical embeddings => identical distance. Insert in reverse order to prove tie-break.
    be.node_add(
        ids=["id_2", "id_1"],
        documents=["2", "1"],
        metadatas=[
            {"lifecycle_status": "active"},
            {"lifecycle_status": "active"},
        ],
        embeddings=[
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )

    got = be.node_query(
        query_embeddings=[[1.0, 0.0]],
        n_results=2,
        where={"lifecycle_status": "active"},
        include=["distances"],
    )

    assert got["ids"][0] == ["id_1", "id_2"]
