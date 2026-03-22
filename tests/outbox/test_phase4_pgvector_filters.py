import pytest
pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.engine_postgres import PgVectorBackend


@pytest.mark.ci_full
def test_pgvector_where_filters_tombstones(sa_engine, pg_schema):
    be = PgVectorBackend(engine=sa_engine, embedding_dim=2, schema=pg_schema)
    be.ensure_schema()

    be.node_delete(ids=["active", "dead"])

    be.node_add(
        ids=["active", "dead"],
        documents=["ok", "rip"],
        metadatas=[
            {"lifecycle_status": "active"},
            {"lifecycle_status": "tombstoned"},
        ],
        embeddings=[
            [1.0, 0.0],
            [1.0, 0.0],
        ],
    )

    got = be.node_query(
        query_embeddings=[[1.0, 0.0]],
        n_results=10,
        where={"lifecycle_status": "active"},
        include=["metadatas"],
    )

    assert got["ids"][0] == ["active"]
    assert got["metadatas"][0][0]["lifecycle_status"] == "active"
