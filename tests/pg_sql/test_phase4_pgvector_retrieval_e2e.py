import os

import pytest
import sqlalchemy as sa

from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend


def _pg_dsn() -> str | None:
    return (
        os.environ.get("GKE_PG_DSN")
        or os.environ.get("PG_DSN")
        or os.environ.get("DATABASE_URL")
    )


@pytest.mark.parametrize("distance", ["cosine"])
def test_pgvector_retrieval_ordering_e2e(distance: str, sa_engine, pg_schema):
    """Insert embeddings into Postgres and query by embedding.

    We use simple 3D vectors:
      A=[1,0,0], B=[0.9,0.1,0], C=[0,1,0]

    Querying with [1,0,0] should rank A first, then B, then C for cosine distance.
    """

    # dsn = _pg_dsn()
    # if not dsn:
    #     pytest.skip("Set GKE_PG_DSN (or PG_DSN/DATABASE_URL) to run Postgres E2E tests")

    # schema = f"gke_test_{uuid.uuid4().hex[:8]}"
    # backend = PgVectorBackend(engine=sa_engine, embedding_dim=8, schema=pg_schema)

    # engine = sa.create_engine(dsn, future=True)

    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, distance=distance, schema=pg_schema
    )

    try:
        backend.node_add(
            ids=["A", "B", "C"],
            documents=["A", "B", "C"],
            metadatas=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )

        got = backend.node_query(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        assert got["ids"][0] == ["A", "B", "C"]

        dists = got["distances"][0]
        assert len(dists) == 3
        assert dists[0] <= dists[1] <= dists[2]

    finally:
        # Clean up schema
        with sa_engine.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE'))
