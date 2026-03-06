import os
import uuid

import pytest
import sqlalchemy as sa

from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend, PostgresUnitOfWork


def _pg_dsn() -> str | None:
    return (
        os.environ.get("GKE_PG_DSN")
        or os.environ.get("PG_DSN")
        or os.environ.get("DATABASE_URL")
    )


def test_pg_transaction_rollback_power_out_simulation(sa_engine, pg_schema):
    """Simulate a crash-before-commit and assert no partial writes persist."""
    # dsn = _pg_dsn()
    # if not dsn:
    #     pytest.skip("Set GKE_PG_DSN (or PG_DSN/DATABASE_URL) to run Postgres E2E tests")
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    # backend = build_postgres_backend(PgVectorConfig(url=pg_url))
    # schema = f"gke_test_{uuid.uuid4().hex[:8]}"
    # engine = sa.create_engine(dsn, future=True)
    # backend = PgVectorBackend(engine=engine, embedding_dim=3, schema=schema)
    uow = PostgresUnitOfWork(engine=sa_engine)

    try:
        # Begin transaction, write, then raise before commit.
        with pytest.raises(RuntimeError):
            with uow.transaction():
                backend.node_add(
                    ids=["X"],
                    documents=["X"],
                    metadatas=[{"name": "X"}],
                    embeddings=[[1.0, 0.0, 0.0]],
                )
                raise RuntimeError("power-out")

        # After rollback, the row should not exist.
        got = backend.node_get(ids=["X"], include=["ids"])
        assert got.get("ids") == []

    finally:
        with sa_engine.begin() as conn:
            conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE'))
