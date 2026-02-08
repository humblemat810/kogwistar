import pytest
import sqlalchemy as sa

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.postgres_backend import PgVectorBackend

# New meta-store introduced by the patch
from graph_knowledge_engine.engine_postgres_meta import EnginePostgresMetaStore


def _fake_ef(texts):
    # deterministic, dimension=3
    return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.mark.parametrize("distance", ["cosine"])
def test_pg_backend_uses_postgres_meta_store(sa_engine, pg_schema, tmp_path, distance: str):
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, distance=distance, schema=pg_schema)

    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "gke_meta"),
        embedding_function=_fake_ef,
        backend=backend,
        kg_graph_type="knowledge",
    )

    assert isinstance(eng.meta_sqlite, EnginePostgresMetaStore)


def _count_user_seq_rows(sa_engine, schema: str) -> int:
    q = sa.text(f"SELECT COUNT(*) FROM {schema}.user_seq")
    with sa_engine.connect() as conn:
        return int(conn.execute(q).scalar_one())


def _count_nodes(sa_engine, schema: str) -> int:
    q = sa.text(f"SELECT COUNT(*) FROM {schema}.gke_nodes")
    with sa_engine.connect() as conn:
        return int(conn.execute(q).scalar_one())


@pytest.mark.parametrize("distance", ["cosine"])
def test_engine_uow_rolls_back_meta_and_graph_writes_together(sa_engine, pg_schema, tmp_path, distance: str):
    '''
    Proves the new PG meta-store participates in the SAME PG transaction as pgvector writes.

    Inside engine.uow():
      - bump user_seq (meta write)
      - insert a node (graph write)
      - raise to simulate crash-before-commit

    After exception:
      - gke_user_seq should have no row (meta write rolled back)
      - gke_nodes should have no row (graph write rolled back)
    '''
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, distance=distance, schema=pg_schema)
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "gke_meta"),
        embedding_function=_fake_ef,
        backend=backend,
        kg_graph_type="knowledge",
    )

    # Pre-assert empty
    assert _count_user_seq_rows(sa_engine, pg_schema) == 0
    assert _count_nodes(sa_engine, pg_schema) == 0

    with pytest.raises(RuntimeError):
        with eng.uow():
            # meta write
            _ = eng.meta_sqlite.next_user_seq("user1")
            # graph write (direct to backend to keep this test stable)
            eng.backend.node_add(
                ids=["X"],
                documents=["X"],
                metadatas=[{"name": "X"}],
                embeddings=[[1.0, 0.0, 0.0]],
            )
            raise RuntimeError("power out")

    # Both should be rolled back if they share the same PG transaction
    assert _count_user_seq_rows(sa_engine, pg_schema) == 0
    assert _count_nodes(sa_engine, pg_schema) == 0
