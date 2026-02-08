import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine

try:
    from graph_knowledge_engine.postgres_backend import PgVectorBackend, PostgresUnitOfWork
except Exception:  # pragma: no cover
    PgVectorBackend = None  # type: ignore
    PostgresUnitOfWork = None  # type: ignore


def _insert_three(backend) -> None:
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


@pytest.mark.parametrize("backend_kind", ["chroma", "pgvector"])
def test_engine_vector_search_nodes_orders_neighbors(backend_kind: str, tmp_path, sa_engine, pg_schema):
    """Engine-level smoke: add 3 nodes with known embeddings, vector search returns correct order.

    This runs against both:
      - Chroma backend (local persistent)
      - Postgres+pgvector backend (sa_engine/pg_schema fixtures)
    """
    if backend_kind == "chroma":
        eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "chroma_engine"))
    else:
        if PgVectorBackend is None:
            pytest.skip("pgvector not installed")
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, distance="cosine", schema=pg_schema)
        eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "pg_engine"), backend=backend)

    _insert_three(eng.backend)
    got = eng.vector_search_nodes([1.0, 0.0, 0.0], top_k=3)
    assert got["ids"][0] == ["A", "B", "C"]


@pytest.mark.parametrize("backend_kind", ["chroma", "pgvector"])
def test_engine_uow_power_out_rollback(backend_kind: str, tmp_path, sa_engine, pg_schema):
    """Transaction atomicity: crash-before-commit rolls back all writes (pgvector only).

    For Chroma we skip because there's no SQL transaction to roll back.
    """
    if backend_kind == "chroma":
        pytest.skip("rollback semantics are only meaningful for the pgvector backend")

    if PgVectorBackend is None or PostgresUnitOfWork is None:
        pytest.skip("pgvector not installed")

    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, distance="cosine", schema=pg_schema)
    eng = GraphKnowledgeEngine(persist_directory=str(tmp_path / "pg_engine_txn"), backend=backend)
    uow = PostgresUnitOfWork(engine=sa_engine)

    with pytest.raises(RuntimeError):
        with uow.transaction():
            eng.backend.node_add(
                ids=["X"],
                documents=["X"],
                metadatas=[{"name": "X"}],
                embeddings=[[1.0, 0.0, 0.0]],
            )
            raise RuntimeError("simulate power outage before commit")

    got = eng.backend.node_get(ids=["X"], include=[])
    assert got.get("ids") == []
