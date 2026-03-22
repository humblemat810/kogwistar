import pytest
pytestmark = pytest.mark.core

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend
from tests.conftest import FakeEmbeddingFunction

EMBEDDING_DIM = 3
TEST_EMBEDDING = FakeEmbeddingFunction(dim=EMBEDDING_DIM)


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


@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
def test_engine_vector_search_nodes_orders_neighbors(
    backend_kind: str, request: pytest.FixtureRequest, tmp_path
):
    """Engine-level smoke: add 3 nodes with known embeddings, vector search returns correct order.

    This runs against both:
      - in-memory fake backend
      - Chroma backend (local persistent)
      - Postgres+pgvector backend (sa_engine/pg_schema fixtures)
    """
    if backend_kind == "fake":
        eng = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "fake_engine"),
            embedding_function=TEST_EMBEDDING,
            backend_factory=build_fake_backend,
        )
    elif backend_kind == "chroma":
        eng = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "chroma_engine"),
            embedding_function=TEST_EMBEDDING,
        )
    else:
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        if sa_engine is None or pg_schema is None:
            pytest.skip("pg test container/fixtures are not available in this environment")
        backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=3, distance="cosine", schema=pg_schema
        )
        eng = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "pg_engine"),
            backend=backend,
            embedding_function=TEST_EMBEDDING,
        )

    _insert_three(eng.backend)
    got = eng.vector_search_nodes([1.0, 0.0, 0.0], top_k=3)
    assert got["ids"][0] == ["A", "B", "C"]


@pytest.mark.ci_full
@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
def test_engine_uow_power_out_rollback(
    backend_kind: str, request: pytest.FixtureRequest, tmp_path
):
    """Transaction atomicity: crash-before-commit rolls back all writes (pgvector only).

    For Chroma we skip because there's no SQL transaction to roll back.
    """
    if backend_kind == "chroma":
        pytest.skip("rollback semantics are only meaningful for the pgvector backend")

    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pgvector")
    from kogwistar.engine_core.postgres_backend import (
        PgVectorBackend,
        PostgresUnitOfWork,
    )

    sa_engine = request.getfixturevalue("sa_engine")
    pg_schema = request.getfixturevalue("pg_schema")
    if sa_engine is None or pg_schema is None:
        pytest.skip("pg test container/fixtures are not available in this environment")

    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, distance="cosine", schema=pg_schema
    )
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "pg_engine_txn"), backend=backend
    )
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
