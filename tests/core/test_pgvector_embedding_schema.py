from __future__ import annotations

from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from kogwistar.engine_core.postgres_backend import (
    PgVectorBackend,
    PgVectorSchemaMismatchError,
)


pytestmark = [pytest.mark.ci, pytest.mark.unit, pytest.mark.regression]


class _Result:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[dict[str, str]]:
        return self._rows


class _Connection:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.params: dict[str, object] | None = None

    def execute(self, _statement: object, params: dict[str, object]) -> _Result:
        self.params = params
        return _Result(self.rows)


class _AsyncCountResult:
    def __init__(self, count: int) -> None:
        self.count = count

    def scalar_one(self) -> int:
        return self.count


class _AsyncConnection:
    def __init__(self) -> None:
        self.index = 0

    async def __aenter__(self) -> "_AsyncConnection":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def execute(self, _statement: object) -> _AsyncCountResult:
        self.index += 1
        return _AsyncCountResult(self.index)


def _backend_for_dimension_check(expected_dimension: int) -> PgVectorBackend:
    backend = object.__new__(PgVectorBackend)
    backend.schema = "wiki"
    backend.embedding_dim = expected_dimension
    backend.nodes = SimpleNamespace(name="gke_nodes")
    backend.edges = SimpleNamespace(name="gke_edges")
    backend.documents = SimpleNamespace(name="gke_documents")
    backend.domains = SimpleNamespace(name="gke_domains")
    return backend


def test_pgvector_dimension_guard_reports_stale_column_before_writes() -> None:
    connection = _Connection(
        [
            {
                "table_name": "gke_nodes",
                "column_name": "embedding",
                "type_name": "vector(2)",
            }
        ]
    )

    with pytest.raises(PgVectorSchemaMismatchError) as exc_info:
        _backend_for_dimension_check(1024)._validate_vector_column_dimensions_sync(
            connection
        )

    error = exc_info.value
    assert connection.params == {
        "schema": "wiki",
        "table_names": ["gke_nodes", "gke_edges", "gke_documents", "gke_domains"],
    }
    assert error.expected_dimension == 1024
    assert error.mismatches[0].dimension == 2
    assert "wiki.gke_nodes.embedding is vector(2)" in str(error)
    assert "No data was written" in str(error)
    assert "re-embed" in str(error)


def test_pgvector_dimension_guard_accepts_matching_live_columns() -> None:
    connection = _Connection(
        [
            {
                "table_name": table_name,
                "column_name": "embedding",
                "type_name": "vector(1024)",
            }
            for table_name in ("gke_nodes", "gke_edges", "gke_documents", "gke_domains")
        ]
    )

    _backend_for_dimension_check(1024)._validate_vector_column_dimensions_sync(
        connection
    )


def test_pgvector_constructor_initializes_metadata_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        PgVectorBackend,
        "ensure_schema",
        lambda self: called.append(hasattr(self, "_md")),
    )

    backend = PgVectorBackend(engine=sa.create_engine("sqlite:///:memory:"), embedding_dim=2)
    try:
        assert hasattr(backend, "_md")
        assert called == [True]
    finally:
        backend.close()


def test_pgvector_async_storage_inspection_awaits_each_count() -> None:
    import asyncio

    backend = _backend_for_dimension_check(2)
    backend.nodes = sa.table("gke_nodes", sa.column("id"))
    backend.edges = sa.table("gke_edges", sa.column("id"))
    backend.documents = sa.table("gke_documents", sa.column("id"))
    backend.domains = sa.table("gke_domains", sa.column("id"))
    backend.engine = SimpleNamespace(
        url=SimpleNamespace(host="localhost", port=5432, database="wiki"),
        connect=lambda: _AsyncConnection(),
    )
    state = asyncio.run(backend.inspect_embedding_storage_async())

    assert state.vector_count == 10
    assert state.details == (
        "gke_nodes=1",
        "gke_edges=2",
        "gke_documents=3",
        "gke_domains=4",
    )


def test_pgvector_scope_ignores_credentials_and_driver() -> None:
    def make(url: str) -> PgVectorBackend:
        backend = _backend_for_dimension_check(2)
        backend.schema = "wiki"
        backend.engine = SimpleNamespace(url=sa.make_url(url))
        return backend

    first = make("postgresql+psycopg://alice:secret@db.example/wiki")
    second = make("postgresql+asyncpg://bob:other@db.example:5432/wiki?sslmode=require")

    assert first.embedding_storage_scope() == second.embedding_storage_scope()
