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
