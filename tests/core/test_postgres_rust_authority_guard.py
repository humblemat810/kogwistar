from __future__ import annotations

from typing import Any

import pytest

from kogwistar.engine_core.postgres_backend import PgVectorBackend

pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(autouse=True)
def _python_authority_for_schema_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep pytest schema setup/teardown outside the Rust-authority check."""
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "python")


def test_raw_sqlalchemy_is_read_only_under_rust_selection_and_writable_for_python(
    monkeypatch: pytest.MonkeyPatch,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    import sqlalchemy as sa

    if pg_schema is None:
        pytest.skip("PostgreSQL schema fixture unavailable")

    schema = pg_schema
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()

    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    with sa_engine.begin() as connection:
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar_one() == "on"
        with pytest.raises(sa.exc.DBAPIError, match="read-only"):
            connection.execute(sa.text(f'UPDATE "{schema}".gke_nodes SET id = id'))

    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    with sa_engine.begin() as connection:
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar_one() == "off"
        connection.execute(sa.text(f'UPDATE "{schema}".gke_nodes SET id = id'))
