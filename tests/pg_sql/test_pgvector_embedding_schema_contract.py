from __future__ import annotations

import pytest

sa = pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.postgres_backend import (
    PgVectorBackend,
    PgVectorSchemaMismatchError,
    _parse_vector_dimension,
)


pytestmark = [pytest.mark.ci_full, pytest.mark.regression]


@pytest.mark.parametrize(
    ("type_name", "expected"),
    [
        ("vector(2)", 2),
        ("vector(1024)", 1024),
        ("vector", None),
        ("halfvec(1024)", None),
    ],
)
def test_pgvector_format_type_dimension_parser(type_name: str, expected: int | None) -> None:
    assert _parse_vector_dimension(type_name) == expected


def test_reopening_pgvector_schema_with_new_dimension_fails_before_writes(
    sa_engine,
    pg_schema,
) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixtures are unavailable")

    PgVectorBackend(engine=sa_engine, embedding_dim=2, schema=pg_schema)

    with pytest.raises(PgVectorSchemaMismatchError) as exc_info:
        PgVectorBackend(engine=sa_engine, embedding_dim=1024, schema=pg_schema)

    error = exc_info.value
    assert error.expected_dimension == 1024
    assert {item.table_name for item in error.mismatches} == {
        "gke_nodes",
        "gke_edges",
        "gke_documents",
        "gke_domains",
    }
    message = str(error)
    assert f"{pg_schema}.gke_nodes.embedding is vector(2)" in message
    assert "No data was written" in message
    assert "re-embed" in message

    with sa_engine.connect() as conn:
        stored_type = conn.execute(
            sa.text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                  FROM pg_catalog.pg_attribute AS a
                  JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid
                  JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
                 WHERE n.nspname = :schema
                   AND c.relname = 'gke_nodes'
                   AND a.attname = 'embedding'
                """
            ),
            {"schema": pg_schema},
        ).scalar_one()
    assert stored_type == "vector(2)"
