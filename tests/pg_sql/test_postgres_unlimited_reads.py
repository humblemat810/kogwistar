from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest

sa = pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.postgres_backend import PgVectorBackend


pytestmark = [pytest.mark.ci, pytest.mark.regression]


def _table() -> sa.Table:
    metadata = sa.MetaData()
    return sa.Table(
        "items",
        metadata,
        sa.Column("id", sa.String),
        sa.Column("document", sa.Text),
        sa.Column("metadata", sa.JSON),
    )


def test_get_flat_omits_limit_for_unlimited_reads() -> None:
    table = _table()
    backend = object.__new__(PgVectorBackend)
    backend.numeric_keys = set()
    statement = None

    class Result:
        def fetchall(self):
            return [SimpleNamespace(id="one", document="text", metadata={})]

    class Connection:
        def execute(self, query):
            nonlocal statement
            statement = query
            return Result()

    @contextmanager
    def connection():
        yield Connection()

    backend._conn = connection
    result = backend._get_flat(
        table,
        ids=None,
        where=None,
        include=["documents"],
        limit=None,
    )

    assert result["ids"] == ["one"]
    assert statement is not None
    assert statement._limit_clause is None


@pytest.mark.asyncio
async def test_get_flat_async_omits_limit_for_unlimited_reads() -> None:
    table = _table()
    backend = object.__new__(PgVectorBackend)
    backend.numeric_keys = set()
    statement = None

    class Result:
        def fetchall(self):
            return [SimpleNamespace(id="one", document="text", metadata={})]

    class Connection:
        async def execute(self, query):
            nonlocal statement
            statement = query
            return Result()

    @asynccontextmanager
    async def connection():
        yield Connection()

    backend._async_conn = connection
    result = await backend._get_flat_async(
        table,
        ids=None,
        where=None,
        include=["documents"],
        limit=None,
    )

    assert result["ids"] == ["one"]
    assert statement is not None
    assert statement._limit_clause is None


@pytest.mark.parametrize("limit", [1, 25])
def test_get_flat_retains_explicit_limit(limit: int) -> None:
    table = _table()
    backend = object.__new__(PgVectorBackend)
    backend.numeric_keys = set()
    statement = None

    class Result:
        def fetchall(self):
            return []

    class Connection:
        def execute(self, query):
            nonlocal statement
            statement = query
            return Result()

    @contextmanager
    def connection():
        yield Connection()

    backend._conn = connection
    backend._get_flat(table, ids=None, where=None, include=[], limit=limit)

    assert statement is not None
    assert statement._limit_clause is not None
    assert statement._limit_clause.value == limit
