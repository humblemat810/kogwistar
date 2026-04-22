from __future__ import annotations

import asyncio

import pytest

from kogwistar.engine_core.postgres_backend import (
    AsyncPostgresUnitOfWork,
    get_active_conn,
)


class _FakeAsyncConn:
    def __init__(self, name: str):
        self.name = name
        self.pending: list[str] = []


class _FakeBeginCtx:
    def __init__(self, engine: "_FakeAsyncEngine"):
        self._engine = engine

    async def __aenter__(self):
        self._engine.begin_count += 1
        conn = _FakeAsyncConn(f"conn-{self._engine.begin_count}")
        self._engine.conns.append(conn)
        return conn

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._engine.committed.extend(self._engine.conns[-1].pending)
        return False


class _FakeAsyncEngine:
    def __init__(self):
        self.begin_count = 0
        self.conns: list[_FakeAsyncConn] = []
        self.committed: list[str] = []

    def begin(self):
        return _FakeBeginCtx(self)


@pytest.mark.asyncio
async def test_async_postgres_uow_nested_transaction_joins_outer_scope():
    engine = _FakeAsyncEngine()
    uow = AsyncPostgresUnitOfWork(engine=engine)

    async with uow.transaction():
        outer_conn = get_active_conn()
        assert outer_conn is not None
        async with uow.transaction():
            assert get_active_conn() is outer_conn

    assert engine.begin_count == 1
    assert len(engine.conns) == 1


@pytest.mark.asyncio
async def test_async_postgres_uow_task_local_context_isolated_across_tasks():
    engine = _FakeAsyncEngine()
    uow = AsyncPostgresUnitOfWork(engine=engine)
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()
    first_conn: list[object] = []
    second_conn: list[object] = []

    async def _first():
        async with uow.transaction():
            first_conn.append(get_active_conn())
            first_ready.set()
            await second_ready.wait()
            assert get_active_conn() is first_conn[0]

    async def _second():
        await first_ready.wait()
        async with uow.transaction():
            second_conn.append(get_active_conn())
            second_ready.set()
            await asyncio.sleep(0)
            assert get_active_conn() is second_conn[0]

    await asyncio.gather(_first(), _second())

    assert engine.begin_count == 2
    assert first_conn and second_conn
    assert first_conn[0] is not second_conn[0]


@pytest.mark.asyncio
async def test_async_postgres_uow_rollback_in_one_task_does_not_touch_other_commit():
    engine = _FakeAsyncEngine()
    uow = AsyncPostgresUnitOfWork(engine=engine)
    first_ready = asyncio.Event()
    second_done = asyncio.Event()

    async def _write(marker: str):
        conn = get_active_conn()
        assert conn is not None
        conn.pending.append(marker)

    async def _first():
        with pytest.raises(RuntimeError):
            async with uow.transaction():
                await _write("first")
                first_ready.set()
                await second_done.wait()
                raise RuntimeError("boom")

    async def _second():
        await first_ready.wait()
        async with uow.transaction():
            await _write("second")
        second_done.set()

    await asyncio.gather(_first(), _second())

    assert engine.committed == ["second"]
