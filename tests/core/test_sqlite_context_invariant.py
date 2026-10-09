from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from pathlib import Path

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.sqlite_context import (
    SQLiteContextError,
    acquire_sqlite_database,
    independent_sqlite_execution_context,
    select_sqlite_implementation,
    sqlite_execution_context,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]


def _assert_code(error: BaseException, code: str) -> None:
    assert getattr(error, "code", None) == code


def test_context_binding_is_immutable_and_nested_calls_inherit() -> None:
    with sqlite_execution_context("python"):
        assert select_sqlite_implementation("python").implementation == "python"
        with pytest.raises(SQLiteContextError) as mismatch:
            select_sqlite_implementation("rust")
        _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
        with pytest.raises(SQLiteContextError):
            with sqlite_execution_context("rust"):
                pass


def test_python_connection_closes_and_releases_database_ownership(tmp_path: Path) -> None:
    engine = EngineSQLite(tmp_path)
    path = engine.db_path
    engine.ensure_initialized()
    with sqlite_execution_context("python"):
        with engine.connect() as connection:
            connection.execute("CREATE TABLE close_check (value INTEGER)")
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    with sqlite_execution_context("rust"):
        lease = acquire_sqlite_database(path, "rust")
        lease.release()


def test_overlapping_python_and_rust_database_ownership_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    python_lease = acquire_sqlite_database(path, "python")
    try:
        with sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(path, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        python_lease.release()


def test_same_implementation_can_concurrently_acquire_same_database(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    first = acquire_sqlite_database(path, "python")
    second = acquire_sqlite_database(path, "python")
    second.release()
    first.release()


@pytest.mark.requires_rust
def test_same_implementation_async_contexts_share_database_normally(tmp_path: Path) -> None:
    async def scenario(implementation: str) -> list[int]:
        database = tmp_path / implementation
        engine = EngineSQLite(database)
        engine.ensure_initialized()
        rust_engine = RustEngineSQLite(database)
        with sqlite_execution_context("rust"):
            rust_engine.ensure_initialized()
        barrier = asyncio.Barrier(2)

        async def allocate() -> int:
            with independent_sqlite_execution_context(implementation):
                await barrier.wait()
                selected = engine if implementation == "python" else rust_engine
                return await asyncio.to_thread(selected.next_global_seq)

        return list(await asyncio.gather(allocate(), allocate()))

    for implementation in ("python", "rust"):
        assert sorted(asyncio.run(scenario(implementation))) == [1, 2]


@pytest.mark.parametrize("alias_kind", ["relative", "uri", "symlink", "hardlink"])
def test_database_aliases_share_ownership_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    path = tmp_path / "engine.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE identity_check (value INTEGER)")
    if alias_kind == "relative":
        monkeypatch.chdir(tmp_path)
        alias: str | Path = Path("engine.db")
    elif alias_kind == "uri":
        alias = f"file:{path.as_posix()}?mode=rw"
    elif alias_kind == "symlink":
        alias = tmp_path / "engine-link.db"
        try:
            os.symlink(path, alias)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links unavailable")
    else:
        alias = tmp_path / "engine-hardlink.db"
        try:
            os.link(path, alias)
        except OSError:
            pytest.skip("hard links unavailable")

    owner = acquire_sqlite_database(path, "python")
    try:
        with pytest.raises(SQLiteContextError) as conflict:
            acquire_sqlite_database(alias, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        owner.release()


def test_shared_memory_uri_query_order_is_canonicalized() -> None:
    owner = acquire_sqlite_database(
        "file:shared-memory?mode=memory&cache=shared", "python"
    )
    try:
        with pytest.raises(SQLiteContextError) as conflict:
            acquire_sqlite_database(
                "file:shared-memory?cache=shared&mode=memory", "rust"
            )
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        owner.release()


def test_database_replacement_is_rejected_while_old_identity_is_owned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "engine.db"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE old_database (value INTEGER)")
    connection.close()
    owner = acquire_sqlite_database(path, "python")
    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    with connection:
        connection.execute("CREATE TABLE new_database (value INTEGER)")
    connection.close()
    path.unlink()
    os.replace(replacement, path)
    try:
        with pytest.raises(SQLiteContextError) as changed:
            acquire_sqlite_database(path, "python")
        _assert_code(changed.value, "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED")
    finally:
        owner.release()


def test_database_deletion_is_rejected_while_identity_is_owned(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    connection = sqlite3.connect(path)
    connection.close()
    owner = acquire_sqlite_database(path, "python")
    path.unlink()
    try:
        with pytest.raises(SQLiteContextError) as changed:
            acquire_sqlite_database(path, "python")
        _assert_code(changed.value, "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED")
    finally:
        owner.release()


@pytest.mark.requires_rust
def test_async_child_inherits_binding_and_independent_contexts_run_in_parallel(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with sqlite_execution_context("python"):
            async def child_switch() -> str:
                try:
                    select_sqlite_implementation("rust")
                except SQLiteContextError as error:
                    return error.code
                raise AssertionError("child task silently changed SQLite implementation")

            assert await asyncio.create_task(child_switch()) == (
                "KOGWISTAR_SQLITE_ENGINE_MISMATCH"
            )

        barrier = asyncio.Barrier(2)

        async def operate(implementation: str, database: Path) -> None:
            with independent_sqlite_execution_context(implementation):
                database.parent.mkdir(parents=True, exist_ok=True)
                if implementation == "python":
                    engine = EngineSQLite(database.parent)
                    await asyncio.to_thread(engine.ensure_initialized)
                    await barrier.wait()
                    await asyncio.to_thread(engine.next_global_seq)
                else:
                    engine = RustEngineSQLite(database.parent)
                    await asyncio.to_thread(engine.ensure_initialized)
                    await barrier.wait()
                    await asyncio.to_thread(engine.next_global_seq)

        await asyncio.gather(
            operate("python", tmp_path / "python" / "engine.db"),
            operate("rust", tmp_path / "rust" / "engine.db"),
        )

        with sqlite_execution_context("python"):
            with independent_sqlite_execution_context("rust"):
                assert select_sqlite_implementation("rust").implementation == "rust"
            assert select_sqlite_implementation("python").implementation == "python"

    asyncio.run(scenario())


def test_executor_context_must_be_copied_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "executor.db"
    with sqlite_execution_context("python"):
        owner = acquire_sqlite_database(path, "python")
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            def unpropagated() -> None:
                lease = acquire_sqlite_database(path, "rust")
                lease.release()

            with pytest.raises(SQLiteContextError) as missing:
                executor.submit(unpropagated).result()
            _assert_code(
                missing.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT"
            )
            assert executor.submit(
                context.run, select_sqlite_implementation, "python"
            ).result().implementation == "python"
        owner.release()


def test_cancellation_does_not_release_live_connection_ownership(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    allow_close = threading.Event()
    finished = threading.Event()
    engine = EngineSQLite(tmp_path)
    engine.ensure_initialized()

    def hold_connection() -> None:
        try:
            with independent_sqlite_execution_context("python"):
                connection = engine.connect()
                started.set()
                try:
                    allow_close.wait(timeout=10)
                finally:
                    connection.close()
        finally:
            finished.set()

    async def scenario() -> None:
        pending = asyncio.create_task(asyncio.to_thread(hold_connection))
        assert await asyncio.to_thread(started.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with independent_sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(engine.db_path, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
        allow_close.set()
        assert await asyncio.to_thread(finished.wait, 10)
        with independent_sqlite_execution_context("rust"):
            lease = acquire_sqlite_database(engine.db_path, "rust")
            lease.release()

    asyncio.run(scenario())


@pytest.mark.requires_rust
def test_direct_native_entrypoint_cannot_bypass_context_guard(tmp_path: Path) -> None:
    from kogwistar import _rust

    path = tmp_path / "must-not-be-created.db"
    payload = json.dumps(
        {
            "path": str(path),
            "operation": {"kind": "open_init"},
        }
    )
    with sqlite_execution_context("python"):
        with pytest.raises(SQLiteContextError) as mismatch:
            _rust.store_sqlite_json(payload)
    _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
    assert not path.exists()


def test_rust_context_rejects_python_connection_before_open(tmp_path: Path) -> None:
    engine = EngineSQLite(tmp_path)
    with sqlite_execution_context("rust"):
        with pytest.raises(SQLiteContextError) as mismatch:
            engine.connect()
    _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
    assert not engine.db_path.exists()


@pytest.mark.requires_rust
def test_both_sqlite_libraries_may_be_loaded_for_disjoint_contexts(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    python_db = EngineSQLite(tmp_path / "python")
    python_db.ensure_initialized()
    with sqlite3.connect(python_db.db_path) as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)

    rust_path = tmp_path / "rust" / "engine.db"
    with sqlite_execution_context("rust"):
        result = _rust.store_sqlite_json(
            json.dumps({"path": str(rust_path), "operation": {"kind": "open_init"}})
        )
    assert json.loads(result) == {"initialized": True}


@pytest.mark.requires_rust
def test_direct_native_python_overlap_fails_before_database_mutation(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    python = EngineSQLite(tmp_path)
    path = python.db_path
    connection = python.connect()
    try:
        before = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        with sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                _rust.store_sqlite_json(
                    json.dumps({"path": str(path), "operation": {"kind": "open_init"}})
                )
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
        after = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        assert after == before
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.requires_rust
def test_native_transaction_id_cannot_cross_contexts(tmp_path: Path) -> None:
    from kogwistar._rust import RustStoreValueError

    from kogwistar import _rust

    path = tmp_path / "transaction.db"
    transaction_id = "same-transaction-token"

    def native(operation: dict[str, object]) -> str:
        return _rust.store_sqlite_json(
            json.dumps(
                {
                    "path": str(path),
                    "transaction_id": transaction_id,
                    "operation": operation,
                }
            )
        )

    with sqlite_execution_context("rust"):
        native({"kind": "begin_transaction"})
        other_context = Context()
        with pytest.raises(RustStoreValueError) as mismatch:
            other_context.run(
                lambda: _run_in_sqlite_context(
                    "rust",
                    lambda: native(
                        {
                            "kind": "raw_append",
                            "namespace": "ns",
                            "event_id": "event",
                            "entity_kind": "node",
                            "entity_id": "node",
                            "op": "UPSERT",
                            "payload_json": "{}",
                        }
                    ),
                )
            )
        _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
        native({"kind": "rollback_transaction"})


@pytest.mark.requires_rust
def test_rejected_nested_native_begin_does_not_release_live_session(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    path = tmp_path / "nested-transaction.db"
    def payload(kind: str) -> str:
        return json.dumps(
            {
                "path": str(path),
                "transaction_id": "outer-session",
                "operation": {"kind": kind},
            }
        )
    with sqlite_execution_context("rust"):
        _rust.store_sqlite_json(payload("begin_transaction"))
        with pytest.raises(SQLiteContextError) as nested:
            _rust.store_sqlite_json(payload("begin_transaction"))
        _assert_code(nested.value, "KOGWISTAR_SQLITE_TRANSACTION_ALREADY_ACTIVE")
        _rust.store_sqlite_json(payload("rollback_transaction"))


def _run_in_sqlite_context(implementation: str, function):
    with sqlite_execution_context(implementation):
        return function()
