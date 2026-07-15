from __future__ import annotations

from pathlib import Path
import sqlite3
import uuid
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def _pg(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _create(run_id: str, **more: Any) -> dict[str, Any]:
    return {
        "kind": "create_server_run",
        "run_id": run_id,
        "conversation_id": more.get("conversation_id", "c"),
        "workflow_id": more.get("workflow_id", "w"),
        "user_id": more.get("user_id"),
        "user_turn_node_id": more.get("user_turn_node_id", "u"),
        "status": more.get("status", "queued"),
    }


def _assert_shape(row: dict[str, Any], run_id: str) -> None:
    assert set(row) == {
        "run_id", "conversation_id", "workflow_id", "user_id", "user_turn_node_id",
        "assistant_turn_node_id", "status", "cancel_requested", "result", "error",
        "created_at_ms", "updated_at_ms", "started_at_ms", "finished_at_ms", "terminal",
    }
    assert row["run_id"] == run_id
    assert row["assistant_turn_node_id"] is None
    assert row["result"] is None and row["error"] is None
    assert row["started_at_ms"] is None and row["finished_at_ms"] is None
    assert row["status"] == "queued" and row["terminal"] is False


@pytest.mark.ci
def test_sqlite_python_rust_bidirectional_run_registry_and_atomic_batch(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    python.create_server_run(run_id="py", conversation_id="c", workflow_id="w", user_id="user", user_turn_node_id="u")
    # Run remaining exercise with a fresh Python-created id omitted from helper.
    path = python.db_path
    _assert_shape(_sqlite(path, {"kind": "get_server_run", "run_id": "py"}), "py")
    for run in ("rust-z", "rust-a"):
        assert _sqlite(path, _create(run, conversation_id="c2", workflow_id="w2")) is None
    rows = _sqlite(path, {"kind": "list_server_runs", "conversation_id": "c2", "limit": 10})
    assert rows == sorted(
        rows, key=lambda row: (row["created_at_ms"], row["run_id"]), reverse=True
    )
    assert len(_sqlite(path, {"kind": "list_server_runs", "workflow_id": "w2", "limit": 1})) == 1
    assert _sqlite(path, {"kind": "list_server_runs", "status": "absent", "limit": 10}) == []
    event = _sqlite(path, {"kind": "append_server_run_event", "run_id": "py", "event_type": "one", "payload_json": ' {"x": 1} '})
    other = _sqlite(path, {"kind": "append_server_run_event", "run_id": "rust-z", "event_type": "other", "payload_json": "{}"})
    second = _sqlite(path, {"kind": "append_server_run_event", "run_id": "py", "event_type": "two", "payload_json": "null"})
    assert event["seq"] < other["seq"] < second["seq"]
    assert _sqlite(path, {"kind": "list_server_run_events", "run_id": "py", "after_seq": event["seq"], "limit": 1}) == [second]
    assert python.list_server_run_events("py")[0]["payload"] == {"x": 1}
    assert _sqlite(path, {"kind": "update_server_run", "run_id": "py", "status": "succeeded", "result_json": ' {"result": 1} ', "error_json": None, "assistant_turn_node_id": None, "started_at_ms": None, "finished_at_ms": 7}) is None
    assert _sqlite(path, {"kind": "get_server_run", "run_id": "py"})["result"] == {"result": 1}
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT result_json, error_json FROM server_runs WHERE run_id = 'py'").fetchone() == (' {"result": 1} ', None)
    assert _sqlite(path, {"kind": "request_server_run_cancel", "run_id": "py"}) is None
    assert _sqlite(path, {"kind": "get_server_run", "run_id": "py"})["status"] == "succeeded"
    assert _sqlite(path, {"kind": "request_server_run_cancel", "run_id": "rust-z"}) is None
    assert _sqlite(path, {"kind": "get_server_run", "run_id": "rust-z"})["status"] == "cancelling"
    assert _sqlite(path, {"kind": "get_server_run", "run_id": "missing"}) is None
    assert _sqlite(path, {"kind": "update_server_run", "run_id": "missing", "status": "x", "assistant_turn_node_id": None, "result_json": None, "error_json": None, "started_at_ms": None, "finished_at_ms": None}) is None
    with pytest.raises(RustParityError) as duplicate:
        _sqlite(path, _create("py"))
    assert duplicate.value.code == "KOGWISTAR_STORE_PERSISTENCE_FAILED"
    with pytest.raises(RustParityError) as rollback:
        _sqlite(path, {"kind": "batch", "operations": [_create("rolled"), {"kind": "append_server_run_event", "run_id": "rolled", "event_type": "x", "payload_json": "{}"}], "abort": True})
    assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _sqlite(path, {"kind": "get_server_run", "run_id": "rolled"}) is None
    kept = _sqlite(path, {"kind": "batch", "operations": [_create("kept"), {"kind": "append_server_run_event", "run_id": "kept", "event_type": "x", "payload_json": "{}"}, {"kind": "raw_append", "namespace": "n", "event_id": "e", "entity_kind": "n", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"}]})
    assert kept[1]["run_id"] == "kept" and python.get_server_run("kept") is not None
    assert event["seq"] > 0


def test_postgres_run_registry_bidirectional_isolated_and_atomic(sa_engine, pg_dsn: str | None, pg_schema: str | None) -> None:
    if sa_engine is None or pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    schema = f"gke_run_registry_{uuid.uuid4().hex}"
    other = f"gke_run_registry_{uuid.uuid4().hex}"
    try:
        python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
        python.ensure_initialized()
        python.create_server_run(run_id="py", conversation_id="c", workflow_id="w", user_id=None, user_turn_node_id="u")
        _assert_shape(_pg(pg_dsn, schema, {"kind": "get_server_run", "run_id": "py"}), "py")
        assert _pg(pg_dsn, schema, _create("rust")) is None
        assert python.get_server_run("rust") is not None
        assert len(_pg(pg_dsn, schema, {"kind": "list_server_runs", "workflow_id": "w", "limit": 1})) == 1
        assert _pg(pg_dsn, schema, {"kind": "list_server_runs", "status": "absent", "limit": 10}) == []
        event = _pg(pg_dsn, schema, {"kind": "append_server_run_event", "run_id": "rust", "event_type": "x", "payload_json": ' {"raw":true} '})
        next_event = _pg(pg_dsn, schema, {"kind": "append_server_run_event", "run_id": "py", "event_type": "y", "payload_json": "null"})
        assert event["seq"] < next_event["seq"]
        assert _pg(pg_dsn, schema, {"kind": "list_server_run_events", "run_id": "rust", "after_seq": event["seq"], "limit": 10}) == []
        assert python.list_server_run_events("rust")[0]["payload"] == {"raw": True}
        assert _pg(pg_dsn, schema, {"kind": "update_server_run", "run_id": "rust", "status": "succeeded", "assistant_turn_node_id": None, "result_json": ' {"r": 1} ', "error_json": None, "started_at_ms": None, "finished_at_ms": 9}) is None
        assert _pg(pg_dsn, schema, {"kind": "request_server_run_cancel", "run_id": "rust"}) is None
        rust = _pg(pg_dsn, schema, {"kind": "get_server_run", "run_id": "rust"})
        assert rust["status"] == "succeeded" and rust["result"] == {"r": 1}
        with sa_engine.connect() as conn:
            assert conn.exec_driver_sql(
                f'SELECT result_json, error_json FROM "{schema}".server_runs WHERE run_id = \'rust\''
            ).one() == (' {"r": 1} ', None)
        assert _pg(pg_dsn, schema, {"kind": "get_server_run", "run_id": "missing"}) is None
        with pytest.raises(RustParityError):
            _pg(pg_dsn, schema, _create("rust"))
        with pytest.raises(RustParityError) as rollback:
            _pg(pg_dsn, schema, {"kind": "batch", "operations": [_create("rolled"), {"kind": "append_server_run_event", "run_id": "rolled", "event_type": "x", "payload_json": "{}"}], "abort": True})
        assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
        assert _pg(pg_dsn, schema, {"kind": "get_server_run", "run_id": "rolled"}) is None
        assert _pg(pg_dsn, schema, {"kind": "batch", "operations": [_create("kept"), {"kind": "append_server_run_event", "run_id": "kept", "event_type": "x", "payload_json": "{}"}, {"kind": "raw_append", "namespace": "n", "event_id": "e", "entity_kind": "n", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"}]})[1]["run_id"] == "kept"
        assert _pg(pg_dsn, other, {"kind": "ensure_schema"}) == {"initialized": True}
        assert _pg(pg_dsn, other, {"kind": "get_server_run", "run_id": "rust"}) is None
        assert event["seq"] > 0
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{other}" CASCADE')
