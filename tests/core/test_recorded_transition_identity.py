from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite

pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_metadata_stores_reject_duplicate_recorded_transition_ids(
    store_kind: str, tmp_path: Path
) -> None:
    store = (
        InMemoryMetaStore()
        if store_kind == "memory"
        else EngineSQLite(tmp_path / "sqlite")
    )
    if isinstance(store, EngineSQLite):
        store.ensure_initialized()

    payload = '{"transition_id":"transition-1","kind":"start"}'
    first = store.append_server_run_event(
        "run-1", "workflow.recorded_transition.v1", payload
    )
    assert first["seq"] == 1

    expected_error = ValueError if store_kind == "memory" else sqlite3.IntegrityError
    with pytest.raises(expected_error):
        store.append_server_run_event(
            "run-2", "workflow.recorded_transition.v1", payload
        )

    assert store.list_server_run_events("run-1") == [first]
    assert store.list_server_run_events("run-2") == []


def test_non_runtime_events_do_not_share_transition_id_uniqueness(
    tmp_path: Path,
) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    for run_id in ("run-1", "run-2"):
        store.append_server_run_event(
            run_id,
            "application.custom_event",
            '{"transition_id":"transition-1"}',
        )
    assert len(store.list_server_run_events("run-1")) == 1
    assert len(store.list_server_run_events("run-2")) == 1


@pytest.mark.requires_rust
def test_python_and_rust_sqlite_metadata_schemas_match(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    python = EngineSQLite(tmp_path / "python")
    python.ensure_initialized()
    rust = RustEngineSQLite(tmp_path / "rust")
    rust.ensure_initialized()
    rust.close()

    def schema_signature(path: Path) -> dict[str, object]:
        with sqlite3.connect(path) as conn:
            table_names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            result: dict[str, object] = {}
            for table in table_names:
                columns = tuple(
                    tuple(row)
                    for row in conn.execute(f'PRAGMA table_info("{table}")')
                )
                indexes = []
                for row in conn.execute(f'PRAGMA index_list("{table}")'):
                    index_name = row[1]
                    sql_row = conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
                        (index_name,),
                    ).fetchone()
                    sql = " ".join((sql_row[0] or "").split()).lower()
                    sql = re.sub(r"\s*([(),=])\s*", r"\1", sql)
                    indexes.append((index_name, row[2], row[3], row[4], sql))
                result[table] = (columns, tuple(sorted(indexes)))
            return result

    assert schema_signature(python.db_path) == schema_signature(rust.db_path)


@pytest.mark.ci_full
def test_postgres_metadata_store_rejects_duplicate_recorded_transition_ids(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    if sa_engine is None or pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    pytest.importorskip("kogwistar._rust")
    schema = pg_schema
    store = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    store.ensure_initialized()
    rust_payload = '{"transition_id":"transition-rust","kind":"start"}'
    rust_event = store_postgres(
        dsn=pg_dsn,
        schema=schema,
        operation={
            "kind": "append_server_run_event",
            "run_id": "run-rust",
            "event_type": "workflow.recorded_transition.v1",
            "payload_json": rust_payload,
        },
    )
    assert isinstance(rust_event, dict)
    rust_seq = rust_event.get("seq")
    assert isinstance(rust_seq, int) and rust_seq > 0
    with pytest.raises(IntegrityError):
        store.append_server_run_event(
            "run-python-retry", "workflow.recorded_transition.v1", rust_payload
        )
    python_payload = '{"transition_id":"transition-python","kind":"start"}'
    python_event = store.append_server_run_event(
        "run-python", "workflow.recorded_transition.v1", python_payload
    )
    with pytest.raises(RustParityError):
        store_postgres(
            dsn=pg_dsn,
            schema=schema,
            operation={
                "kind": "append_server_run_event",
                "run_id": "run-rust-retry",
                "event_type": "workflow.recorded_transition.v1",
                "payload_json": python_payload,
            },
        )
    assert store.list_server_run_events("run-rust") == [rust_event]
    assert store.list_server_run_events("run-python") == [python_event]
