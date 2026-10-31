from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import inspect as sqlalchemy_inspect, text as sqlalchemy_text
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


@pytest.mark.ci_full
def test_python_and_rust_postgres_metadata_schemas_match_shared_tables(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    if sa_engine is None or pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    pytest.importorskip("kogwistar._rust")

    python_schema = f"{pg_schema}_python"
    rust_schema = f"{pg_schema}_rust"
    try:
        EnginePostgresMetaStore(engine=sa_engine, schema=python_schema).ensure_initialized()
        rust_result = store_postgres(
            dsn=pg_dsn,
            schema=rust_schema,
            operation={"kind": "ensure_schema"},
        )
        assert rust_result == {"initialized": True}

        inspector = sqlalchemy_inspect(sa_engine)

        def schema_signature(schema: str) -> dict[str, object]:
            tables = inspector.get_table_names(schema=schema)
            signature: dict[str, object] = {}
            for table in tables:
                columns = inspector.get_columns(table, schema=schema)
                primary_key = inspector.get_pk_constraint(table, schema=schema)
                primary_key_columns = set(primary_key.get("constrained_columns") or ())
                signature[table] = tuple(
                    sorted(
                        (
                            column["name"],
                            str(column["type"]).lower(),
                            bool(column["nullable"]),
                            column["name"] in primary_key_columns,
                        )
                        for column in columns
                    )
                )
            return signature

        python_signature = schema_signature(python_schema)
        rust_signature = schema_signature(rust_schema)
        assert python_signature.keys() == rust_signature.keys()
        assert python_signature == rust_signature
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{python_schema}" CASCADE')
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{rust_schema}" CASCADE')


@pytest.mark.ci_full
def test_python_and_rust_migrate_legacy_postgres_index_jobs_equivalently(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    if sa_engine is None or pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    pytest.importorskip("kogwistar._rust")

    python_schema = f"{pg_schema}_python_legacy"
    rust_schema = f"{pg_schema}_rust_legacy"
    legacy_index_jobs = """
        CREATE TABLE {schema}.index_jobs (
            job_id TEXT PRIMARY KEY,
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            index_kind TEXT NOT NULL,
            op TEXT NOT NULL,
            status TEXT NOT NULL,
            lease_until TIMESTAMPTZ NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NULL,
            payload_json TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        INSERT INTO {schema}.index_jobs (
            job_id, entity_kind, entity_id, index_kind, op, status, payload_json
        ) VALUES ('legacy-job', 'node', 'n1', 'node_embedding', 'UPSERT', 'PENDING', '{"legacy":true}');
    """
    try:
        with sa_engine.begin() as conn:
            for schema in (python_schema, rust_schema):
                conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
                conn.exec_driver_sql(
                    legacy_index_jobs.replace("{schema}", f'"{schema}"')
                )

        EnginePostgresMetaStore(engine=sa_engine, schema=python_schema).ensure_initialized()
        assert store_postgres(
            dsn=pg_dsn,
            schema=rust_schema,
            operation={"kind": "ensure_schema"},
        ) == {"initialized": True}

        inspector = sqlalchemy_inspect(sa_engine)

        def index_jobs_signature(schema: str) -> tuple[tuple[object, ...], ...]:
            primary_key = inspector.get_pk_constraint("index_jobs", schema=schema)
            primary_key_columns = set(primary_key.get("constrained_columns") or ())
            return tuple(
                sorted(
                    (
                        column["name"],
                        str(column["type"]).lower(),
                        bool(column["nullable"]),
                        column["name"] in primary_key_columns,
                    )
                    for column in inspector.get_columns("index_jobs", schema=schema)
                )
            )

        assert index_jobs_signature(python_schema) == index_jobs_signature(rust_schema)
        expected_row = ("default", "", None, 10, 0, '{"legacy":true}')
        for schema in (python_schema, rust_schema):
            with sa_engine.connect() as conn:
                row = conn.execute(
                    sqlalchemy_text(
                        f'SELECT namespace, coalesce_key, next_run_at, max_retries, '
                        f'claim_attempts, payload_json FROM "{schema}".index_jobs '
                        "WHERE job_id='legacy-job'"
                    )
                ).one()
            assert tuple(row) == expected_row
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{python_schema}" CASCADE')
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{rust_schema}" CASCADE')
