from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
import uuid

import pytest
import sqlalchemy as sa

pytest.importorskip("sqlalchemy")

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _write(kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "namespace": "proj",
        "key": "k",
        "payload": {"z": "雪😀", "a": [True, {"β": "值"}]},
        "last_authoritative_seq": 7,
        "last_materialized_seq": 6,
        "projection_schema_version": 3,
        "materialization_status": "ready",
        **extra,
    }


def _assert_projection_contract(
    call: Callable[[dict[str, Any]], Any],
    read_python: Callable[[str, str], dict[str, Any] | None],
    list_python: Callable[[str], list[dict[str, Any]]],
    raw_payload: Callable[[str, str], str],
) -> None:
    assert call({"kind": "get_named_projection", "namespace": "proj", "key": "missing"}) is None
    assert call(_write("compare_and_swap_named_projection", expected_last_authoritative_seq=None, expected_last_materialized_seq=None)) is True
    assert call(_write("compare_and_swap_named_projection", expected_last_authoritative_seq=None, expected_last_materialized_seq=None)) is False
    row = read_python("proj", "k")
    assert row is not None
    assert row["payload"] == {"a": [True, {"β": "值"}], "z": "雪😀"}
    assert row["last_authoritative_seq"] == 7
    assert row["last_materialized_seq"] == 6
    assert row["updated_at_ms"] > 0
    assert raw_payload("proj", "k") == json.dumps(row["payload"], sort_keys=True, separators=(",", ":"))

    assert call(_write("compare_and_swap_named_projection", expected_last_authoritative_seq=0, expected_last_materialized_seq=0)) is False
    assert call(_write("compare_and_swap_named_projection", key="missing", expected_last_authoritative_seq=7, expected_last_materialized_seq=6)) is False
    assert call(
        _write(
            "compare_and_swap_named_projection",
            payload={"changed": "新"},
            expected_last_authoritative_seq=7,
            expected_last_materialized_seq=6,
            last_authoritative_seq=8,
            last_materialized_seq=8,
        )
    ) is True
    assert read_python("proj", "k")["payload"] == {"changed": "新"}  # type: ignore[index]

    call(_write("replace_named_projection", key="z"))
    call(_write("replace_named_projection", key="a"))
    assert [row["key"] for row in call({"kind": "list_named_projections", "namespace": "proj"})] == ["a", "k", "z"]
    assert [row["key"] for row in list_python("proj")] == ["a", "k", "z"]
    call(_write("replace_named_projection", namespace="other", key="a"))
    call({"kind": "clear_named_projection", "namespace": "proj", "key": "k"})
    assert read_python("proj", "k") is None
    call({"kind": "clear_projection_namespace", "namespace": "proj"})
    assert list_python("proj") == []
    assert read_python("other", "a") is not None


def test_sqlite_named_projections_python_and_rust_interoperate(tmp_path: Path) -> None:
    db = EngineSQLite(tmp_path)
    db.ensure_initialized()
    db.replace_named_projection(
        "py", "first", {"b": "雪", "a": 1}, last_authoritative_seq=1,
        last_materialized_seq=1, projection_schema_version=1, materialization_status="ready",
    )
    call = lambda operation: store_sqlite(path=db.db_path, operation=operation)
    assert call({"kind": "get_named_projection", "namespace": "py", "key": "first"})["payload"] == {"a": 1, "b": "雪"}

    def raw(namespace: str, key: str) -> str:
        with db.connect() as conn:
            return str(conn.execute("SELECT payload_json FROM named_projections WHERE namespace=? AND key=?", (namespace, key)).fetchone()[0])

    _assert_projection_contract(call, db.get_named_projection, db.list_named_projections, raw)


def test_sqlite_rust_created_projection_schema_python_reads_and_atomic_batch(tmp_path: Path) -> None:
    path = tmp_path / "rust.db"
    call = lambda operation: store_sqlite(path=path, operation=operation)
    call(_write("replace_named_projection", namespace="rust", key="one"))
    db = EngineSQLite(tmp_path, filename="rust.db")
    db.ensure_initialized()
    assert db.get_named_projection("rust", "one")["payload"]["z"] == "雪😀"  # type: ignore[index]

    committed = {
        "kind": "batch",
        "operations": [
            {"kind": "raw_append", "namespace": "atomic", "event_id": "commit", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"},
            _write("replace_named_projection", namespace="atomic", key="projection"),
        ],
    }
    assert call(committed)[0]["inserted"] is True
    assert db.get_latest_entity_event_seq(namespace="atomic") == 1
    assert db.get_named_projection("atomic", "projection") is not None
    aborted = {**committed, "abort": True}
    aborted["operations"] = [
        {"kind": "raw_append", "namespace": "atomic", "event_id": "rollback", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"},
        _write("replace_named_projection", namespace="atomic", key="rollback"),
    ]
    with pytest.raises(RustParityError) as exc:
        call(aborted)
    assert exc.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert db.get_latest_entity_event_seq(namespace="atomic") == 1
    assert db.get_named_projection("atomic", "rollback") is None


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def test_postgres_named_projections_python_and_rust_interoperate(sa_engine, pg_dsn: str | None, pg_schema: str | None) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    db = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    db.ensure_initialized()
    db.replace_named_projection("py", "first", {"b": "雪", "a": 1}, last_authoritative_seq=1, last_materialized_seq=1, projection_schema_version=1, materialization_status="ready")
    call = lambda operation: store_postgres(dsn=dsn, schema=schema, operation=operation)
    assert call({"kind": "get_named_projection", "namespace": "py", "key": "first"})["payload"] == {"a": 1, "b": "雪"}

    def raw(namespace: str, key: str) -> str:
        with db.transaction() as conn:
            return str(
                conn.execute(
                    sa.text(
                        f'SELECT payload_json FROM "{schema}".named_projections '
                        "WHERE namespace = :ns AND key = :key"
                    ),
                    {"ns": namespace, "key": key},
                ).fetchone()[0]
            )

    _assert_projection_contract(call, db.get_named_projection, db.list_named_projections, raw)


def test_postgres_rust_created_projection_schema_python_reads_and_atomic_batch(sa_engine, pg_dsn: str | None, pg_schema: str | None) -> None:
    dsn, _ = _require_pg(pg_dsn, pg_schema)
    schema = f"gke_projection_{uuid.uuid4().hex}"
    call = lambda operation: store_postgres(dsn=dsn, schema=schema, operation=operation)
    try:
        call({"kind": "ensure_schema"})
        call(_write("replace_named_projection", namespace="rust", key="one"))
        db = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
        db.ensure_initialized()
        assert db.get_named_projection("rust", "one")["payload"]["z"] == "雪😀"  # type: ignore[index]
        batch = {"kind": "batch", "operations": [
            {"kind": "raw_append", "namespace": "atomic", "event_id": "commit", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"},
            _write("replace_named_projection", namespace="atomic", key="projection"),
        ]}
        assert call(batch)[0]["inserted"] is True
        assert db.get_latest_entity_event_seq(namespace="atomic") == 1
        assert db.get_named_projection("atomic", "projection") is not None
        batch["abort"] = True
        batch["operations"] = [
            {"kind": "raw_append", "namespace": "atomic", "event_id": "rollback", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"},
            _write("replace_named_projection", namespace="atomic", key="rollback"),
        ]
        with pytest.raises(RustParityError) as exc:
            call(batch)
        assert exc.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
        assert db.get_latest_entity_event_seq(namespace="atomic") == 1
        assert db.get_named_projection("atomic", "rollback") is None
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
