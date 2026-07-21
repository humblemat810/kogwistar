from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import pytest

pytest.importorskip("sqlalchemy")

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci_full, pytest.mark.core]

_SNAPSHOT_A = ' { "名" : "雪 ☃", "spacing" : [ 1,  2 ] } '
_SNAPSHOT_B = '\n{"newer": true, "emoji":"🧭"}\t'
_FORWARD = ' [ { "op" : "add", "path" : "/節" } ] '
_INVERSE = '{ "undo" : [ "節", "雪" ] }'


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def _postgres(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _snapshot_op(version: int, payload_json: str, schema_version: int = 1) -> dict[str, Any]:
    return {
        "kind": "put_workflow_design_snapshot",
        "workflow_id": "wf-雪",
        "version": version,
        "seq": version * 10,
        "payload_json": payload_json,
        "schema_version": schema_version,
    }


def _delta_op(version: int, schema_version: int = 1) -> dict[str, Any]:
    return {
        "kind": "put_workflow_design_delta",
        "workflow_id": "wf-雪",
        "version": version,
        "prev_version": version - 1,
        "target_seq": version * 10,
        "forward_json": _FORWARD,
        "inverse_json": _INVERSE,
        "schema_version": schema_version,
    }


def _projection_op() -> dict[str, Any]:
    return {
        "kind": "replace_named_projection",
        "namespace": "workflow_design",
        "key": "wf-雪",
        "payload": {"stable": "雪"},
        "last_authoritative_seq": 10,
        "last_materialized_seq": 10,
        "projection_schema_version": 1,
        "materialization_status": "ready",
    }


def _event_op(event_id: str, namespace: str = "branch") -> dict[str, Any]:
    return {
        "kind": "raw_append",
        "namespace": namespace,
        "event_id": event_id,
        "entity_kind": "workflow",
        "entity_id": "wf-雪",
        "op": "UPSERT",
        "payload_json": ' { "raw" : "雪" } ',
    }


def _assert_history_row(snapshot: dict[str, Any], *, version: int, payload_json: str) -> None:
    assert snapshot["workflow_id"] == "wf-雪"
    assert snapshot["version"] == version
    assert snapshot["seq"] == version * 10
    assert snapshot["payload_json"] == payload_json
    assert snapshot["schema_version"] == 1
    assert isinstance(snapshot["created_at_ms"], int)


def test_sqlite_workflow_history_both_directions_pruning_and_atomic_batches(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    path = python.db_path

    # Python-created DDL and raw TEXT values are visible unchanged to Rust.
    python.put_workflow_design_snapshot(
        workflow_id="wf-雪", version=1, seq=10, payload_json=_SNAPSHOT_A, schema_version=1
    )
    python.put_workflow_design_snapshot(
        workflow_id="wf-雪", version=2, seq=20, payload_json=_SNAPSHOT_B, schema_version=2
    )
    python.put_workflow_design_delta(
        workflow_id="wf-雪",
        version=1,
        prev_version=0,
        target_seq=10,
        forward_json=_FORWARD,
        inverse_json=_INVERSE,
        schema_version=1,
    )
    _assert_history_row(
        _sqlite(
            path,
            {
                "kind": "get_workflow_design_snapshot",
                "workflow_id": "wf-雪",
                "max_version": 99,
                "schema_version": 1,
            },
        ),
        version=1,
        payload_json=_SNAPSHOT_A,
    )
    assert _sqlite(
        path,
        {
            "kind": "get_workflow_design_snapshot",
            "workflow_id": "wf-雪",
            "max_version": 99,
            "schema_version": 9,
        },
    ) is None
    delta = _sqlite(
        path,
        {
            "kind": "get_workflow_design_delta",
            "workflow_id": "wf-雪",
            "version": 1,
            "schema_version": 1,
        },
    )
    assert delta["forward_json"] == _FORWARD
    assert delta["inverse_json"] == _INVERSE

    # Rust writes use same tables; Python observes upsert and exact raw text.
    assert _sqlite(path, _snapshot_op(1, _SNAPSHOT_B)) is None
    assert _sqlite(path, _delta_op(2)) is None
    assert python.get_workflow_design_snapshot(
        workflow_id="wf-雪", max_version=1, schema_version=1
    )["payload_json"] == _SNAPSHOT_B
    assert python.get_workflow_design_delta(
        workflow_id="wf-雪", version=2, schema_version=1
    )["forward_json"] == _FORWARD

    assert _sqlite(path, _event_op("keep"))["seq"] == 1
    assert _sqlite(path, _event_op("prune"))["seq"] == 2
    assert _sqlite(path, _event_op("other", namespace="other"))["seq"] == 1
    assert _sqlite(
        path, {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 1}
    ) == 1
    assert _sqlite(path, {"kind": "latest_retained_event_seq", "namespace": "branch"}) == 1
    assert _sqlite(path, {"kind": "latest_retained_event_seq", "namespace": "other"}) == 1

    committed = _sqlite(
        path,
        {
            "kind": "batch",
            "operations": [
                _event_op("committed"),
                _event_op("superseded"),
                {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 3},
                _projection_op(),
                _snapshot_op(3, _SNAPSHOT_A),
                _delta_op(3),
            ],
        },
    )
    assert committed[0]["seq"] == 3
    assert _sqlite(path, {"kind": "get_named_projection", "namespace": "workflow_design", "key": "wf-雪"})[
        "payload"
    ] == {"stable": "雪"}
    assert _sqlite(
        path,
        {"kind": "get_workflow_design_snapshot", "workflow_id": "wf-雪", "max_version": 3, "schema_version": 1},
    )["version"] == 3

    with pytest.raises(RustParityError) as rollback:
        _sqlite(
            path,
            {
                "kind": "batch",
                "operations": [
                    _event_op("rolled-back"),
                    {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 1},
                    _projection_op(),
                    _snapshot_op(4, _SNAPSHOT_B),
                    _delta_op(4),
                ],
                "abort": True,
            },
        )
    assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _sqlite(path, {"kind": "latest_retained_event_seq", "namespace": "branch"}) == 3
    assert _sqlite(
        path,
        {"kind": "get_workflow_design_snapshot", "workflow_id": "wf-雪", "max_version": 4, "schema_version": 1},
    )["version"] == 3
    assert _sqlite(path, {"kind": "get_workflow_design_delta", "workflow_id": "wf-雪", "version": 4, "schema_version": 1}) is None

    assert _sqlite(path, {"kind": "clear_workflow_design_snapshots", "workflow_id": "wf-雪"}) is None
    assert _sqlite(path, {"kind": "clear_workflow_design_deltas", "workflow_id": "wf-雪"}) is None
    assert python.get_workflow_design_snapshot(workflow_id="wf-雪", max_version=99, schema_version=1) is None
    assert python.get_workflow_design_delta(workflow_id="wf-雪", version=1, schema_version=1) is None


def _require_pg(dsn: str | None, schema: str | None) -> tuple[str, str]:
    if dsn is None or schema is None:
        pytest.skip("live Testcontainers pgvector/pgvector:pg16 unavailable")
    return dsn, schema


def test_postgres_workflow_history_python_created_schema_and_atomic_batch(
    sa_engine: Any, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    python.ensure_initialized()
    python.put_workflow_design_snapshot(
        workflow_id="wf-雪", version=1, seq=10, payload_json=_SNAPSHOT_A, schema_version=1
    )
    python.put_workflow_design_delta(
        workflow_id="wf-雪", version=1, prev_version=0, target_seq=10,
        forward_json=_FORWARD, inverse_json=_INVERSE, schema_version=1,
    )
    _assert_history_row(
        _postgres(dsn, schema, {"kind": "get_workflow_design_snapshot", "workflow_id": "wf-雪", "max_version": 1, "schema_version": 1}),
        version=1,
        payload_json=_SNAPSHOT_A,
    )
    assert _postgres(dsn, schema, _snapshot_op(2, _SNAPSHOT_B)) is None
    assert _postgres(dsn, schema, _delta_op(2)) is None
    assert python.get_workflow_design_snapshot(workflow_id="wf-雪", max_version=2, schema_version=1)["payload_json"] == _SNAPSHOT_B

    assert _postgres(dsn, schema, _event_op("keep"))["seq"] == 1
    assert _postgres(dsn, schema, _event_op("prune"))["seq"] == 2
    assert _postgres(dsn, schema, _event_op("other", namespace="other"))["seq"] == 1
    assert _postgres(dsn, schema, {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 1}) == 1
    assert _postgres(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "branch"}) == 1
    assert _postgres(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "other"}) == 1

    assert _postgres(
        dsn,
        schema,
        {
            "kind": "batch",
            "operations": [
                _event_op("committed"),
                _event_op("superseded"),
                {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 3},
                _projection_op(),
                _snapshot_op(3, _SNAPSHOT_A),
                _delta_op(3),
            ],
        },
    )[0]["seq"] == 3
    with pytest.raises(RustParityError) as rollback:
        _postgres(
            dsn,
            schema,
            {
                "kind": "batch",
                "operations": [
                    _event_op("rolled-back"),
                    {"kind": "prune_entity_events_after", "namespace": "branch", "to_seq": 1},
                    _projection_op(),
                    _snapshot_op(4, _SNAPSHOT_B),
                    _delta_op(4),
                ],
                "abort": True,
            },
        )
    assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _postgres(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "branch"}) == 3
    assert _postgres(dsn, schema, {"kind": "get_workflow_design_snapshot", "workflow_id": "wf-雪", "max_version": 4, "schema_version": 1})["version"] == 3
    assert _postgres(dsn, schema, {"kind": "get_workflow_design_delta", "workflow_id": "wf-雪", "version": 4, "schema_version": 1}) is None


def test_postgres_rust_created_schema_python_reads_history(
    sa_engine: Any, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, _ = _require_pg(pg_dsn, pg_schema)
    schema = f"gke_history_{uuid.uuid4().hex}"
    try:
        assert _postgres(dsn, schema, {"kind": "ensure_schema"}) == {"initialized": True}
        assert _postgres(dsn, schema, _snapshot_op(1, _SNAPSHOT_A)) is None
        assert _postgres(dsn, schema, _delta_op(1)) is None
        python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
        python.ensure_initialized()
        assert python.get_workflow_design_snapshot(
            workflow_id="wf-雪", max_version=1, schema_version=1
        )["payload_json"] == _SNAPSHOT_A
        assert python.get_workflow_design_delta(
            workflow_id="wf-雪", version=1, schema_version=1
        )["inverse_json"] == _INVERSE
        assert _postgres(dsn, schema, {"kind": "clear_workflow_design_snapshots", "workflow_id": "wf-雪"}) is None
        assert _postgres(dsn, schema, {"kind": "clear_workflow_design_deltas", "workflow_id": "wf-雪"}) is None
        assert python.get_workflow_design_snapshot(workflow_id="wf-雪", max_version=1, schema_version=1) is None
        assert python.get_workflow_design_delta(workflow_id="wf-雪", version=1, schema_version=1) is None
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
