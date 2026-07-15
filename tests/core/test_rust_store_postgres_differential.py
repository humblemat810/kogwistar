from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import Any

import pytest

pytest.importorskip("sqlalchemy")

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore


pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _rust(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def test_python_created_schema_rust_reads_writes_and_replay_maps_exclusive(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    python.ensure_initialized()
    assert python.append_entity_event(
        namespace="alpha",
        event_id="python-event",
        entity_kind="node",
        entity_id="n1",
        op="UPSERT",
        payload_json=' {"kept": true, "text":"雪"} ',
    ) == 1
    python.cursor_set(namespace="alpha", consumer="sink", last_seq=1)

    replay = _rust(
        dsn,
        schema,
        {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 10},
    )
    assert replay[0]["event_id"] == "python-event"
    assert replay[0]["payload_json"] == ' {"kept": true, "text":"雪"} '
    assert _rust(dsn, schema, {"kind": "cursor_get", "namespace": "alpha", "consumer": "sink"})[
        "last_seq"
    ] == 1

    appended = _rust(
        dsn,
        schema,
        {
            "kind": "raw_append",
            "namespace": "alpha",
            "event_id": "rust-event",
            "entity_kind": "edge",
            "entity_id": "e1",
            "op": "UPSERT",
            "payload_json": '{ "raw" : [1, 2] }',
        },
    )
    assert appended["seq"] == 2
    assert appended["event"]["payload_json"] == '{ "raw" : [1, 2] }'

    # Python iterates inclusive from_seq; native raw replay starts strictly after after_seq.
    assert list(python.iter_entity_events(namespace="alpha", from_seq=2)) == [
        (2, "edge", "e1", "UPSERT", '{ "raw" : [1, 2] }')
    ]
    assert _rust(
        dsn,
        schema,
        {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 1, "limit": 10},
    )[0]["seq"] == 2


def test_rust_created_schema_python_reads_writes_and_cursor_contracts(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, _ = _require_pg(pg_dsn, pg_schema)
    schema = f"gke_rust_created_{uuid.uuid4().hex}"
    try:
        assert _rust(dsn, schema, {"kind": "ensure_schema"}) == {"initialized": True}
        assert _rust(
            dsn,
            schema,
            {
                "kind": "raw_append",
                "namespace": "alpha",
                "event_id": "rust-first",
                "entity_kind": "node",
                "entity_id": "n1",
                "op": "UPSERT",
                "payload_json": '{"from":"rust"}',
            },
        )["seq"] == 1

        # Python bootstrap may add its wider metadata schema, but must retain Rust rows.
        python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
        python.ensure_initialized()
        assert list(python.iter_entity_events(namespace="alpha", from_seq=1)) == [
            (1, "node", "n1", "UPSERT", '{"from":"rust"}')
        ]
        assert python.append_entity_event(
            namespace="alpha",
            event_id="python-second",
            entity_kind="node",
            entity_id="n2",
            op="UPSERT",
            payload_json='{"from":"python"}',
        ) == 2

        assert _rust(
            dsn,
            schema,
            {"kind": "legacy_cursor_set", "namespace": "alpha", "consumer": "sink", "last_seq": 99},
        )["last_seq"] == 99
        assert _rust(
            dsn,
            schema,
            {"kind": "strict_cursor_advance", "namespace": "alpha", "consumer": "strict", "last_seq": 2},
        )["last_seq"] == 2
        with pytest.raises(RustParityError) as regresses:
            _rust(
                dsn,
                schema,
                {"kind": "strict_cursor_advance", "namespace": "alpha", "consumer": "strict", "last_seq": 1},
            )
        assert regresses.value.code == "KOGWISTAR_STORE_CURSOR_REGRESSES"
        with pytest.raises(RustParityError) as out_of_range:
            _rust(
                dsn,
                schema,
                {"kind": "strict_cursor_advance", "namespace": "alpha", "consumer": "strict", "last_seq": 3},
            )
        assert out_of_range.value.code == "KOGWISTAR_STORE_CURSOR_OUT_OF_RANGE"
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def test_rust_postgres_concurrent_idempotency_collision_and_atomic_rollback(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()

    def append(index: int) -> int:
        return _rust(
            dsn,
            schema,
            {
                "kind": "raw_append",
                "namespace": "concurrent",
                "event_id": f"concurrent-{index}",
                "entity_kind": "node",
                "entity_id": str(index),
                "op": "UPSERT",
                "payload_json": "{}",
            },
        )["seq"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences = sorted(pool.map(append, range(8)))
    assert sequences == list(range(1, 9))

    first = _rust(
        dsn,
        schema,
        {
            "kind": "raw_append",
            "namespace": "one",
            "event_id": "same-id",
            "entity_kind": "node",
            "entity_id": "n1",
            "op": "UPSERT",
            "payload_json": '{"original":true}',
        },
    )
    retry = _rust(
        dsn,
        schema,
        {
            "kind": "raw_append",
            "namespace": "one",
            "event_id": "same-id",
            "entity_kind": "changed",
            "entity_id": "changed",
            "op": "NOPE",
            "payload_json": "{}",
        },
    )
    assert retry["inserted"] is False
    assert retry["event"] == first["event"]
    with pytest.raises(RustParityError) as collision:
        _rust(
            dsn,
            schema,
            {
                "kind": "raw_append",
                "namespace": "two",
                "event_id": "same-id",
                "entity_kind": "node",
                "entity_id": "n2",
                "op": "UPSERT",
                "payload_json": "{}",
            },
        )
    assert collision.value.code == "KOGWISTAR_STORE_EVENT_ID_NAMESPACE_COLLISION"

    with pytest.raises(RustParityError) as rollback:
        _rust(
            dsn,
            schema,
            {
                "kind": "batch",
                "operations": [
                    {
                        "kind": "raw_append",
                        "namespace": "rollback",
                        "event_id": "rolled-back",
                        "entity_kind": "node",
                        "entity_id": "n",
                        "op": "UPSERT",
                        "payload_json": "{}",
                    },
                    {"kind": "legacy_cursor_set", "namespace": "rollback", "consumer": "sink", "last_seq": 1},
                ],
                "abort": True,
            },
        )
    assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "rollback"}) == 0
    assert _rust(dsn, schema, {"kind": "cursor_get", "namespace": "rollback", "consumer": "sink"})[
        "last_seq"
    ] == 0


def test_rust_postgres_schema_isolation_and_rejects_unknown_fields(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    other_schema = f"gke_rust_{uuid.uuid4().hex}"
    try:
        EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()
        _rust(
            dsn,
            schema,
            {
                "kind": "raw_append",
                "namespace": "isolated",
                "event_id": "primary-only",
                "entity_kind": "node",
                "entity_id": "n",
                "op": "UPSERT",
                "payload_json": "{}",
            },
        )
        EnginePostgresMetaStore(engine=sa_engine, schema=other_schema).ensure_initialized()
        assert _rust(
            dsn, other_schema, {"kind": "latest_retained_event_seq", "namespace": "isolated"}
        ) == 0
        assert _rust(
            dsn,
            other_schema,
            {
                "kind": "raw_append",
                "namespace": "isolated",
                "event_id": "other-only",
                "entity_kind": "node",
                "entity_id": "n",
                "op": "UPSERT",
                "payload_json": "{}",
            },
        )["seq"] == 1
        assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "isolated"}) == 1
        with pytest.raises(RustParityError) as invalid:
            _rust(dsn, schema, {"kind": "ensure_schema", "unexpected": True})
        assert invalid.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"
    finally:
        # UUID-derived identifier is trusted and isolated to this test.
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{other_schema}" CASCADE')
