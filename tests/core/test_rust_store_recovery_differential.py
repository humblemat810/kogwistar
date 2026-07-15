from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def _postgres(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _event(*, entity_id: str, revision: int) -> str:
    """Full Node-shaped Python event payload, intentionally not Rust-specific."""
    return json.dumps(
        {
            "id": entity_id,
            "document": f"node {entity_id} revision {revision}",
            "metadata": {"revision": revision, "source": "python"},
            "groundings": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _recover(
    namespace: str = "alpha",
    consumer: str = "sink",
    projection_namespace: str = "entity_projection",
    projection_key: str = "main",
    batch_limit: int = 2,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": "recover_entity_projection",
        "namespace": namespace,
        "consumer": consumer,
        "projection_namespace": projection_namespace,
        "projection_key": projection_key,
        "batch_limit": batch_limit,
        **extra,
    }


def _rebuild(
    namespace: str = "alpha",
    consumer: str = "sink",
    projection_namespace: str = "entity_projection",
    projection_key: str = "main",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": "rebuild_entity_projection",
        "namespace": namespace,
        "consumer": consumer,
        "projection_namespace": projection_namespace,
        "projection_key": projection_key,
        **extra,
    }


def _assert_projection_shape(payload: dict[str, Any]) -> None:
    entities = payload["entities"]
    deleted = entities['["node","n1"]']
    assert deleted["deleted"] is True
    assert deleted["op"] == "DELETE"
    active = entities['["alien_kind","n2"]']
    assert active["deleted"] is False
    assert active["entity"]["id"] == "n2"


def _run_recovery_contract(
    *,
    append_python: Any,
    append_rust: Any,
    call: Any,
    projection: Any,
    cursor: Any,
    event_count: Any,
) -> None:
    """Run same recovery invariants over a Python-owned durable corpus."""
    append_python("py-add", "node", "n1", "ADD", _event(entity_id="n1", revision=1))
    append_python("py-replace", "node", "n1", "REPLACE", _event(entity_id="n1", revision=2))
    append_python("py-tombstone", "node", "n1", "TOMBSTONE", '{"reason":"gone"}')
    append_python("py-delete", "node", "n1", "DELETE", '{"reason":"confirmed"}')
    append_rust(
        "rust-add-unknown",
        "alien_kind",
        "n2",
        "ADD",
        _event(entity_id="n2", revision=1),
    )
    append_rust(
        "rust-other-namespace",
        "node",
        "other",
        "ADD",
        _event(entity_id="other", revision=1),
        namespace="beta",
    )
    assert event_count() == 5

    first = call(_recover())
    assert first["processed_count"] == 2
    assert (first["prior_cursor"], first["new_cursor"], first["caught_up"]) == (0, 2, False)
    assert cursor("sink") == 2

    beta = call(
        _recover(
            namespace="beta",
            projection_namespace="beta_entity_projection",
            projection_key="main",
        )
    )
    assert (beta["processed_count"], beta["prior_cursor"], beta["new_cursor"]) == (1, 0, 1)
    assert projection("beta_entity_projection", "main")["payload"]["entities"]

    # Fold and write a real next batch, then deliberately abort. The named
    # projection and consumer cursor must both remain at the first batch.
    before_abort = projection("entity_projection", "main")
    before_cursor = cursor("sink")
    with pytest.raises(RustParityError) as aborted:
        call(_recover(abort_after_projection=True))
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert projection("entity_projection", "main") == before_abort
    assert cursor("sink") == before_cursor

    # Every bridge call opens the durable store again; this is the restart edge.
    second = call(_recover())
    assert second["processed_count"] == 2
    assert (second["prior_cursor"], second["new_cursor"], second["caught_up"]) == (2, 4, False)
    assert cursor("sink") == 4

    third = call(_recover())
    assert third["processed_count"] == 1
    assert (third["prior_cursor"], third["new_cursor"], third["caught_up"]) == (4, 5, True)
    assert cursor("sink") == 5
    baseline = projection("entity_projection", "main")
    assert baseline is not None
    _assert_projection_shape(baseline["payload"])

    # Replaying a caught-up consumer is idempotent, including canonical bytes.
    idempotent = call(_recover())
    assert idempotent["processed_count"] == 0
    assert idempotent["canonical_payload"] == third["canonical_payload"]
    assert idempotent["digest"] == third["digest"]
    assert event_count() == 5

    # Same event namespace, independent consumer and target projection.
    isolated = call(_recover(consumer="audit", projection_key="audit", batch_limit=1))
    assert (isolated["prior_cursor"], isolated["new_cursor"]) == (0, 1)
    assert cursor("audit") == 1
    assert projection("entity_projection", "audit") is not None
    assert projection("entity_projection", "main") == baseline

    # Explicit rebuild changes no event rows and is byte-identical when corpus is unchanged.
    rebuilt = call(_rebuild())
    rebuilt_again = call(_rebuild())
    assert rebuilt["canonical_payload"] == rebuilt_again["canonical_payload"]
    assert rebuilt["digest"] == rebuilt_again["digest"]
    assert rebuilt_again["new_cursor"] == 5
    assert event_count() == 5

    # Invalid authoritative payload must roll back the named projection and cursor.
    append_rust("rust-bad", "node", "bad", "REPLACE", '{"id":"different"}')
    before_invalid = projection("entity_projection", "main")
    before_invalid_cursor = cursor("sink")
    with pytest.raises(RustParityError) as invalid:
        call(_recover())
    assert invalid.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"
    assert projection("entity_projection", "main") == before_invalid
    assert cursor("sink") == before_invalid_cursor

    with pytest.raises(RustParityError) as zero_limit:
        call(_recover(batch_limit=0))
    assert zero_limit.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"
    with pytest.raises(RustParityError) as unknown_field:
        call({**_recover(), "unexpected": True})
    assert unknown_field.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"
    with pytest.raises(RustParityError) as invalid_namespace:
        call(_recover(namespace="bad\x00namespace"))
    assert invalid_namespace.value.code == "KOGWISTAR_STORE_INVALID_ENTITY_EVENT"


def test_sqlite_atomic_bounded_recovery_rebuild_and_python_rust_interop(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    path = python.db_path

    def append_python(event_id: str, kind: str, entity_id: str, op: str, payload: str) -> None:
        assert (
            python.append_entity_event(
                namespace="alpha",
                event_id=event_id,
                entity_kind=kind,
                entity_id=entity_id,
                op=op,
                payload_json=payload,
            )
            > 0
        )

    def append_rust(
        event_id: str, kind: str, entity_id: str, op: str, payload: str, *, namespace: str = "alpha"
    ) -> None:
        _sqlite(
            path,
            {
                "kind": "raw_append",
                "namespace": namespace,
                "event_id": event_id,
                "entity_kind": kind,
                "entity_id": entity_id,
                "op": op,
                "payload_json": payload,
            },
        )

    def call(operation: dict[str, Any]) -> Any:
        return _sqlite(path, operation)

    def projection(namespace: str, key: str) -> Any:
        return _sqlite(path, {"kind": "get_named_projection", "namespace": namespace, "key": key})

    def cursor(consumer: str) -> int:
        return _sqlite(path, {"kind": "cursor_get", "namespace": "alpha", "consumer": consumer})[
            "last_seq"
        ]

    def event_count() -> int:
        return len(
            _sqlite(
                path,
                {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 100},
            )
        )

    _run_recovery_contract(
        append_python=append_python,
        append_rust=append_rust,
        call=call,
        projection=projection,
        cursor=cursor,
        event_count=event_count,
    )
    # Rust-created event remains Python-readable through existing public meta-store.
    assert [row[1] for row in python.iter_entity_events(namespace="alpha", from_seq=5)] == [
        "alien_kind",
        "node",
    ]


def test_postgres_atomic_bounded_recovery_rebuild_and_python_rust_interop(
    sa_engine: Any, pg_dsn: str | None, pg_schema: str | None
) -> None:
    pytest.importorskip("sqlalchemy")
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore

    python = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    python.ensure_initialized()
    prefix = uuid.uuid4().hex

    def append_python(event_id: str, kind: str, entity_id: str, op: str, payload: str) -> None:
        assert (
            python.append_entity_event(
                namespace="alpha",
                event_id=f"{prefix}-{event_id}",
                entity_kind=kind,
                entity_id=entity_id,
                op=op,
                payload_json=payload,
            )
            > 0
        )

    def append_rust(
        event_id: str, kind: str, entity_id: str, op: str, payload: str, *, namespace: str = "alpha"
    ) -> None:
        _postgres(
            pg_dsn,
            pg_schema,
            {
                "kind": "raw_append",
                "namespace": namespace,
                "event_id": f"{prefix}-{event_id}",
                "entity_kind": kind,
                "entity_id": entity_id,
                "op": op,
                "payload_json": payload,
            },
        )

    def call(operation: dict[str, Any]) -> Any:
        return _postgres(pg_dsn, pg_schema, operation)

    def projection(namespace: str, key: str) -> Any:
        return _postgres(
            pg_dsn, pg_schema, {"kind": "get_named_projection", "namespace": namespace, "key": key}
        )

    def cursor(consumer: str) -> int:
        return _postgres(
            pg_dsn, pg_schema, {"kind": "cursor_get", "namespace": "alpha", "consumer": consumer}
        )["last_seq"]

    def event_count() -> int:
        return len(
            _postgres(
                pg_dsn,
                pg_schema,
                {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 100},
            )
        )

    _run_recovery_contract(
        append_python=append_python,
        append_rust=append_rust,
        call=call,
        projection=projection,
        cursor=cursor,
        event_count=event_count,
    )
    assert [row[1] for row in python.iter_entity_events(namespace="alpha", from_seq=5)] == [
        "alien_kind",
        "node",
    ]
