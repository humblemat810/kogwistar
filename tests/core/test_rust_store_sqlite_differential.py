from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_sqlite
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _rust(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def test_python_sqlite_then_rust_reads_and_writes_actual_database(tmp_path: Path) -> None:
    db = EngineSQLite(tmp_path)
    db.ensure_initialized()
    path = db.db_path
    assert _rust(path, {"kind": "open_init"}) == {"initialized": True}

    assert db.next_global_seq() == 1
    assert db.next_user_seq("alice") == 1
    assert db.next_scoped_seq("scope-a") == 1
    assert db.append_entity_event(
        namespace="alpha",
        event_id="py-event",
        entity_kind="node",
        entity_id="n1",
        op="UPSERT",
        payload_json=' {"kept": true, "text":"雪"} ',
    ) == 1
    db.cursor_set(namespace="alpha", consumer="sink", last_seq=9)

    assert _rust(path, {"kind": "current_global_seq"}) == 1
    assert _rust(path, {"kind": "current_user_seq", "user_id": "alice"}) == 1
    assert _rust(path, {"kind": "current_scoped_seq", "scope_id": "scope-a"}) == 1
    replay = _rust(
        path,
        {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 0, "limit": 10},
    )
    assert replay[0]["event_id"] == "py-event"
    assert replay[0]["payload_json"] == ' {"kept": true, "text":"雪"} '
    assert isinstance(replay[0]["created_at"], int)
    assert _rust(path, {"kind": "cursor_get", "namespace": "alpha", "consumer": "sink"})[
        "last_seq"
    ] == 9

    appended = _rust(
        path,
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
    assert appended["inserted"] is True
    assert appended["event"]["payload_json"] == '{ "raw" : [1, 2] }'
    assert _rust(path, {"kind": "latest_retained_event_seq", "namespace": "alpha"}) == 2

    # Python replay is inclusive; Rust raw replay is exclusive. Explicit mapping.
    assert list(db.iter_entity_events(namespace="alpha", from_seq=2)) == [
        (2, "edge", "e1", "UPSERT", '{ "raw" : [1, 2] }')
    ]
    assert _rust(
        path,
        {"kind": "exclusive_raw_replay", "namespace": "alpha", "after_seq": 1, "limit": 10},
    )[0]["seq"] == 2

    duplicate = _rust(
        path,
        {
            "kind": "raw_append",
            "namespace": "alpha",
            "event_id": "rust-event",
            "entity_kind": "changed",
            "entity_id": "changed",
            "op": "NOPE",
            "payload_json": "{}",
        },
    )
    assert duplicate["inserted"] is False
    assert duplicate["event"] == appended["event"]
    with pytest.raises(RustParityError) as collision:
        _rust(
            path,
            {
                "kind": "raw_append",
                "namespace": "beta",
                "event_id": "rust-event",
                "entity_kind": "node",
                "entity_id": "n2",
                "op": "UPSERT",
                "payload_json": "{}",
            },
        )
    assert collision.value.code == "KOGWISTAR_STORE_EVENT_ID_NAMESPACE_COLLISION"


def test_rust_sqlite_then_python_initializes_reads_writes_aliases_and_cursors(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    assert _rust(path, {"kind": "next_global_seq"}) == 1
    assert _rust(path, {"kind": "next_user_seq", "user_id": "bob"}) == 1
    assert _rust(path, {"kind": "set_scoped_seq", "scope_id": "scope-b", "value": 7}) is None
    assert _rust(path, {"kind": "next_scoped_seq", "scope_id": "scope-b"}) == 8
    assert _rust(path, {"kind": "alloc_event_seq", "namespace": "reserved"}) == 1
    event = _rust(
        path,
        {
            "kind": "raw_append",
            "namespace": "alpha",
            "event_id": "rust-first",
            "entity_kind": "node",
            "entity_id": "n1",
            "op": "UPSERT",
            "payload_json": '{"from":"rust"}',
        },
    )
    assert event["seq"] == 1
    assert _rust(
        path,
        {"kind": "legacy_cursor_set", "namespace": "alpha", "consumer": "sink", "last_seq": 99},
    )["last_seq"] == 99

    db = EngineSQLite(tmp_path)
    db.ensure_initialized()
    assert db.current_global_seq() == 1
    assert db.current_user_seq("bob") == 1
    assert db.current_scoped_seq("scope-b") == 8
    assert db.alloc_event_seq("reserved") == 2
    assert list(db.iter_entity_events(namespace="alpha", from_seq=1)) == [
        (1, "node", "n1", "UPSERT", '{"from":"rust"}')
    ]
    assert db.cursor_get(namespace="alpha", consumer="sink") == 99
    assert db.append_entity_event(
        namespace="alpha",
        event_id="python-second",
        entity_kind="node",
        entity_id="n2",
        op="UPSERT",
        payload_json='{"from":"python"}',
    ) == 2

    strict = _rust(
        path,
        {"kind": "strict_cursor_advance", "namespace": "alpha", "consumer": "strict", "last_seq": 2},
    )
    assert strict["last_seq"] == 2
    with pytest.raises(RustParityError) as regresses:
        _rust(
            path,
            {"kind": "strict_cursor_advance", "namespace": "alpha", "consumer": "strict", "last_seq": 1},
        )
    assert regresses.value.code == "KOGWISTAR_STORE_CURSOR_REGRESSES"


def test_python_sqlite_nested_transaction_does_not_capture_another_database(tmp_path: Path) -> None:
    """A global UoW context may only join the database that opened it.

    This is the Python/Rust handoff edge: a leaked or long-lived transaction on
    one SQLite database must not make a second EngineSQLite write invisible to
    the cached native reader for that second database.
    """
    outer = EngineSQLite(tmp_path / "outer")
    target = EngineSQLite(tmp_path / "target")
    outer.ensure_initialized()
    target.ensure_initialized()
    _rust(target.db_path, {"kind": "open_init"})

    with outer.transaction():
        assert target.append_entity_event(
            namespace="target",
            event_id="target-event",
            entity_kind="node",
            entity_id="target-node",
            op="UPSERT",
            payload_json='{"writer":"python"}',
        ) == 1
        assert list(outer.iter_entity_events(namespace="target", from_seq=1)) == []
        assert _rust(
            target.db_path,
            {"kind": "latest_retained_event_seq", "namespace": "target"},
        ) == 1

    strict = _rust(
        target.db_path,
        {
            "kind": "strict_cursor_advance",
            "namespace": "target",
            "consumer": "native-reader",
            "last_seq": 1,
        },
    )
    assert strict["last_seq"] == 1


def test_rust_sqlite_batch_is_immediate_atomic_and_durable_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    batch = {
        "kind": "batch",
        "operations": [
            {"kind": "next_global_seq"},
            {"kind": "set_user_seq", "user_id": "u", "value": 4},
            {
                "kind": "raw_append",
                "namespace": "ns",
                "event_id": "kept",
                "entity_kind": "node",
                "entity_id": "n",
                "op": "UPSERT",
                "payload_json": '{"kept":true}',
            },
            {"kind": "legacy_cursor_set", "namespace": "ns", "consumer": "sink", "last_seq": 1},
        ],
    }
    assert _rust(path, batch)[0] == 1
    assert _rust(path, {"kind": "current_global_seq"}) == 1
    assert _rust(path, {"kind": "current_user_seq", "user_id": "u"}) == 4

    aborted = dict(batch)
    aborted["abort"] = True
    aborted["operations"] = [
        {"kind": "next_global_seq"},
        {"kind": "set_user_seq", "user_id": "u", "value": 99},
        {
            "kind": "raw_append",
            "namespace": "ns",
            "event_id": "rolled-back",
            "entity_kind": "node",
            "entity_id": "rollback",
            "op": "UPSERT",
            "payload_json": "{}",
        },
        {"kind": "legacy_cursor_set", "namespace": "ns", "consumer": "sink", "last_seq": 2},
    ]
    with pytest.raises(RustParityError) as rollback:
        _rust(path, aborted)
    assert rollback.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _rust(path, {"kind": "current_global_seq"}) == 1
    assert _rust(path, {"kind": "current_user_seq", "user_id": "u"}) == 4
    assert _rust(path, {"kind": "latest_retained_event_seq", "namespace": "ns"}) == 1
    assert _rust(path, {"kind": "cursor_get", "namespace": "ns", "consumer": "sink"})["last_seq"] == 1

    with pytest.raises(RustParityError) as unsupported:
        _rust(path, {"kind": "batch", "operations": [{"kind": "current_global_seq"}]})
    assert unsupported.value.code == "KOGWISTAR_STORE_OPERATION_INVALID"

    reopened = EngineSQLite(tmp_path)
    reopened.ensure_initialized()
    assert reopened.current_global_seq() == 1
    assert reopened.current_user_seq("u") == 4
    assert list(reopened.iter_entity_events(namespace="ns", from_seq=1)) == [
        (1, "node", "n", "UPSERT", '{"kept":true}')
    ]


def test_rust_sqlite_bridge_rejects_active_python_transaction_without_waiting(
    tmp_path: Path,
) -> None:
    db = EngineSQLite(tmp_path)
    db.ensure_initialized()

    with db.transaction():
        with pytest.raises(RustParityError) as active:
            _rust(db.db_path, {"kind": "next_global_seq"})

    assert active.value.code == "KOGWISTAR_STORE_ACTIVE_PYTHON_TRANSACTION"
    assert db.current_global_seq() == 0
