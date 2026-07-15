from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres, store_sqlite
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _row(message_id: str, **extra: Any) -> dict[str, Any]:
    row = {
        "kind": "project_lane_message", "message_id": message_id, "namespace": "ns",
        "inbox_id": "inbox", "conversation_id": "conversation", "recipient_id": "recipient",
        "sender_id": "sender", "msg_type": "request", "status": "pending", "created_at": 10,
        "available_at": 0, "run_id": "run", "step_id": "step", "correlation_id": "corr",
        "payload_json": '{"v":1}', "error_json": None,
    }
    row.update(extra)
    return row


def _sqlite(path: Path, operation: dict[str, Any]) -> Any:
    return store_sqlite(path=path, operation=operation)


def test_sqlite_python_rust_lane_round_trip_links_filters_and_raw_schema(tmp_path: Path) -> None:
    python = EngineSQLite(tmp_path)
    python.ensure_initialized()
    path = python.db_path
    python.project_lane_message(
        message_id="python-0", namespace="ns", inbox_id="inbox", conversation_id="python-conversation",
        recipient_id="recipient", sender_id="python", msg_type="notice", status="pending", created_at=1,
        available_at=0, run_id=None, step_id=None, correlation_id=None,
    )
    assert _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "python-0"})["sender_id"] == "python"
    _sqlite(path, _row("rust-1"))
    _sqlite(path, _row("rust-2", created_at=20, conversation_id="conversation-2"))
    # Rust -> Python readable, including Python's SQLite epoch lease ABI.
    rows = python.list_projected_lane_messages(namespace="ns")
    assert [row.message_id for row in rows] == ["python-0", "rust-1", "rust-2"]
    assert (rows[2].seq, rows[2].conversation_seq, rows[2].prev_message_id) == (3, 1, "rust-1")
    python.update_projected_lane_message_links(
        message_id="rust-2", prev_message_id="rust-1", next_message_id="tail",
        inbox_tail_message_id="rust-2", conversation_tail_message_id="rust-2",
    )
    shown = _sqlite(path, {"kind": "list_projected_lane_messages", "namespace": "ns", "newest_first": True})
    assert [row["message_id"] for row in shown] == ["rust-2", "rust-1", "python-0"]
    assert shown[0]["next_message_id"] == "tail"
    assert _sqlite(path, _row("rust-1", sender_id="changed")) is None  # duplicate no-op
    assert _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "rust-1"})["sender_id"] == "sender"
    filtered = _sqlite(path, {"kind": "list_projected_lane_messages", "namespace": "ns", "created_at_gte": 15, "available_at_lte": 0})
    assert [row["message_id"] for row in filtered] == ["rust-2"]
    with python.connect() as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(projected_lane_messages)")]
        assert columns == [
            "message_id", "namespace", "purpose", "inbox_id", "conversation_id", "recipient_id", "sender_id", "msg_type", "status", "seq", "conversation_seq", "claimed_by", "lease_until", "retry_count", "created_at", "available_at", "run_id", "step_id", "correlation_id", "payload_json", "error_json", "prev_message_id", "next_message_id", "inbox_tail_message_id", "conversation_tail_message_id",
        ]


def test_sqlite_claim_stale_owner_terminal_and_atomic_rollback(tmp_path: Path) -> None:
    path = tmp_path / "lane.db"
    _sqlite(path, _row("message"))
    claim = _sqlite(path, {"kind": "claim_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox", "claimed_by": "old", "lease_seconds": -1})
    assert claim[0]["claimed_by"] == "old"
    reclaim = _sqlite(path, {"kind": "claim_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox", "claimed_by": "new"})
    assert reclaim[0]["claimed_by"] == "new"
    for kind in ("ack_projected_lane_message", "requeue_projected_lane_message", "dead_letter_projected_lane_message"):
        _sqlite(path, {"kind": kind, "message_id": "message", "claimed_by": "old"})
    row = _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "message"})
    assert row["status"] == "claimed" and row["claimed_by"] == "new"
    _sqlite(path, {"kind": "ack_projected_lane_message", "message_id": "message", "claimed_by": "new"})
    _sqlite(path, {"kind": "requeue_projected_lane_message", "message_id": "message", "claimed_by": "new"})
    _sqlite(path, {"kind": "dead_letter_projected_lane_message", "message_id": "message", "claimed_by": "new"})
    assert _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "message"})["status"] == "completed"
    _sqlite(path, _row("dead"))
    _sqlite(path, {"kind": "dead_letter_projected_lane_message", "message_id": "dead", "claimed_by": "admin", "error_json": '{"dead":true}'})
    assert _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "dead"})["status"] == "dead-letter"
    with pytest.raises(RustParityError) as aborted:
        _sqlite(path, {"kind": "batch", "abort": True, "operations": [
            {"kind": "raw_append", "namespace": "ns", "event_id": "rolled-event", "entity_kind": "node", "entity_id": "n", "op": "UPSERT", "payload_json": "{}"},
            _row("rolled"),
            {"kind": "claim_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox", "claimed_by": "x"},
        ]})
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _sqlite(path, {"kind": "get_projected_lane_message", "message_id": "rolled"}) is None
    assert _sqlite(path, {"kind": "latest_retained_event_seq", "namespace": "ns"}) == 0
    assert _sqlite(path, {"kind": "clear_projected_lane_messages", "namespace": "ns"}) == 2
    assert _sqlite(path, {"kind": "list_projected_lane_messages", "namespace": "ns"}) == []


def test_sqlite_concurrent_project_and_claim_are_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "lane.db"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda n: _sqlite(path, _row(f"m-{n}", created_at=n)), range(8)))
    rows = _sqlite(path, {"kind": "list_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox"})
    assert sorted(row["seq"] for row in rows) == list(range(1, 9))
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda owner: _sqlite(path, {"kind": "claim_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox", "claimed_by": owner, "limit": 8}), ["a", "b"]))
    claimed = [row["message_id"] for group in claims for row in group]
    assert len(claimed) == len(set(claimed)) == 8


def test_lane_unknown_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RustParityError) as error:
        _sqlite(tmp_path / "lane.db", {**_row("bad"), "unexpected": True})
    assert error.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"


def test_postgres_lane_python_rust_and_concurrency(sa_engine, pg_dsn: str | None, pg_schema: str | None) -> None:
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    python = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    python.ensure_initialized()
    def rust(operation: dict[str, Any]) -> Any:
        return store_postgres(dsn=pg_dsn, schema=pg_schema, operation=operation)
    rust(_row("pg-first"))
    assert python.list_projected_lane_messages(namespace="ns")[0].message_id == "pg-first"
    python.project_lane_message(
        message_id="pg-python", namespace="ns", inbox_id="inbox", conversation_id="pg-python-conversation",
        recipient_id="recipient", sender_id="python", msg_type="notice", status="pending", created_at=2,
        available_at=0, run_id=None, step_id=None, correlation_id=None,
    )
    assert rust({"kind": "get_projected_lane_message", "message_id": "pg-python"})["sender_id"] == "python"
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda n: rust(_row(f"pg-{n}", created_at=n)), range(6)))
    rows = rust({"kind": "list_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox"})
    assert sorted(row["seq"] for row in rows) == list(range(1, 9))
    newest = rust({
        "kind": "list_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox",
        "newest_first": True, "limit": 2,
    })
    assert [row["message_id"] for row in newest] == [
        row.message_id for row in python.list_projected_lane_messages(
            namespace="ns", inbox_id="inbox", newest_first=True, limit=2,
        )
    ] == ["pg-first", "pg-5"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        groups = list(pool.map(lambda owner: rust({"kind": "claim_projected_lane_messages", "namespace": "ns", "inbox_id": "inbox", "claimed_by": owner, "limit": 20}), ["a", "b"]))
    ids = [row["message_id"] for group in groups for row in group]
    assert len(ids) == len(set(ids)) == 8
    assert isinstance(next(row for group in groups for row in group)["lease_until"], str)
