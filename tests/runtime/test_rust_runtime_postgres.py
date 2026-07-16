from __future__ import annotations

import copy
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres


pytestmark = [pytest.mark.ci_full, pytest.mark.runtime]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _pg(dsn: str, schema: str, kind: str, **values: Any) -> Any:
    return store_postgres(
        dsn=dsn,
        schema=schema,
        operation={"kind": kind, **values},
    )


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def _start(run_id: str, transition_id: str) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "user_turn_node_id": "turn-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "initial_state": {"answer": "seed"},
        "frontier": {
            "pending": [["step-1", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    }


def _result(
    run_id: str,
    transition_id: str,
    expected_event_seq: int,
    answer: str = "worker",
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": expected_event_seq,
        "kind": "recorded_step_success",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "state_update": [["u", {"answer": answer}]],
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
        "result": {"answer": answer},
    }


def _lane(run_id: str, message_id: str) -> dict[str, Any]:
    return {
        "message_id": message_id,
        "namespace": "runtime",
        "inbox_id": "python-workers",
        "conversation_id": "conv-1",
        "recipient_id": "python-worker",
        "sender_id": "rust-runtime",
        "msg_type": "workflow.worker.request.v1",
        "status": "pending",
        "created_at": 1,
        "available_at": 0,
        "run_id": run_id,
        "step_id": "step-1",
        "correlation_id": "corr-1",
        "payload_json": "{}",
        "error_json": None,
    }


def _handoff(run_id: str, message_id: str, owner: str) -> dict[str, str]:
    return {
        "message_id": message_id,
        "claimed_by": owner,
        "run_id": run_id,
        "step_id": "step-1",
        "correlation_id": "corr-1",
    }


def _prepare_claim(
    dsn: str,
    schema: str,
    run_id: str,
    message_id: str,
    owner: str,
    *,
    lease_seconds: int = 30,
) -> int:
    started = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=_start(run_id, f"start-{run_id}"),
    )
    _pg(dsn, schema, "project_lane_message", **_lane(run_id, message_id))
    claimed = _pg(
        dsn,
        schema,
        "claim_projected_lane_messages",
        namespace="runtime",
        inbox_id="python-workers",
        claimed_by=owner,
        limit=1,
        lease_seconds=lease_seconds,
    )
    assert [row["message_id"] for row in claimed] == [message_id]
    return int(started["event_seq"])


def test_postgres_recorded_worker_handoff_retry_conflict_and_reopen(
    pg_dsn: str | None,
    pg_schema: str | None,
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    _pg(dsn, schema, "ensure_schema")
    event_seq = _prepare_claim(dsn, schema, "run-1", "request-1", "worker-1")
    transition = _result("run-1", "result-1", event_seq)
    handoff = _handoff("run-1", "request-1", "worker-1")

    first = _pg(
        dsn,
        schema,
        "apply_claimed_recorded_runtime_transition",
        handoff=handoff,
        transition=transition,
    )
    duplicate = _pg(
        dsn,
        schema,
        "apply_claimed_recorded_runtime_transition",
        handoff=handoff,
        transition=transition,
    )
    assert first["idempotent"] is False
    assert duplicate["idempotent"] is True
    assert duplicate["event_seq"] == first["event_seq"]
    lane = _pg(
        dsn,
        schema,
        "get_projected_lane_message",
        message_id="request-1",
    )
    assert lane["status"] == "completed" and lane["claimed_by"] is None
    reopened = _pg(
        dsn,
        schema,
        "read_recorded_runtime_state",
        run_id="run-1",
        workflow_id="wf-1",
        conversation_id="conv-1",
    )
    assert reopened["state"]["answer"] == "worker"

    changed = copy.deepcopy(transition)
    changed["result"] = {"answer": "changed"}
    with pytest.raises(RustParityError):
        _pg(
            dsn,
            schema,
            "apply_claimed_recorded_runtime_transition",
            handoff=handoff,
            transition=changed,
        )


def test_postgres_recorded_worker_reclaim_and_fault_rollback(
    pg_dsn: str | None,
    pg_schema: str | None,
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    _pg(dsn, schema, "ensure_schema")
    event_seq = _prepare_claim(
        dsn,
        schema,
        "run-2",
        "request-2",
        "dead-worker",
        lease_seconds=-1,
    )
    reclaimed = _pg(
        dsn,
        schema,
        "claim_projected_lane_messages",
        namespace="runtime",
        inbox_id="python-workers",
        claimed_by="new-worker",
        limit=1,
        lease_seconds=30,
    )
    assert reclaimed[0]["claimed_by"] == "new-worker"
    transition = _result("run-2", "result-2", event_seq)
    with pytest.raises(RustParityError):
        _pg(
            dsn,
            schema,
            "apply_claimed_recorded_runtime_transition",
            handoff=_handoff("run-2", "request-2", "dead-worker"),
            transition=transition,
        )

    with pytest.raises(RustParityError) as aborted:
        _pg(
            dsn,
            schema,
            "apply_claimed_recorded_runtime_transition",
            handoff=_handoff("run-2", "request-2", "new-worker"),
            transition=transition,
            abort_after_writes=True,
        )
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    lane = _pg(
        dsn,
        schema,
        "get_projected_lane_message",
        message_id="request-2",
    )
    assert lane["status"] == "claimed" and lane["claimed_by"] == "new-worker"
    reopened = _pg(
        dsn,
        schema,
        "read_recorded_runtime_state",
        run_id="run-2",
        workflow_id="wf-1",
        conversation_id="conv-1",
    )
    assert reopened["state"]["answer"] == "seed"

    accepted = _pg(
        dsn,
        schema,
        "apply_claimed_recorded_runtime_transition",
        handoff=_handoff("run-2", "request-2", "new-worker"),
        transition=transition,
    )
    assert accepted["idempotent"] is False
