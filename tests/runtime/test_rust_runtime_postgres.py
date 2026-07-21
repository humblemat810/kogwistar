from __future__ import annotations

import copy
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.runtime.projections import (
    workflow_checkpoint_latest_projection_namespace,
    workflow_run_status_projection_namespace,
)


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


def _durable_snapshot(
    dsn: str, schema: str, run_id: str
) -> dict[str, Any]:
    return {
        "run": _pg(dsn, schema, "get_server_run", run_id=run_id),
        "events": _pg(
            dsn,
            schema,
            "list_server_run_events",
            run_id=run_id,
            after_seq=0,
            limit=50,
        ),
        "checkpoint": _pg(
            dsn,
            schema,
            "get_named_projection",
            namespace=workflow_checkpoint_latest_projection_namespace("conv-1"),
            key=run_id,
        ),
        "status": _pg(
            dsn,
            schema,
            "get_named_projection",
            namespace=workflow_run_status_projection_namespace("conv-1"),
            key=run_id,
        ),
        "runtime": _pg(
            dsn,
            schema,
            "read_recorded_runtime_state",
            run_id=run_id,
            workflow_id="wf-1",
            conversation_id="conv-1",
        ),
    }


def _matrix_transition(
    dsn: str, schema: str, run_id: str, kind: str
) -> tuple[dict[str, Any], str]:
    if kind == "start":
        return _start(run_id, f"matrix-{kind}"), "running"

    _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=_start(run_id, f"start-{run_id}"),
    )
    base = {
        "contract_version": 1,
        "transition_id": f"matrix-{kind}",
        "expected_event_seq": 1,
        "kind": kind,
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    }
    if kind == "recorded_step_success":
        base.update(
            state_update=[["u", {"answer": "matrix"}]],
            result={"answer": "matrix"},
        )
        return base, "running"
    if kind == "suspend":
        base.update(
            wait_reason="approval",
            resume_payload={"question": "continue?"},
            frontier={
                **base["frontier"],
                "suspended": [["step-1", 0, "token-1", None]],
            },
        )
        return base, "suspended"
    if kind == "resume_result":
        suspend = copy.deepcopy(base)
        suspend.update(
            transition_id=f"prerequisite-suspend-{run_id}",
            kind="suspend",
            wait_reason="approval",
            resume_payload={"question": "continue?"},
            frontier={
                **base["frontier"],
                "suspended": [["step-1", 0, "token-1", None]],
            },
        )
        _pg(
            dsn,
            schema,
            "apply_recorded_runtime_transition",
            transition=suspend,
        )
        base.update(
            expected_event_seq=2,
            step_seq=1,
            state_update=[["u", {"approved": True}]],
            frontier={
                **base["frontier"],
                "pending": [["step-2", 0, "token-1", None]],
            },
        )
        return base, "running"
    if kind == "complete":
        return base, "completed"
    if kind == "fail":
        base["errors"] = ["matrix failure"]
        return base, "failed"
    base["errors"] = ["matrix cancellation"]
    return base, "cancelled"


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


@pytest.mark.parametrize(
    "kind",
    [
        "start",
        "recorded_step_success",
        "suspend",
        "resume_result",
        "complete",
        "fail",
        "cancel",
    ],
)
def test_postgres_every_durable_transition_rolls_back_then_retries(
    pg_dsn: str | None,
    pg_schema: str | None,
    kind: str,
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    _pg(dsn, schema, "ensure_schema")
    run_id = f"matrix-{kind.replace('_', '-')}"
    transition, expected_status = _matrix_transition(
        dsn, schema, run_id, kind
    )
    before = _durable_snapshot(dsn, schema, run_id)

    with pytest.raises(RustParityError) as aborted:
        _pg(
            dsn,
            schema,
            "apply_recorded_runtime_transition",
            transition=transition,
            abort_after_writes=True,
        )
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _durable_snapshot(dsn, schema, run_id) == before

    accepted = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=transition,
    )
    assert accepted["idempotent"] is False
    after = _durable_snapshot(dsn, schema, run_id)
    assert len(after["events"]) == len(before["events"]) + 1
    assert after["runtime"]["status"] == expected_status


def test_postgres_mixed_python_rust_restart_preserves_sequence_and_projections(
    sa_engine: Any,
    pg_dsn: str | None,
    pg_schema: str | None,
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    if sa_engine is None:
        pytest.skip("SQLAlchemy engine unavailable")
    python_v1 = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    python_v1.ensure_initialized()
    _pg(dsn, schema, "ensure_schema")
    run_id = "mixed-rolling-run"

    started = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=_start(run_id, "mixed-start"),
    )
    legacy = python_v1.append_server_run_event(
        run_id, "python.rollback-window.observed", '{"owner":"python-v1"}'
    )
    assert legacy["seq"] == started["event_seq"] + 1

    transition = _result(run_id, "mixed-step", legacy["seq"], "mixed")
    stepped = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=transition,
    )
    assert stepped["event_seq"] == legacy["seq"] + 1

    python_v2 = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    python_v2.ensure_initialized()
    assert python_v2.get_server_run(run_id)["status"] == "running"
    python_events = python_v2.list_server_run_events(run_id)
    assert [event["seq"] for event in python_events] == [
        started["event_seq"],
        legacy["seq"],
        stepped["event_seq"],
    ]
    assert python_events[1]["payload"] == {"owner": "python-v1"}
    checkpoint = python_v2.get_named_projection(
        workflow_checkpoint_latest_projection_namespace("conv-1"), run_id
    )
    assert checkpoint is not None
    assert checkpoint["payload"]["node_id"] == f"wf_ckpt|{run_id}|0"
    assert checkpoint["last_authoritative_seq"] == stepped["event_seq"]
    assert _pg(
        dsn,
        schema,
        "read_recorded_runtime_state",
        run_id=run_id,
        workflow_id="wf-1",
        conversation_id="conv-1",
    )["state"]["answer"] == "mixed"

    duplicate = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=copy.deepcopy(transition),
    )
    assert duplicate["idempotent"] is True
    assert len(python_v2.list_server_run_events(run_id)) == 3

    complete = {
        "contract_version": 1,
        "transition_id": "mixed-complete",
        "expected_event_seq": stepped["event_seq"],
        "kind": "complete",
        "run_id": run_id,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    }
    completed = _pg(
        dsn,
        schema,
        "apply_recorded_runtime_transition",
        transition=complete,
    )
    assert completed["event_seq"] == stepped["event_seq"] + 1
    assert python_v2.get_server_run(run_id)["status"] == "succeeded"
    assert len(python_v2.list_server_run_events(run_id)) == 4


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
