from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_sqlite
from kogwistar.runtime.rust_runtime_adapter import (
    apply_claimed_worker_result,
    apply_recorded_transition,
    read_recorded_runtime_state,
)


pytestmark = [pytest.mark.ci_full, pytest.mark.runtime]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _store(path: Path, kind: str, **values: Any) -> Any:
    return store_sqlite(path=path, operation={"kind": kind, **values})


def _start(path: Path) -> None:
    apply_recorded_transition(
        path=path,
        transition={
            "contract_version": 1,
            "transition_id": "start",
            "expected_event_seq": 0,
            "kind": "start",
            "run_id": "run-1",
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
        },
    )


def _request(path: Path, *, lease_seconds: int = 30, owner: str = "worker-1") -> None:
    _store(
        path,
        "project_lane_message",
        message_id="request-1",
        namespace="runtime",
        inbox_id="python-workers",
        conversation_id="conv-1",
        recipient_id="python-worker",
        sender_id="rust-runtime",
        msg_type="workflow.worker.request.v1",
        status="pending",
        created_at=1,
        available_at=0,
        run_id="run-1",
        step_id="step-1",
        correlation_id="corr-1",
        payload_json='{"input":"work"}',
        error_json=None,
    )
    claimed = _store(
        path,
        "claim_projected_lane_messages",
        namespace="runtime",
        inbox_id="python-workers",
        claimed_by=owner,
        limit=1,
        lease_seconds=lease_seconds,
    )
    assert [row["message_id"] for row in claimed] == ["request-1"]


def _handoff(owner: str = "worker-1") -> dict[str, str]:
    return {
        "message_id": "request-1",
        "claimed_by": owner,
        "run_id": "run-1",
        "step_id": "step-1",
        "correlation_id": "corr-1",
    }


def _result() -> dict[str, Any]:
    return {
        "contract_version": 1,
        "transition_id": "worker-result-1",
        "expected_event_seq": 1,
        "kind": "recorded_step_success",
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "step-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "state_update": [["u", {"answer": "worker"}]],
        "frontier": {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
        "result": {"answer": "worker"},
    }


def _lane(path: Path) -> dict[str, Any]:
    return _store(path, "get_projected_lane_message", message_id="request-1")


def _events(path: Path) -> list[dict[str, Any]]:
    return _store(
        path, "list_server_run_events", run_id="run-1", after_seq=0, limit=20
    )


def test_claimed_worker_result_atomically_records_and_acks(tmp_path: Path) -> None:
    path = tmp_path / "handoff.sqlite"
    _start(path)
    _request(path)

    first = apply_claimed_worker_result(
        path=path, handoff=_handoff(), transition=_result()
    )
    duplicate = apply_claimed_worker_result(
        path=path, handoff=_handoff(), transition=_result()
    )

    assert first["idempotent"] is False
    assert duplicate["idempotent"] is True
    assert duplicate["event_seq"] == first["event_seq"] == 2
    assert _lane(path)["status"] == "completed"
    assert _lane(path)["claimed_by"] is None
    assert len(_events(path)) == 2
    reopened = read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )
    assert reopened is not None
    assert reopened["state"]["answer"] == "worker"
    assert reopened["last_step_seq"] == 0

    changed = copy.deepcopy(_result())
    changed["result"] = {"answer": "changed"}
    with pytest.raises(RustParityError):
        apply_claimed_worker_result(
            path=path, handoff=_handoff(), transition=changed
        )


@pytest.mark.parametrize(
    "handoff",
    [
        {**_handoff(), "claimed_by": "stale-worker"},
        {**_handoff(), "correlation_id": "wrong"},
        {**_handoff(), "run_id": "wrong"},
        {**_handoff(), "step_id": "wrong"},
    ],
)
def test_wrong_worker_or_request_identity_has_no_effect(
    tmp_path: Path, handoff: dict[str, str]
) -> None:
    path = tmp_path / f"reject-{handoff['claimed_by']}-{handoff['correlation_id']}.sqlite"
    _start(path)
    _request(path)

    with pytest.raises(RustParityError):
        apply_claimed_worker_result(path=path, handoff=handoff, transition=_result())

    assert _lane(path)["status"] == "claimed"
    assert len(_events(path)) == 1


def test_expired_worker_is_harmless_after_process_restart_reclaim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reclaim.sqlite"
    _start(path)
    _request(path, lease_seconds=-1, owner="dead-process")
    reclaimed = _store(
        path,
        "claim_projected_lane_messages",
        namespace="runtime",
        inbox_id="python-workers",
        claimed_by="restarted-process",
        limit=1,
        lease_seconds=30,
    )
    assert reclaimed[0]["claimed_by"] == "restarted-process"

    with pytest.raises(RustParityError):
        apply_claimed_worker_result(
            path=path, handoff=_handoff("dead-process"), transition=_result()
        )
    accepted = apply_claimed_worker_result(
        path=path, handoff=_handoff("restarted-process"), transition=_result()
    )
    assert accepted["event_seq"] == 2
    assert _lane(path)["status"] == "completed"


def test_fault_after_writes_rolls_back_result_checkpoint_and_ack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "abort.sqlite"
    _start(path)
    _request(path)

    with pytest.raises(RustParityError) as raised:
        apply_claimed_worker_result(
            path=path,
            handoff=_handoff(),
            transition=_result(),
            abort_after_writes=True,
        )
    assert raised.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _lane(path)["status"] == "claimed"
    assert _lane(path)["claimed_by"] == "worker-1"
    assert len(_events(path)) == 1
    reopened = read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )
    assert reopened is not None
    assert reopened["state"]["answer"] == "seed"
    assert reopened["last_step_seq"] == -1
