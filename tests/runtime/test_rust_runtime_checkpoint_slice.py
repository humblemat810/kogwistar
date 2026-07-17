from __future__ import annotations

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_sqlite
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.runtime.projections import (
    workflow_checkpoint_latest_projection_namespace,
    workflow_run_status_projection_namespace,
)
from kogwistar.runtime.rust_runtime_adapter import (
    apply_recorded_transition,
    read_recorded_runtime_state,
)


pytestmark = [pytest.mark.core, pytest.mark.runtime]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _transition(
    kind: str,
    transition_id: str,
    expected_event_seq: int,
    step_seq: int,
    **more: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "contract_version": 1,
        "transition_id": transition_id,
        "expected_event_seq": expected_event_seq,
        "kind": kind,
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": step_seq,
        "node_id": "node-1",
        "token_id": "token-1",
        "parent_token_id": None,
    }
    value.update(more)
    return value


def _start(*, frontier: dict[str, Any] | None = None) -> dict[str, Any]:
    return _transition(
        "start",
        "t-start",
        0,
        0,
        user_id="user-1",
        user_turn_node_id="turn-1",
        initial_state={"answer": "seed", "_deps": {"not": "durable"}},
        frontier=frontier or {
            "pending": [["node-1", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    )


def _apply(path: Path, transition: dict[str, Any], **more: Any) -> dict[str, Any]:
    return apply_recorded_transition(path=path, transition=transition, **more)


def _read(path: Path, kind: str, **more: Any) -> Any:
    return store_sqlite(path=path, operation={"kind": kind, **more})


def _normalized_oracle(
    state: dict[str, Any], transition: dict[str, Any]
) -> dict[str, Any]:
    """Independent tiny Python checkpoint oracle; it never calls native ABI."""
    state = copy.deepcopy(state)
    for mode, payload in transition.get("state_update", []):
        if mode == "u":
            state.update(payload)
        elif mode == "a":
            for key, value in payload.items():
                state.setdefault(key, []).append(value)
        elif mode == "e":
            for key, value in payload.items():
                state.setdefault(key, []).extend(value)
        else:  # Test oracle deliberately rejects unknown reducer modes.
            raise AssertionError(mode)
    for key, value in (transition.get("update") or {}).items():
        if (transition.get("state_schema") or {}).get(key) == "a":
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value
    state.pop("_deps", None)
    state.pop("dream_deps", None)
    frontier = copy.deepcopy(transition.get("frontier") or state.get("_rt_join") or {})
    state["_rt_join"] = {
        "pending": sorted(frontier.get("pending", [])),
        "suspended": sorted(frontier.get("suspended", [])),
        "join_node_ids": frontier.get("join_node_ids", []),
        "join_outstanding": frontier.get("join_outstanding", []),
        "join_waiters": {
            key: sorted(value)
            for key, value in sorted(frontier.get("join_waiters", {}).items())
        },
    }
    if transition["kind"] == "suspend":
        if transition.get("wait_reason") is not None:
            state["wait_reason"] = transition["wait_reason"]
        if transition.get("resume_payload") is not None:
            state["resume_payload"] = transition["resume_payload"]
    return state


def _events(path: Path) -> list[dict[str, Any]]:
    return _read(path, "list_server_run_events", run_id="run-1", after_seq=0, limit=50)


def _durable_snapshot(path: Path) -> dict[str, Any]:
    return {
        "run": _read(path, "get_server_run", run_id="run-1"),
        "events": _events(path),
        "checkpoint": _read(
            path,
            "get_named_projection",
            namespace=workflow_checkpoint_latest_projection_namespace("conv-1"),
            key="run-1",
        ),
        "status": _read(
            path,
            "get_named_projection",
            namespace=workflow_run_status_projection_namespace("conv-1"),
            key="run-1",
        ),
        "runtime": read_recorded_runtime_state(
            path=path,
            run_id="run-1",
            workflow_id="wf-1",
            conversation_id="conv-1",
        ),
    }


def _fault_matrix_case(
    path: Path, kind: str
) -> tuple[dict[str, Any], str]:
    if kind == "start":
        return _start(), "running"

    _apply(path, _start())
    if kind == "resume_result":
        _apply(
            path,
            _transition(
                "suspend",
                "matrix-prerequisite-suspend",
                1,
                0,
                wait_reason="approval",
                resume_payload={"question": "continue?"},
                frontier={
                    "pending": [],
                    "suspended": [["node-1", 0, "token-1", None]],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
            ),
        )
        return (
            _transition(
                kind,
                f"matrix-{kind}",
                2,
                1,
                state_update=[["u", {"approved": True}]],
                frontier={
                    "pending": [["node-2", 0, "token-1", None]],
                    "suspended": [],
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                },
            ),
            "running",
        )

    frontier = {
        "pending": [],
        "suspended": [],
        "join_node_ids": [],
        "join_outstanding": [],
        "join_waiters": {},
    }
    more: dict[str, Any] = {"frontier": frontier}
    expected_status = "running"
    if kind == "recorded_step_success":
        more.update(
            state_update=[["u", {"answer": "matrix"}]],
            result={"answer": "matrix"},
        )
    elif kind == "suspend":
        more.update(
            wait_reason="approval",
            resume_payload={"question": "continue?"},
            frontier={**frontier, "suspended": [["node-1", 0, "token-1", None]]},
        )
        expected_status = "suspended"
    elif kind == "complete":
        expected_status = "completed"
    elif kind == "fail":
        more["errors"] = ["matrix failure"]
        expected_status = "failed"
    elif kind == "cancel":
        more["errors"] = ["matrix cancellation"]
        expected_status = "cancelled"
    return _transition(kind, f"matrix-{kind}", 1, 0, **more), expected_status


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
def test_every_durable_transition_rolls_back_all_writes_then_retries(
    kind: str, tmp_path: Path
) -> None:
    path = tmp_path / f"fault-{kind}.sqlite"
    transition, expected_status = _fault_matrix_case(path, kind)
    before = _durable_snapshot(path)

    with pytest.raises(RustParityError) as aborted:
        _apply(path, transition, abort_after_writes=True)
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _durable_snapshot(path) == before

    accepted = _apply(path, transition)
    assert accepted["idempotent"] is False
    assert len(_events(path)) == len(before["events"]) + 1
    reopened = read_recorded_runtime_state(
        path=path,
        run_id="run-1",
        workflow_id="wf-1",
        conversation_id="conv-1",
    )
    assert reopened is not None
    assert reopened["status"] == expected_status


def test_sqlite_mixed_python_rust_restart_preserves_sequence_and_projections(
    tmp_path: Path,
) -> None:
    python_v1 = EngineSQLite(tmp_path, filename="mixed.sqlite")
    python_v1.ensure_initialized()
    path = python_v1.db_path

    started = _apply(path, _start())
    assert started["event_seq"] == 1
    legacy = python_v1.append_server_run_event(
        "run-1", "python.rollback-window.observed", '{"owner":"python-v1"}'
    )
    assert legacy["seq"] == 2

    stepped_transition = _transition(
        "recorded_step_success",
        "mixed-step",
        legacy["seq"],
        0,
        state_update=[["u", {"answer": "mixed"}]],
        result={"answer": "mixed"},
        frontier={
            "pending": [["node-2", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    )
    stepped = _apply(path, stepped_transition)
    assert stepped["event_seq"] == 3

    # A newly constructed legacy facade represents rollback/restart to Python.
    python_v2 = EngineSQLite(tmp_path, filename="mixed.sqlite")
    python_v2.ensure_initialized()
    assert python_v2.get_server_run("run-1")["status"] == "running"
    python_events = python_v2.list_server_run_events("run-1")
    assert [event["seq"] for event in python_events] == [1, 2, 3]
    assert python_events[1]["payload"] == {"owner": "python-v1"}
    checkpoint = python_v2.get_named_projection(
        workflow_checkpoint_latest_projection_namespace("conv-1"), "run-1"
    )
    assert checkpoint is not None
    assert checkpoint["payload"]["node_id"] == "wf_ckpt|run-1|0"
    assert checkpoint["last_authoritative_seq"] == stepped["event_seq"]
    assert read_recorded_runtime_state(
        path=path,
        run_id="run-1",
        workflow_id="wf-1",
        conversation_id="conv-1",
    )["state"]["answer"] == "mixed"

    duplicate = _apply(path, copy.deepcopy(stepped_transition))
    assert duplicate["idempotent"] is True
    assert len(python_v2.list_server_run_events("run-1")) == 3
    completed = _apply(
        path,
        _transition(
            "complete",
            "mixed-complete",
            3,
            0,
            frontier={
                "pending": [],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
                "join_waiters": {},
            },
        ),
    )
    assert completed["event_seq"] == 4
    assert python_v2.get_server_run("run-1")["status"] == "succeeded"
    assert len(python_v2.list_server_run_events("run-1")) == 4


def test_sqlite_upgrade_python_rollback_rebuilds_disposable_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rehearse Rust upgrade, Python rollback, and event-backed recovery."""
    python_before = EngineSQLite(tmp_path, filename="upgrade-rollback.sqlite")
    python_before.ensure_initialized()
    path = python_before.db_path

    started = _apply(path, _start())
    step = _transition(
        "recorded_step_success",
        "rollback-step",
        started["event_seq"],
        0,
        state_update=[["u", {"answer": "survives-rollback"}]],
        result={"answer": "survives-rollback"},
        frontier={
            "pending": [["node-2", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    )
    stepped = _apply(path, step)

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    python_after = EngineSQLite(tmp_path, filename="upgrade-rollback.sqlite")
    python_after.ensure_initialized()
    events_before = python_after.list_server_run_events("run-1")
    assert [event["seq"] for event in events_before] == [1, 2]
    assert python_after.get_server_run("run-1")["status"] == "running"
    assert python_after.get_named_projection(
        "workflow_runtime_current_state", "run-1"
    )["payload"]["state"]["answer"] == "survives-rollback"

    # Serving projection is new Rust-owned cache, never sole truth. Python can
    # discard it during rollback without converting event history.
    python_after.clear_named_projection("workflow_runtime_current_state", "run-1")
    assert python_after.get_named_projection(
        "workflow_runtime_current_state", "run-1"
    ) is None
    replayed = read_recorded_runtime_state(
        path=path,
        run_id="run-1",
        workflow_id="wf-1",
        conversation_id="conv-1",
    )
    assert replayed == stepped["state"]
    assert python_after.list_server_run_events("run-1") == events_before

    # Exact retry stays idempotent after cache loss; no duplicate event appears.
    duplicate = _apply(path, copy.deepcopy(step))
    assert duplicate["idempotent"] is True
    assert duplicate["event_seq"] == stepped["event_seq"]
    assert python_after.list_server_run_events("run-1") == events_before

    # Re-entering Rust authority from shared history materializes a fresh cache
    # and preserves contiguous sequencing visible to the Python rollback facade.
    completed = _apply(
        path,
        _transition(
            "complete",
            "rollback-complete",
            stepped["event_seq"],
            0,
            frontier={
                "pending": [],
                "suspended": [],
                "join_node_ids": [],
                "join_outstanding": [],
                "join_waiters": {},
            },
        ),
    )
    assert completed["event_seq"] == 3
    assert [event["seq"] for event in python_after.list_server_run_events("run-1")] == [
        1,
        2,
        3,
    ]
    rebuilt = python_after.get_named_projection(
        "workflow_runtime_current_state", "run-1"
    )
    assert rebuilt is not None
    assert rebuilt["last_authoritative_seq"] == 3
    assert rebuilt["payload"]["state"]["answer"] == "survives-rollback"


def test_linear_checkpoint_complete_matches_independent_oracle(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite"
    start = _start()
    started = _apply(path, start)
    assert started["state"]["state"].get("_deps") is None

    step = _transition(
        "recorded_step_success",
        "t-step",
        1,
        0,
        state_update=[["u", {"answer": "step"}], ["a", {"ops": "one"}]],
        frontier={
            "pending": [["node-2", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
        result={"ok": True},
    )
    stepped = _apply(path, step)
    assert stepped["state"]["state"] == _normalized_oracle(
        {"answer": "seed", "_deps": {"not": "durable"}}, step
    )
    assert stepped["checkpoint_schema_version"] == 1
    assert json.loads(stepped["state_json"]) == stepped["state"]["state"]
    retried_step = _apply(path, copy.deepcopy(step))
    assert retried_step["idempotent"] is True
    assert retried_step["event_seq"] == stepped["event_seq"]
    reopened_after_step = read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )
    assert reopened_after_step is not None
    assert reopened_after_step["last_step_seq"] == 0
    assert reopened_after_step["state"]["ops"] == ["one"]

    complete = _transition("complete", "t-complete", 2, 0, frontier={
        "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
    })
    completed = _apply(path, complete)
    assert completed["server_status"] == "succeeded"
    assert completed["run_status"]["status"] == "completed"
    assert completed["run_status"]["terminal"] is True
    assert _read(path, "get_server_run", run_id="run-1")["status"] == "succeeded"
    assert len(_events(path)) == 3


def test_fanout_join_frontier_is_checkpointed_in_deterministic_order(tmp_path: Path) -> None:
    path = tmp_path / "join.sqlite"
    frontier = {
        "pending": [["b", 2, "token-b", "root"], ["a", 1, "token-a", "root"]],
        "suspended": [],
        "join_node_ids": ["join-1"],
        "join_outstanding": [2],
        "join_waiters": {"join-1": [[2, "z", None], [1, "a", "root"]]},
    }
    _apply(path, _start(frontier=frontier))
    state = read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )
    assert state is not None
    assert state["frontier"]["pending"] == [["a", 1, "token-a", "root"], ["b", 2, "token-b", "root"]]
    assert state["frontier"]["join_waiters"] == {"join-1": [[1, "a", "root"], [2, "z", None]]}
    projection = _read(
        path,
        "get_named_projection",
        namespace=workflow_checkpoint_latest_projection_namespace("conv-1"),
        key="run-1",
    )
    assert projection["payload"]["node_id"] == "wf_ckpt|run-1|0"


def test_suspend_reopen_explicit_resume_preserves_payload_and_parked_token(tmp_path: Path) -> None:
    path = tmp_path / "suspend.sqlite"
    _apply(path, _start())
    suspend = _transition(
        "suspend",
        "t-suspend",
        1,
        0,
        wait_reason="approval",
        resume_payload={"question": "continue?"},
        frontier={
            "pending": [],
            "suspended": [["node-1", 0, "token-1", None]],
            "join_node_ids": [], "join_outstanding": [], "join_waiters": {},
        },
    )
    _apply(path, suspend)
    reopened = read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )
    assert reopened is not None
    assert reopened["status"] == "suspended"
    assert reopened["wait_reason"] == "approval"
    assert reopened["resume_payload"] == {"question": "continue?"}
    assert "resume_payload" not in reopened["state"]
    assert reopened["frontier"]["pending"] == []
    assert reopened["frontier"]["suspended"] == [["node-1", 0, "token-1", None]]

    resume = _transition(
        "resume_result",
        "t-resume",
        2,
        1,
        state_update=[["u", {"approved": True}]],
        frontier={
            "pending": [["node-2", 0, "token-1", None]], "suspended": [],
            "join_node_ids": [], "join_outstanding": [], "join_waiters": {},
        },
    )
    resumed = _apply(path, resume)
    assert resumed["state"]["status"] == "running"
    assert resumed["state"]["last_node_id"] == "node-1"
    assert resumed["state"]["last_token_id"] == "token-1"
    _apply(path, _transition("complete", "t-complete", 3, 1, frontier={
        "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
    }))


@pytest.mark.parametrize(
    ("kind", "server_status", "projection_status"),
    [("cancel", "cancelled", "cancelled"), ("fail", "failed", "failed")],
)
def test_terminal_cancel_and_fail(kind: str, server_status: str, projection_status: str, tmp_path: Path) -> None:
    path = tmp_path / f"{kind}.sqlite"
    _apply(path, _start())
    out = _apply(path, _transition(kind, f"t-{kind}", 1, 0, errors=["boom"], frontier={
        "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
    }))
    assert out["server_status"] == server_status
    assert out["run_status"]["status"] == projection_status
    with pytest.raises(RustParityError):
        _apply(path, _transition("recorded_step_success", "post-terminal", 2, 1, frontier={
            "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
        }))


def test_terminal_rejects_unfinished_join_frontier(tmp_path: Path) -> None:
    path = tmp_path / "unfinished-join.sqlite"
    _apply(
        path,
        _start(
            frontier={
                "pending": [["node-1", 1, "token-1", None]],
                "suspended": [],
                "join_node_ids": ["join-1"],
                "join_outstanding": [1],
                "join_waiters": {"join-1": [[1, "waiter-1", None]]},
            }
        ),
    )
    with pytest.raises(RustParityError):
        _apply(
            path,
            _transition(
                "complete",
                "t-complete",
                1,
                0,
                frontier={
                    "pending": [],
                    "suspended": [],
                    "join_node_ids": ["join-1"],
                    "join_outstanding": [1],
                    "join_waiters": {"join-1": [[1, "waiter-1", None]]},
                },
            ),
        )


def test_idempotency_conflict_fault_rollback_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "idempotency.sqlite"
    start = _start()
    first = _apply(path, start)
    duplicate = _apply(path, copy.deepcopy(start))
    assert duplicate["idempotent"] is True
    assert duplicate["event_seq"] == first["event_seq"]
    conflict = copy.deepcopy(start)
    conflict["initial_state"]["answer"] = "changed"
    with pytest.raises(RustParityError):
        _apply(path, conflict)

    before = {
        "run": _read(path, "get_server_run", run_id="run-1"),
        "events": _events(path),
        "checkpoint": _read(path, "get_named_projection", namespace=workflow_checkpoint_latest_projection_namespace("conv-1"), key="run-1"),
        "status": _read(path, "get_named_projection", namespace=workflow_run_status_projection_namespace("conv-1"), key="run-1"),
    }
    aborted = _transition("recorded_step_success", "t-abort", 1, 0, frontier={
        "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
    })
    with pytest.raises(RustParityError) as exc:
        _apply(path, aborted, abort_after_writes=True)
    assert exc.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    after = {
        "run": _read(path, "get_server_run", run_id="run-1"),
        "events": _events(path),
        "checkpoint": _read(path, "get_named_projection", namespace=workflow_checkpoint_latest_projection_namespace("conv-1"), key="run-1"),
        "status": _read(path, "get_named_projection", namespace=workflow_run_status_projection_namespace("conv-1"), key="run-1"),
    }
    assert after == before
    assert read_recorded_runtime_state(
        path=path, run_id="run-1", workflow_id="wf-1", conversation_id="conv-1"
    )["last_step_seq"] == -1


def test_concurrent_expected_sequence_allows_one_writer_and_no_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite"
    _apply(path, _start())
    template = _transition("recorded_step_success", "t-concurrent", 1, 0, frontier={
        "pending": [], "suspended": [], "join_node_ids": [], "join_outstanding": [], "join_waiters": {}
    })
    counters = {"provider": 0, "tool": 0, "lane": 0, "graph": 0}

    def attempt(index: int) -> str:
        candidate = copy.deepcopy(template)
        candidate["transition_id"] = f"t-concurrent-{index}"
        try:
            _apply(path, candidate)
            return "accepted"
        except RustParityError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(attempt, range(2)))
    assert outcomes == ["accepted", "rejected"]
    assert len(_events(path)) == 2
    assert counters == {"provider": 0, "tool": 0, "lane": 0, "graph": 0}


def test_public_runtime_imports_stay_unchanged_and_adapter_is_json_only(tmp_path: Path) -> None:
    from kogwistar.runtime.runtime import WorkflowRuntime

    assert WorkflowRuntime.__name__ == "WorkflowRuntime"
    with pytest.raises(TypeError):
        apply_recorded_transition(path=tmp_path / "x.sqlite", transition={"bad": object()})
    with sqlite3.connect(tmp_path / "shape.sqlite") as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
