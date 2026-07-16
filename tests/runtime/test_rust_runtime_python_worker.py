from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from kogwistar.runtime.rust_worker import (
    AmbiguousWorkerExecution,
    RustRuntimeWorker,
    WorkerResultJournal,
)


pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def _work(*, owner: str = "worker-1") -> dict[str, Any]:
    return {
        "message_id": "lane-1",
        "claimed_by": owner,
        "run_id": "run-1",
        "step_id": "start",
        "correlation_id": "run-1",
        "expected_event_seq": 2,
        "payload": {
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "step_seq": 0,
            "token_id": "run-1",
            "parent_token_id": None,
            "state": {"seed": 1},
        },
    }


def _effect() -> dict[str, Any]:
    return {
        "state_update": [["u", {"answer": "worker"}]],
        "successors": [],
        "result": {"answer": "worker"},
    }


def test_worker_journals_callback_before_send_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    calls = {"callback": 0, "result": 0}
    accepted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/runtime/results"
        body = json.loads(request.content)
        calls["result"] += 1
        if calls["result"] == 1:
            raise httpx.ReadError("response lost", request=request)
        accepted.append(body)
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        return _effect()

    journal = tmp_path / "worker.sqlite"
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match="response lost"):
            worker.process(_work())

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-2",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as restarted:
        assert restarted.process(_work(owner="worker-2"))["event_seq"] == 3

    assert calls == {"callback": 1, "result": 2}
    assert accepted[0]["handoff"]["claimed_by"] == "worker-2"
    assert "run_id" not in accepted[0]["effect"]
    assert "frontier" not in accepted[0]["effect"]
    assert accepted[0]["effect"]["state_update"] == [
        ["u", {"answer": "worker"}]
    ]


def test_executing_journal_row_fails_closed_after_crash(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.sqlite"
    journal = WorkerResultJournal(path)
    work = _work()
    journal.begin(message_id="lane-1", work_digest="digest")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id='lane-1'"
        ).fetchone() == ("executing",)

    with pytest.raises(AmbiguousWorkerExecution, match="may already have run"):
        journal.begin(message_id="lane-1", work_digest="digest")


def test_worker_journal_releases_database_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "released.sqlite"
    journal = WorkerResultJournal(path)
    journal.begin(message_id="lane-1", work_digest="digest")
    path.unlink()
    assert not path.exists()


def test_worker_claim_and_scheduler_identity_are_validated(tmp_path: Path) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("claim"):
            return httpx.Response(200, json={"work": [_work()]}, request=request)
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "worker.sqlite",
        execute=lambda _item: _effect(),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        assert worker.poll_once(lease_seconds=45) == 1

    assert requests[0] == (
        "/internal/runtime/claim",
        {"claimed_by": "worker-1", "limit": 1, "lease_seconds": 45},
    )
    assert requests[1][0] == "/internal/runtime/results"
    assert requests[1][1]["effect"]["successors"] == []

    bad_effect = _effect()
    bad_effect["run_id"] = "forged"
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "forged.sqlite",
        execute=lambda _item: bad_effect,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match=r"may not override scheduler fields: \['run_id'\]"):
            worker.process(_work())


def test_out_of_order_result_retries_without_reexecuting_callback(tmp_path: Path) -> None:
    calls = {"callback": 0, "result": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["result"] += 1
        if calls["result"] == 1:
            return httpx.Response(
                409,
                json={"detail": "not next in canonical frontier order"},
                request=request,
            )
        return httpx.Response(200, json={"event_seq": 4}, request=request)

    def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        return _effect()

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "ordered-worker.sqlite",
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match="409 Conflict"):
            worker.process(_work())
        assert worker.process(_work())["event_seq"] == 4

    assert calls == {"callback": 1, "result": 2}
