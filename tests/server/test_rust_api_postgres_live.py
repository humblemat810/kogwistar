from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

from kogwistar.runtime.rust_worker import RustRuntimeWorker, RustWorkerError


pytestmark = [pytest.mark.ci]
ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "rust" / "target" / "debug" / "kogwistar-server.exe"
SERVER_SOURCES = (
    ROOT / "rust" / "crates" / "kogwistar-api" / "src" / "lib.rs",
    ROOT / "rust" / "crates" / "kogwistar-runtime" / "src" / "lib.rs",
    ROOT / "rust" / "crates" / "kogwistar-store-postgres" / "src" / "lib.rs",
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _exercise_postgres_transport(
    *, base: str, headers: dict[str, str], tmp_path: Path
) -> None:
    """Exercise durable runtime transport against an already-running server."""
    if True:  # Keep transport scenario visually grouped beneath one helper.
        submitted = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-workflow",
                "conversation_id": "pg-conversation",
                "initial_state": {"seed": 1},
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text
        run = submitted.json()

        callback_calls = 0

        def execute(work: dict[str, object]) -> dict[str, object]:
            nonlocal callback_calls
            callback_calls += 1
            assert work["run_id"] == run["run_id"]
            assert work["payload"]["runtime_started"] is True  # type: ignore[index]
            return {
                "state_update": [["u", {"postgres_worker": True}]],
                "successors": [],
                "result": {"postgres_worker": True},
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-python-worker",
            journal_path=tmp_path / "pg-worker.sqlite",
            execute=execute,
            headers=headers,
        ) as worker:
            work = worker.claim()
            assert len(work) == 1
            # A raw public submission has no Python-side frozen graph plan,
            # but the durable server must still send an explicit worker op.
            # Null would make the worker (correctly) reject an unsafe
            # node-id resolver guess.
            assert work[0]["payload"]["op"] == "noop"
            first = worker.process(work[0])
            retry = worker.process(work[0])
            assert retry["event_seq"] == first["event_seq"]
        assert callback_calls == 1

        status = httpx.get(
            f"{base}/api/workflow/runs/{run['run_id']}", headers=headers, timeout=10
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded"
        assert status.json()["result"] == {"postgres_worker": True}

        cancelled = httpx.post(
            f"{base}/api/workflow/runs/{run['run_id']}/cancel",
            headers=headers,
            timeout=10,
        )
        assert cancelled.status_code == 202, cancelled.text
        assert cancelled.json()["cancel_requested"] is True
        assert cancelled.json()["status"] == "succeeded"

        suspended_run = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-resume-workflow",
                "conversation_id": "pg-resume-conversation",
            },
            timeout=10,
        ).json()

        def suspend(_work: dict[str, object]) -> dict[str, object]:
            return {
                "status": "suspended",
                "state_update": [["u", {"before_suspend": True}]],
                "successors": [],
                "wait_reason": "approval",
                "resume_payload": {"question": "continue?"},
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-python-suspender",
            journal_path=tmp_path / "pg-suspender.sqlite",
            execute=suspend,
            headers=headers,
        ) as worker:
            suspended_work = worker.claim()
            assert len(suspended_work) == 1
            worker.process(suspended_work[0])

        contract_response = httpx.get(
            f"{base}/api/workflow/runs/{suspended_run['run_id']}/resume-contract",
            headers=headers,
            timeout=10,
        )
        assert contract_response.status_code == 200, contract_response.text
        contract = contract_response.json()
        assert contract["wait_reason"] == "approval"
        assert contract["resume_payload"] == {"question": "continue?"}
        assert contract["suspended"][0][0] == "start"
        suspended_token_id = contract["suspended"][0][2]

        resumed = httpx.post(
            f"{base}/api/workflow/runs/{suspended_run['run_id']}/resume",
            headers=headers,
            json={
                "workflow_id": "pg-resume-workflow",
                "conversation_id": "pg-resume-conversation",
                "suspended_node_id": "start",
                "suspended_token_id": suspended_token_id,
                "client_result": {
                    "status": "success",
                    "state_update": [["u", {"approved": True}]],
                    "successors": [],
                    "route_next": [],
                    "result": {"workflow_status": "succeeded"},
                },
            },
            timeout=10,
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["state"]["status"] == "running"

        def finish(work: dict[str, object]) -> dict[str, object]:
            payload = work["payload"]
            assert isinstance(payload, dict)
            assert payload["resume_effect"]["status"] == "success"
            assert payload["resume_effect"]["state_update"] == [
                ["u", {"approved": True}]
            ]
            return {
                "state_update": [["u", {"after_resume": True}]],
                "successors": [],
                "result": {"resumed": True},
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-python-resume-finisher",
            journal_path=tmp_path / "pg-resume-finisher.sqlite",
            execute=finish,
            headers=headers,
        ) as worker:
            resumed_work = worker.claim()
            assert len(resumed_work) == 1
            worker.process(resumed_work[0])

        resumed_status = httpx.get(
            f"{base}/api/workflow/runs/{suspended_run['run_id']}",
            headers=headers,
            timeout=10,
        )
        assert resumed_status.status_code == 200, resumed_status.text
        assert resumed_status.json()["status"] == "succeeded"
        assert resumed_status.json()["result"] == {"resumed": True}

        fanout_run = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-fanout-workflow",
                "conversation_id": "pg-fanout-conversation",
                "join_node_ids": ["join"],
                "start_join_mask": 1,
                "runtime_routes": [
                    {"source_node_id": "start", "target_node_id": "left", "join_mask": 1},
                    {"source_node_id": "start", "target_node_id": "right", "join_mask": 1},
                    {"source_node_id": "left", "target_node_id": "join", "join_mask": 1},
                    {"source_node_id": "right", "target_node_id": "join", "join_mask": 1},
                ],
            },
            timeout=10,
        ).json()

        def spawn_fanout(_work: dict[str, object]) -> dict[str, object]:
            return {
                "state_update": [["u", {"fanout_started": True}]],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_cost": 0.01},
                "trace_events": [{"type": "step_completed", "span_id": "fanout-start"}],
                "successors": [{"node_id": "forged", "join_mask": 0}],
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-fanout-spawner",
            journal_path=tmp_path / "pg-fanout-spawner.sqlite",
            execute=spawn_fanout,
            headers=headers,
        ) as worker:
            fanout_start = worker.claim()
            assert len(fanout_start) == 1
            worker.process(fanout_start[0])
            worker.process(fanout_start[0])

        branch_calls = {"left": 0, "right": 0}

        def finish_branch(work: dict[str, object]) -> dict[str, object]:
            node = str(work["step_id"])
            branch_calls[node] += 1
            effect: dict[str, object] = {
                "state_update": [["u", {f"fanout_{node}": True}]],
                "successors": [{"node_id": "forged", "join_mask": 0}],
            }
            return effect

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-fanout-branches",
            journal_path=tmp_path / "pg-fanout-branches.sqlite",
            execute=finish_branch,
            headers=headers,
        ) as worker:
            branches = worker.claim(limit=10)
            assert [item["step_id"] for item in branches] == ["left", "right"]
            with pytest.raises(RustWorkerError, match="409 Conflict"):
                worker.process(branches[1])
            worker.process(branches[0])
            worker.process(branches[1])
        assert branch_calls == {"left": 1, "right": 1}

        def finish_join(work: dict[str, object]) -> dict[str, object]:
            assert work["step_id"] == "join"
            return {
                "state_update": [["u", {"fanout_join": True}]],
                "successors": [],
                "result": {"fanout_done": True},
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-fanout-join",
            journal_path=tmp_path / "pg-fanout-join.sqlite",
            execute=finish_join,
            headers=headers,
        ) as worker:
            join_work = worker.claim(limit=10)
            assert [item["step_id"] for item in join_work] == ["join"]
            worker.process(join_work[0])

        fanout_status = httpx.get(
            f"{base}/api/workflow/runs/{fanout_run['run_id']}",
            headers=headers,
            timeout=10,
        )
        assert fanout_status.status_code == 200, fanout_status.text
        assert fanout_status.json()["status"] == "succeeded"
        assert fanout_status.json()["result"] == {"fanout_done": True}
        fanout_events = httpx.get(
            f"{base}/api/workflow/runs/{fanout_run['run_id']}/events",
            headers=headers,
            timeout=10,
        )
        assert fanout_events.status_code == 200, fanout_events.text
        event_payloads = [
            json.loads(line.removeprefix("data: "))
            for line in fanout_events.text.splitlines()
            if line.startswith("data: ")
        ]
        recorded = [
            event
            for event in event_payloads
            if event["event_type"] == "workflow.recorded_transition.v1"
        ]
        assert [event["reduced"]["state"]["last_node_id"] for event in recorded] == [
            "start",
            "start",
            "left",
            "right",
            "join",
            "join",
        ]
        assert [event["event_type"] for event in event_payloads].count(
            "workflow.usage.v1"
        ) == 1
        assert [event["event_type"] for event in event_payloads].count(
            "workflow.trace.v1"
        ) == 1

        for prefix in ("/api/runs", "/api/workflow/runs"):
            steps = httpx.get(
                f"{base}{prefix}/{fanout_run['run_id']}/steps",
                headers=headers,
                timeout=10,
            )
            steps.raise_for_status()
            assert [step["workflow_node_id"] for step in steps.json()["steps"]] == [
                "start",
                "left",
                "right",
                "join",
            ]
            checkpoints = httpx.get(
                f"{base}{prefix}/{fanout_run['run_id']}/checkpoints",
                headers=headers,
                timeout=10,
            )
            checkpoints.raise_for_status()
            assert checkpoints.json()["checkpoints"][-1]["state"]["fanout_join"] is True
            checkpoint = httpx.get(
                f"{base}{prefix}/{fanout_run['run_id']}/checkpoints/3",
                headers=headers,
                timeout=10,
            )
            checkpoint.raise_for_status()
            assert checkpoint.json()["state"]["fanout_join"] is True
            replay = httpx.get(
                f"{base}{prefix}/{fanout_run['run_id']}/replay",
                headers=headers,
                params={"target_step_seq": 3},
                timeout=10,
            )
            replay.raise_for_status()
            assert replay.json()["state"]["fanout_join"] is True

        for path, expected_key in (
            ("/api/workflow/resources", "migration"),
            ("/api/workflow/budget", "cost_ledger"),
            ("/api/workflow/budget/history", "events"),
            (
                "/api/workflow/lane/progress?conversation_id=pg-fanout-conversation",
                "items",
            ),
            ("/api/workflow/operator/dashboard", "resources"),
        ):
            response = httpx.get(f"{base}{path}", headers=headers, timeout=10)
            response.raise_for_status()
            assert expected_key in response.json()
        resources = httpx.get(
            f"{base}/api/workflow/resources", headers=headers, timeout=10
        ).json()
        assert resources["migration"] == {
            "implementation_mode": "rust",
            "contract_version": 1,
            "schema_version": 1,
            "parity_mismatch_count": 0,
            "queue_lag": {
                "pending_count": 0,
                "oldest_pending_age_seconds": None,
            },
            "replay_lag": {
                "events_behind": 0,
                "mode": "transactional_projection",
            },
        }
        visibility = httpx.get(
            f"{base}/api/workflow/visibility", headers=headers, timeout=10
        )
        visibility.raise_for_status()
        assert visibility.json()["current_role"] == "ro"
        assert visibility.json()["namespaces"] == {
            "storage_namespace": "workflow",
            "execution_namespace": "workflow",
        }
        assert {
            "project_view",
            "read_security_scope",
            "workflow.run.read",
        }.issubset(set(visibility.json()["current_capabilities"]))
        capabilities = httpx.get(
            f"{base}/api/workflow/capabilities", headers=headers, timeout=10
        )
        capabilities.raise_for_status()
        assert capabilities.json()["current_subject"] == "anonymous"
        assert capabilities.json()["audit_log"][-1]["action"] == "project_view"
        assert capabilities.json()["audit_log"][-1]["outcome"] == "allow"
        assert httpx.post(
            f"{base}/mcp/workflow",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "pg-mcp-replay",
                "method": "tools/call",
                "params": {
                    "name": "workflow.run_replay",
                    "arguments": {
                        "run_id": fanout_run["run_id"],
                        "target_step_seq": 3,
                    },
                },
            },
            timeout=10,
        ).json()["result"]["state"]["fanout_join"] is True
        mcp_visibility = httpx.post(
            f"{base}/mcp/workflow",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "pg-mcp-visibility",
                "method": "tools/call",
                "params": {
                    "name": "workflow.visibility_snapshot",
                    "arguments": {},
                },
            },
            timeout=10,
        )
        mcp_visibility.raise_for_status()
        assert mcp_visibility.json()["result"]["security_scope"] == "workflow"

        predicate_run = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-predicate-workflow",
                "conversation_id": "pg-predicate-conversation",
                "runtime_routes": [
                    {
                        "source_node_id": "start",
                        "target_node_id": "left",
                        "join_mask": 0,
                        "predicate": "if_true",
                    },
                    {
                        "source_node_id": "start",
                        "target_node_id": "right",
                        "join_mask": 0,
                        "predicate": "if_false",
                    },
                ],
            },
            timeout=10,
        ).json()
        predicate_claim = httpx.post(
            f"{base}/internal/runtime/claim",
            headers=headers,
            json={"claimed_by": "predicate-worker", "limit": 1, "lease_seconds": 60},
            timeout=10,
        )
        predicate_claim.raise_for_status()
        predicate_work = predicate_claim.json()["work"][0]

        def predicate_envelope(successor: dict[str, object]) -> dict[str, object]:
            return {
                "handoff": {
                    key: predicate_work[key]
                    for key in (
                        "message_id",
                        "claimed_by",
                        "run_id",
                        "step_id",
                        "correlation_id",
                    )
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": f"predicate-{predicate_run['run_id']}",
                    "successors": [successor],
                },
            }

        forged_predicate = httpx.post(
            f"{base}/internal/runtime/results",
            headers=headers,
            json=predicate_envelope({"node_id": "right", "join_mask": 1}),
            timeout=10,
        )
        assert forged_predicate.status_code == 409, forged_predicate.text
        accepted_predicate = httpx.post(
            f"{base}/internal/runtime/results",
            headers=headers,
            json=predicate_envelope({"node_id": "right", "join_mask": 0}),
            timeout=10,
        )
        accepted_predicate.raise_for_status()
        predicate_next = httpx.post(
            f"{base}/internal/runtime/claim",
            headers=headers,
            json={"claimed_by": "predicate-finisher", "limit": 1, "lease_seconds": 60},
            timeout=10,
        )
        predicate_next.raise_for_status()
        assert predicate_next.json()["work"][0]["step_id"] == "right"

        second = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-cancel-before-claim",
                "conversation_id": "pg-conversation",
            },
            timeout=10,
        ).json()
        cancel_before_claim = httpx.post(
            f"{base}/api/workflow/runs/{second['run_id']}/cancel",
            headers=headers,
            timeout=10,
        )
        cancel_before_claim.raise_for_status()
        assert cancel_before_claim.json()["status"] == "cancelled"
        assert cancel_before_claim.json()["terminal"] is True
        blocked = httpx.post(
            f"{base}/internal/runtime/claim",
            headers=headers,
            json={"claimed_by": "must-not-run", "limit": 10, "lease_seconds": 60},
            timeout=10,
        )
        assert blocked.status_code == 200, blocked.text
        assert blocked.json()["work"] == []

        claimed_then_cancelled = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-cancel-after-claim",
                "conversation_id": "pg-conversation",
            },
            timeout=10,
        ).json()
        claimed = httpx.post(
            f"{base}/internal/runtime/claim",
            headers=headers,
            json={"claimed_by": "cancelled-worker", "limit": 1, "lease_seconds": 60},
            timeout=10,
        )
        claimed.raise_for_status()
        claimed_work = claimed.json()["work"][0]
        cancelled = httpx.post(
            f"{base}/api/workflow/runs/{claimed_then_cancelled['run_id']}/cancel",
            headers=headers,
            timeout=10,
        )
        cancelled.raise_for_status()
        assert cancelled.json()["status"] == "cancelled"
        stale_result = httpx.post(
            f"{base}/internal/runtime/results",
            headers=headers,
            json={
                "handoff": {
                    key: claimed_work[key]
                    for key in (
                        "message_id",
                        "claimed_by",
                        "run_id",
                        "step_id",
                        "correlation_id",
                    )
                },
                "effect": {
                    "contract_version": 1,
                    "effect_id": f"stale-{claimed_then_cancelled['run_id']}",
                    "state_update": [["u", {"must_not_apply": True}]],
                    "successors": [],
                    "result": {"must_not_apply": True},
                },
            },
            timeout=10,
        )
        assert stale_result.status_code == 409, stale_result.text


def test_rust_postgres_server_submit_claim_read_cancel_true_socket(
    pg_dsn: str | None,
    tmp_path: Path,
) -> None:
    if not pg_dsn:
        pytest.skip("PostgreSQL fixture unavailable")
    if not SERVER.exists():
        pytest.skip("build kogwistar-server before live transport test")
    if any(source.stat().st_mtime > SERVER.stat().st_mtime for source in SERVER_SOURCES):
        pytest.skip("rebuild current kogwistar-server before live transport test")
    port = _port()
    schema = f"rust_api_{uuid.uuid4().hex}"
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "pg",
            "KOGWISTAR_PG_DSN": pg_dsn,
            "KOGWISTAR_PG_SCHEMA_BASE": schema,
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
        }
    )
    process = subprocess.Popen(
        [str(SERVER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(f"Rust server exited\nstdout={stdout}\nstderr={stderr}")
            try:
                if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise AssertionError("Rust PostgreSQL server did not become ready")
        _exercise_postgres_transport(
            base=base,
            headers={"x-kogwistar-roles": "reader"},
            tmp_path=tmp_path,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
