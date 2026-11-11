from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from kogwistar.engine_core.embedding_profile import EmbeddingProfile
from kogwistar.engine_core.edge_endpoint_rows import edge_endpoint_rows
from kogwistar.runtime.rust_worker import RustRuntimeWorker, RustWorkerError

pytestmark = [pytest.mark.ci]
ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "rust" / "target" / "debug" / "kogwistar-server.exe"
RUST_ROOT = ROOT / "rust"
SERVER_SOURCES = tuple(
    sorted(
        path
        for path in RUST_ROOT.rglob("*")
        if path.is_file()
        and "target" not in path.parts
        and "tests" not in path.parts
        and (path.suffix == ".rs" or path.name in {"Cargo.toml", "Cargo.lock"})
    )
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drop_postgres_schema(sa_engine, schema: str) -> None:
    bootstrap_engine = sa_engine.execution_options(kogwistar_schema_bootstrap=True)
    with bootstrap_engine.begin() as connection:
        connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _hs256_token(secret: str, claims: dict[str, object]) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f'{encode({"alg": "HS256", "typ": "JWT"})}.{encode(claims)}'
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


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
            assert retry["idempotent"] is True
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
        assert cancelled.json()["cancel_requested"] is False
        assert cancelled.json()["terminal"] is True
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

        worker_errors: list[Exception] = []

        def finish_resumed_run() -> None:
            try:
                with RustRuntimeWorker(
                    base_url=base,
                    worker_id="pg-python-resume-finisher",
                    journal_path=tmp_path / "pg-resume-finisher.sqlite",
                    execute=finish,
                    headers=headers,
                ) as worker:
                    deadline = time.monotonic() + 10
                    resumed_work: list[dict[str, object]] = []
                    while not resumed_work and time.monotonic() < deadline:
                        resumed_work = worker.claim()
                        if not resumed_work:
                            time.sleep(0.05)
                    assert len(resumed_work) == 1
                    worker.process(resumed_work[0])
            except Exception as error:
                worker_errors.append(error)

        finisher = threading.Thread(target=finish_resumed_run, daemon=True)
        finisher.start()
        resumed = httpx.post(
            f"{base}/api/workflow/runs/{suspended_run['run_id']}/resume?wait_for_completion=true",
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
        finisher.join(timeout=10)
        assert not finisher.is_alive(), "PostgreSQL resume worker did not stop"
        assert not worker_errors, worker_errors
        assert resumed.status_code == 200, resumed.text
        resume_result_contract = json.loads(
            (ROOT / "contracts/golden/workflow-run-resume-result-v1.json").read_text(
                encoding="utf-8"
            )
        )
        resumed_result = resumed.json()
        assert set(resumed_result) == set(resume_result_contract["required_fields"])
        assert resumed_result["workflow_status"] == "succeeded"
        assert resumed_result["final_state"]["after_resume"] is True

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
        fanout_start_traces = [
            event
            for event in event_payloads
            if event["event_type"] == "workflow.trace.v1"
            and event["trace"].get("span_id") == "fanout-start"
        ]
        assert len(fanout_start_traces) == 1

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
            assert response.status_code == 200, response.text
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
        assert visibility.json()["current_role"] == "rw"
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
        replay_response = httpx.post(
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
        )
        assert replay_response.status_code == 200, replay_response.text
        assert replay_response.json()["result"]["state"]["fanout_join"] is True
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


def _exercise_postgres_synchronous_resume(
    *, base: str, headers: dict[str, str], tmp_path: Path
) -> None:
    contract = json.loads(
        (ROOT / "contracts/golden/workflow-run-resume-result-v1.json").read_text(
            encoding="utf-8"
        )
    )
    submitted = httpx.post(
        f"{base}/api/workflow/runs",
        headers=headers,
        json={
            "workflow_id": "pg-sync-resume-workflow",
            "conversation_id": "pg-sync-resume-conversation",
        },
        timeout=10,
    )
    assert submitted.status_code == 202, submitted.text
    run_id = submitted.json()["run_id"]

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
        worker_id="pg-sync-resume-suspender",
        journal_path=tmp_path / "pg-sync-resume-suspender.sqlite",
        execute=suspend,
        headers=headers,
    ) as worker:
        work = worker.claim()
        assert len(work) == 1
        worker.process(work[0])

    resume_contract = httpx.get(
        f"{base}/api/workflow/runs/{run_id}/resume-contract",
        headers=headers,
        timeout=10,
    )
    assert resume_contract.status_code == 200, resume_contract.text
    token_id = resume_contract.json()["suspended"][0][2]
    resume_body = {
        "workflow_id": "pg-sync-resume-workflow",
        "conversation_id": "pg-sync-resume-conversation",
        "suspended_node_id": "start",
        "suspended_token_id": token_id,
        "client_result": {
            "status": "success",
            "state_update": [["u", {"approved": True}]],
            "successors": [],
            "route_next": [],
            "result": {"workflow_status": "succeeded"},
        },
    }
    denied = httpx.post(
        f"{base}/api/workflow/runs/{run_id}/resume?wait_for_completion=true",
        headers={"x-kogwistar-roles": "reader"},
        json=resume_body,
        timeout=10,
    )
    assert denied.status_code == 403, denied.text

    def finish(work: dict[str, object]) -> dict[str, object]:
        payload = work["payload"]
        assert isinstance(payload, dict)
        assert payload["resume_effect"]["status"] == "success"
        return {
            "state_update": [["u", {"after_resume": True}]],
            "successors": [],
            "result": {"resumed": True},
        }

    worker_errors: list[Exception] = []

    def run_finisher() -> None:
        try:
            with RustRuntimeWorker(
                base_url=base,
                worker_id="pg-sync-resume-finisher",
                journal_path=tmp_path / "pg-sync-resume-finisher.sqlite",
                execute=finish,
                headers=headers,
            ) as worker:
                deadline = time.monotonic() + 10
                work: list[dict[str, object]] = []
                while not work and time.monotonic() < deadline:
                    work = worker.claim()
                    if not work:
                        time.sleep(0.05)
                assert len(work) == 1
                worker.process(work[0])
        except Exception as error:
            worker_errors.append(error)

    finisher = threading.Thread(target=run_finisher, daemon=True)
    finisher.start()
    resumed = httpx.post(
        f"{base}/api/workflow/runs/{run_id}/resume",
        headers=headers,
        json=resume_body,
        timeout=15,
    )
    finisher.join(timeout=10)
    assert not finisher.is_alive(), "PostgreSQL resume worker did not stop"
    assert not worker_errors, worker_errors
    assert resumed.status_code == 200, resumed.text
    result = resumed.json()
    assert set(result) == set(contract["required_fields"])
    assert result["workflow_status"] == "succeeded"
    assert result["final_state"]["after_resume"] is True
    assert result["budget"] == {}


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
            headers={"x-kogwistar-roles": "reader,rw"},
            tmp_path=tmp_path,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_rust_postgres_worker_retry_returns_original_transition_sequence(
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
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "pg",
            "KOGWISTAR_PG_DSN": pg_dsn,
            "KOGWISTAR_PG_SCHEMA_BASE": f"rust_retry_{uuid.uuid4().hex}",
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
    headers = {"x-kogwistar-roles": "reader,rw"}
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

        submitted = httpx.post(
            f"{base}/api/workflow/runs",
            headers=headers,
            json={
                "workflow_id": "pg-worker-retry-workflow",
                "conversation_id": "pg-worker-retry-conversation",
                "initial_state": {"seed": 1},
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text
        run_id = submitted.json()["run_id"]
        callback_calls = 0

        def execute(_work: dict[str, object]) -> dict[str, object]:
            nonlocal callback_calls
            callback_calls += 1
            return {
                "state_update": [["u", {"worker_done": True}]],
                "successors": [],
                "result": {"worker_done": True},
            }

        with RustRuntimeWorker(
            base_url=base,
            worker_id="pg-retry-worker",
            journal_path=tmp_path / "pg-worker-retry.sqlite",
            execute=execute,
            headers=headers,
        ) as worker:
            work = worker.claim()
            assert len(work) == 1
            first = worker.process(work[0])
            retry = worker.process(work[0])
        assert callback_calls == 1
        assert retry["event_seq"] == first["event_seq"]
        assert retry["idempotent"] is True
        status = httpx.get(f"{base}/api/workflow/runs/{run_id}", headers=headers, timeout=10)
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded"
        assert status.json()["result"] == {"worker_done": True}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_rust_postgres_synchronous_resume_matches_shared_contract(
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
    schema = f"rust_resume_{uuid.uuid4().hex}"
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
        _exercise_postgres_synchronous_resume(
            base=base,
            headers={"x-kogwistar-roles": "reader,rw"},
            tmp_path=tmp_path,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_rust_postgres_graph_read_tools_match_python_over_shared_projection(
    request: pytest.FixtureRequest,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not pg_dsn or not pg_schema or sa_engine is None:
        pytest.skip("PostgreSQL fixture unavailable")
    if not SERVER.exists():
        pytest.skip("build kogwistar-server before live graph parity test")
    if any(source.stat().st_mtime > SERVER.stat().st_mtime for source in SERVER_SOURCES):
        pytest.skip("rebuild current kogwistar-server before live graph parity test")

    for name in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
    ):
        monkeypatch.setenv(name, "python")

    from kogwistar.conversation.models import ConversationNode
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import (
        Edge,
        Node,
        PureChromaEdge,
        PureChromaNode,
    )
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
    from kogwistar.graph_query import GraphQuery
    from kogwistar.server.bootstrap import load_server_storage_settings
    from kogwistar.server.auth_middleware import claims_ctx
    from tests._helpers.graph_builders import (
        build_entity_node,
        build_relationship_edge,
        mk_document_grounding,
    )

    namespace = "alpha"
    graph_type = "knowledge"
    python_storage = load_server_storage_settings(
        {
            "GKE_BACKEND": "pg",
            "GKE_PG_URL": pg_dsn,
            "GKE_PG_SCHEMA": pg_schema,
        }
    )
    knowledge_schema = python_storage.schema_for(graph_type)
    request.addfinalizer(lambda: _drop_postgres_schema(sa_engine, knowledge_schema))
    wisdom_schema = python_storage.schema_for("wisdom")
    request.addfinalizer(lambda: _drop_postgres_schema(sa_engine, wisdom_schema))
    conversation_schema = python_storage.schema_for("conversation")
    request.addfinalizer(
        lambda: _drop_postgres_schema(sa_engine, conversation_schema)
    )
    conversation_backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, schema=conversation_schema
    )
    conversation_backend.ensure_schema()
    doc_id = "D-neighbor-parity"
    id_prefix = uuid.uuid4().hex
    source = build_entity_node(
        node_id=f"{id_prefix}-source",
        doc_id=doc_id,
        label="Semantic source",
        summary="Semantic source node",
        metadata={"visibility": "public"},
    )
    visible_target = build_entity_node(
        node_id=f"{id_prefix}-visible-target",
        doc_id=doc_id,
        label="Semantic parity query match",
        summary="Deterministic semantic parity query result",
        metadata={"visibility": "public"},
    )
    private_target = build_entity_node(
        node_id=f"{id_prefix}-private-target",
        doc_id=doc_id,
        label="Private semantic target",
        summary="Semantic confidential target",
        metadata={"visibility": "private", "owner_agent_id": "owner-a"},
    )
    other_doc_target = build_entity_node(
        node_id=f"{id_prefix}-other-doc-target",
        doc_id="D-neighbor-parity-other",
        label="Unrelated filler",
        summary="Unrelated material",
        metadata={"visibility": "public"},
    )
    restricted_evidence_target = build_entity_node(
        node_id=f"{id_prefix}-restricted-evidence-target",
        doc_id=doc_id,
        label="Restricted source payload",
        summary="Semantic parity evidence with unrelated source context",
        metadata={"visibility": "public"},
        mentions=[mk_document_grounding(doc_id)],
    )

    def fixture_vector(value: str) -> list[float]:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            searchable = " ".join(
                str(payload.get(field) or "") for field in ("label", "summary")
            )
        else:
            searchable = value
        normalized = searchable.casefold()
        vector = [
            float("semantic" in normalized),
            float("parity" in normalized),
            float("unrelated" in normalized),
        ]
        norm = sum(component * component for component in vector) ** 0.5
        return [component / norm for component in vector] if norm else [0.0, 0.0, 1.0]

    def fixture_embedding(values: list[str]) -> list[list[float]]:
        return [fixture_vector(value) for value in values]
    visible_edge = build_relationship_edge(
        edge_id=f"{id_prefix}-visible-edge",
        src=source.id,
        tgt=visible_target.id,
        doc_id=doc_id,
        metadata={"visibility": "public"},
    )
    private_endpoint_edge = build_relationship_edge(
        edge_id=f"{id_prefix}-private-endpoint-edge",
        src=source.id,
        tgt=private_target.id,
        doc_id=doc_id,
        metadata={"visibility": "public"},
    )
    other_doc_edge = build_relationship_edge(
        edge_id=f"{id_prefix}-other-doc-edge",
        src=source.id,
        tgt=other_doc_target.id,
        doc_id="D-neighbor-parity-other",
        metadata={"visibility": "public"},
    )

    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, schema=knowledge_schema
    )
    backend.ensure_schema()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "shared_postgres_graph"),
        backend=backend,
        namespace=namespace,
        kg_graph_type=graph_type,
        embedding_function=fixture_embedding,
        embedding_profile=EmbeddingProfile(
            provider="ollama",
            model="fixture-embedding-v1",
            dimension=3,
            similarity_metric="cosine",
            embedding_kind="single_vector",
        ),
        acl_enabled=True,
    )
    owner_token = claims_ctx.set(
        {
            "agent_id": "owner-a",
            "sub": "owner-a",
            "role": "rw",
            "ns": graph_type,
            "storage_ns": namespace,
        }
    )
    try:
        for node in (
            source,
            visible_target,
            private_target,
            other_doc_target,
            restricted_evidence_target,
        ):
            engine.write.add_node(node)
        for edge in (visible_edge, private_endpoint_edge, other_doc_edge):
            engine.write.add_edge(edge)
        _grounding_ids, span_ids = engine.acl.usage_ids_for_item(
            restricted_evidence_target
        )
        assert len(span_ids) == 1
        engine.acl.record_acl(
            grain="span",
            truth_graph=graph_type,
            entity_id=restricted_evidence_target.id,
            target_item_id=span_ids[0],
            version=2,
            mode="private",
            owner_id="owner-a",
        )
    finally:
        claims_ctx.reset(owner_token)

    wisdom_backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, schema=wisdom_schema
    )
    wisdom_backend.ensure_schema()
    wisdom_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "python_persisted_wisdom"),
        backend=wisdom_backend,
        namespace="wisdom",
        kg_graph_type="wisdom",
        embedding_function=fixture_embedding,
        embedding_profile=EmbeddingProfile(
            provider="ollama",
            model="fixture-embedding-v1",
            dimension=3,
            similarity_metric="cosine",
            embedding_kind="single_vector",
        ),
        acl_enabled=True,
    )
    wisdom_doc_id = f"python-wisdom-{id_prefix}"
    wisdom_seed = build_entity_node(
        node_id=f"{id_prefix}-python-wisdom-seed",
        doc_id=wisdom_doc_id,
        label="Python persisted semantic parity lesson",
        summary="A useful Wisdom record persisted by the Python graph engine.",
        metadata={"visibility": "public"},
    )
    wisdom_related = build_entity_node(
        node_id=f"{id_prefix}-python-wisdom-related",
        doc_id=wisdom_doc_id,
        label="Related Python Wisdom concept",
        summary="This concept is connected to the persisted lesson.",
        metadata={"visibility": "public"},
    )
    wisdom_edge = build_relationship_edge(
        edge_id=f"{id_prefix}-python-wisdom-edge",
        src=wisdom_seed.id,
        tgt=wisdom_related.id,
        doc_id=wisdom_doc_id,
        metadata={"visibility": "public"},
    )
    pure_seed = PureChromaNode(
        id=f"{id_prefix}-python-pure-wisdom-seed",
        label="Semantic parity pure Wisdom seed",
        type="entity",
        summary="A public pure graph model persisted by Python.",
        metadata={"visibility": "public"},
    )
    pure_related = PureChromaNode(
        id=f"{id_prefix}-python-pure-wisdom-related",
        label="Semantic parity pure Wisdom related node",
        type="entity",
        summary="A second public pure graph model persisted by Python.",
        metadata={"visibility": "public"},
    )
    pure_edge = PureChromaEdge(
        id=f"{id_prefix}-python-pure-wisdom-edge",
        label="Pure Wisdom relation",
        type="relationship",
        summary="Connects pure Wisdom graph entities.",
        relation="related_to",
        source_ids=[pure_seed.id],
        target_ids=[pure_related.id],
        source_edge_ids=[],
        target_edge_ids=[],
        metadata={"visibility": "public"},
    )
    wisdom_writer = {
        "agent_id": "wisdom-owner",
        "sub": "wisdom-owner",
        "role": "rw",
        "ns": "wisdom",
        "storage_ns": "wisdom",
    }
    wisdom_token = claims_ctx.set(wisdom_writer)
    try:
        wisdom_engine.write.add_node(wisdom_seed)
        wisdom_engine.write.add_node(wisdom_related)
        wisdom_engine.write.add_edge(wisdom_edge)
        wisdom_engine.write.add_pure_node(pure_seed)
        wisdom_engine.write.add_pure_node(pure_related)
        wisdom_engine.write.add_pure_edge(pure_edge)
    finally:
        claims_ctx.reset(wisdom_token)

    reader_claims: dict[str, object] = {
        "sub": "reader-b",
        "role": "ro",
        "roles": ["reader"],
        # Auth namespace, storage namespace, and graph role are independent:
        # this tool is authorized in docs while reading the knowledge graph.
        "ns": ["docs"],
        "storage_ns": namespace,
        "capabilities": ["read_graph"],
        "exp": 4_102_444_800,
    }
    owner_claims: dict[str, object] = {
        "sub": "owner-a",
        "role": "rw",
        "roles": ["reader", "rw"],
        "ns": ["docs"],
        "storage_ns": namespace,
        "capabilities": ["read_graph", "write_graph"],
        "exp": 4_102_444_800,
    }
    reader_token = claims_ctx.set(reader_claims)
    try:
        python_result = GraphQuery(engine).neighbors(source.id, doc_id=doc_id)
    finally:
        claims_ctx.reset(reader_token)

    port = _port()
    secret = uuid.uuid4().hex

    class FixedEmbeddingHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/api/embed":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            inputs = request.get("input")
            if not isinstance(inputs, list):
                self.send_error(400)
                return
            body = json.dumps(
                {"embeddings": [fixture_vector(value) for value in inputs]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    embedding_server = ThreadingHTTPServer(("127.0.0.1", 0), FixedEmbeddingHandler)
    embedding_thread = threading.Thread(
        target=embedding_server.serve_forever, daemon=True
    )
    embedding_thread.start()
    embedding_endpoint = f"http://127.0.0.1:{embedding_server.server_port}"
    auth_db_path = tmp_path / "owner-auth.sqlite"
    with sqlite3.connect(auth_db_path) as auth_db:
        auth_db.execute(
            """CREATE TABLE users (
                user_id VARCHAR(64) NOT NULL PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                display_name VARCHAR(255),
                is_active BOOLEAN NOT NULL,
                global_role VARCHAR(32),
                global_ns VARCHAR(255),
                created_at DATETIME NOT NULL,
                last_login_at DATETIME
            )"""
        )
        auth_db.execute(
            """INSERT INTO users (
                user_id, email, display_name, is_active, global_role, global_ns, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                "conversation-owner",
                "conversation-owner@example.test",
                "Conversation Test Owner",
                1,
                "rw",
                "conversation",
            ),
        )
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "AUTH_MODE": "jwt",
            "JWT_SECRET": secret,
            "JWT_ALG": "HS256",
            "AUTH_DB_URL": f"sqlite:///{auth_db_path.as_posix()}",
            "KOGWISTAR_BACKEND": "pg",
            "KOGWISTAR_PG_DSN": pg_dsn,
            "KOGWISTAR_PG_SCHEMA_BASE": pg_schema,
            "KOGWISTAR_EMBEDDING_PROVIDER": "ollama",
            "KOGWISTAR_EMBEDDING_MODEL": "fixture-embedding-v1",
            "KOGWISTAR_EMBEDDING_DIMENSION": "3",
            "KOGWISTAR_EMBEDDING_ENDPOINT": embedding_endpoint,
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
            "KOGWISTAR_RUNTIME_NATIVE_INLINE": "true",
            "RUST_LOG": "warn",
        }
    )
    server_log_path = tmp_path / "rust-postgres-graph-read.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(SERVER)],
        cwd=ROOT,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"Rust server exited\nstdout={stdout}\nstderr={stderr}"
                )
            try:
                if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise AssertionError("Rust PostgreSQL server did not become ready")

        def rust_mcp_tool(
            tool_name: str,
            arguments: dict[str, object],
            claims: dict[str, object],
            endpoint: str = "/mcp",
        ) -> dict[str, object]:
            response = httpx.post(
                f"{base}{endpoint}",
                headers={
                    "Authorization": f"Bearer {_hs256_token(secret, claims)}",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": f"graph-read-parity-{tool_name}",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                },
                timeout=10,
            )
            assert response.status_code == 200, response.text
            if "text/event-stream" in response.headers.get("content-type", ""):
                data_lines = [
                    line.removeprefix("data: ")
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert len(data_lines) == 1, response.text
                payload = json.loads(data_lines[0])
            else:
                payload = response.json()
            result = payload.get("result")
            if isinstance(result, dict):
                structured = result.get("structuredContent")
                if isinstance(structured, dict):
                    result = structured
            assert isinstance(result, dict), response.text
            return result

        def rust_syscall(
            op: str, args: dict[str, object], claims: dict[str, object]
        ) -> dict[str, object]:
            response = httpx.post(
                f"{base}/api/syscall/v1/{op}",
                headers={
                    "Authorization": f"Bearer {_hs256_token(secret, claims)}"
                },
                json={"version": "v1", "op": op, "args": args},
                timeout=10,
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert isinstance(result, dict), response.text
            return result

        def rust_mcp_neighbors(claims: dict[str, object]) -> dict[str, object]:
            return rust_mcp_tool(
                "kg_neighbors", {"rid": source.id, "doc_id": doc_id}, claims
            )

        def semantic_ids(
            result: dict[str, object],
        ) -> tuple[list[str], list[dict[str, set[str]]]]:
            seeds = result["seeds"]
            layers = result["layers"]
            assert isinstance(seeds, list)
            assert isinstance(layers, list)
            seed_ids = [json.loads(item)["id"] for item in seeds]
            layer_ids = []
            for layer in layers:
                assert isinstance(layer, dict)
                layer_ids.append(
                    {
                        "nodes": {json.loads(item)["id"] for item in layer["nodes"]},
                        "edges": {json.loads(item)["id"] for item in layer["edges"]},
                    }
                )
            return seed_ids, layer_ids

        wisdom_reader_claims: dict[str, object] = {
            "sub": "wisdom-reader",
            "role": "ro",
            "roles": ["reader"],
            "ns": ["wisdom"],
            "storage_ns": "wisdom",
            "capabilities": ["read_graph"],
            "exp": 4_102_444_800,
        }
        rust_pure_seed_id = f"{id_prefix}-rust-pure-wisdom-seed"
        rust_pure_related_id = f"{id_prefix}-rust-pure-wisdom-related"
        rust_private_node_id = f"{id_prefix}-rust-private-wisdom-node"
        rust_shared_node_id = f"{id_prefix}-rust-shared-wisdom-node"
        rust_pure_edge_id = f"{id_prefix}-rust-pure-wisdom-edge"
        rust_wisdom_write = rust_mcp_tool(
            "wisdom.kg_upsert_graph",
            {
                "inp": {
                    "nodes": [
                        {
                            "id": rust_pure_seed_id,
                            "label": "Rust persisted pure Wisdom seed",
                            "type": "entity",
                            "summary": "Public pure node written by the Rust Wisdom API.",
                            "metadata": {"visibility": "public"},
                        },
                        {
                            "id": rust_pure_related_id,
                            "label": "Rust persisted pure Wisdom related node",
                            "type": "entity",
                            "summary": "Public pure node connected by a Rust-written edge.",
                            "metadata": {"visibility": "public"},
                        },
                        {
                            "id": rust_private_node_id,
                            "label": "Rust persisted private Wisdom node",
                            "type": "entity",
                            "summary": "Private node written by the Rust Wisdom API.",
                            "metadata": {"visibility": "private"},
                        },
                        {
                            "id": rust_shared_node_id,
                            "label": "Rust persisted shared Wisdom node",
                            "type": "entity",
                            "summary": "Shared node written by the Rust Wisdom API.",
                            "metadata": {
                                "visibility": "shared",
                                "shared_with_agents": ["wisdom-reader"],
                            },
                        },
                    ],
                    "edges": [
                        {
                            "id": rust_pure_edge_id,
                            "label": "Rust pure Wisdom relation",
                            "type": "relationship",
                            "summary": "Connects the two Rust-written pure graph nodes.",
                            "relation": "related_to",
                            "source_ids": [rust_pure_seed_id],
                            "target_ids": [rust_pure_related_id],
                            "source_edge_ids": [],
                            "target_edge_ids": [],
                            "metadata": {"visibility": "public"},
                        }
                    ],
                }
            },
            {
                "sub": "rust-wisdom-owner",
                "user_id": "rust-wisdom-owner",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": ["wisdom"],
                "storage_ns": "wisdom",
                "capabilities": ["write_graph"],
                "exp": 4_102_444_800,
            },
        )
        assert rust_wisdom_write.get("node_ids") == [
            rust_pure_seed_id,
            rust_pure_related_id,
            rust_private_node_id,
            rust_shared_node_id,
        ]
        assert rust_wisdom_write.get("edge_ids") == [rust_pure_edge_id]

        rust_pure_read_token = claims_ctx.set(wisdom_reader_claims)
        try:
            rust_written_node = wisdom_engine.read.get_nodes(ids=[rust_pure_seed_id])
            rust_written_edge = wisdom_engine.read.get_edges(ids=[rust_pure_edge_id])
            hidden_rust_node = wisdom_engine.read.get_nodes(ids=[rust_private_node_id])
            shared_rust_node = wisdom_engine.read.get_nodes(ids=[rust_shared_node_id])
        finally:
            claims_ctx.reset(rust_pure_read_token)
        assert len(rust_written_node) == 1
        assert len(rust_written_edge) == 1
        assert isinstance(rust_written_node[0], PureChromaNode)
        assert isinstance(rust_written_edge[0], PureChromaEdge)
        assert rust_written_edge[0].source_ids == [rust_pure_seed_id]
        assert rust_written_edge[0].target_ids == [rust_pure_related_id]
        expected_endpoint_rows = edge_endpoint_rows(rust_written_edge[0])
        endpoint_projection = wisdom_backend.edge_endpoints_get(
            where={"edge_id": rust_pure_edge_id}, include=["metadatas"]
        )
        expected_endpoint_keys = set().union(
            *(row.keys() for row in expected_endpoint_rows)
        )
        actual_endpoint_rows = [
            {
                key: metadata[key]
                for key in expected_endpoint_keys
                if key in metadata
            }
            for metadata in endpoint_projection["metadatas"]
        ]
        assert sorted(actual_endpoint_rows, key=lambda row: row["id"]) == sorted(
            expected_endpoint_rows, key=lambda row: row["id"]
        )
        assert hidden_rust_node == [], "Python public reader must honor Rust ACL graph"
        assert len(shared_rust_node) == 1
        assert isinstance(shared_rust_node[0], PureChromaNode)
        intruder_token = claims_ctx.set({**wisdom_reader_claims, "sub": "unrelated-reader"})
        try:
            hidden_shared_node = wisdom_engine.read.get_nodes(ids=[rust_shared_node_id])
        finally:
            claims_ctx.reset(intruder_token)
        assert hidden_shared_node == [], "shared ACL must not leak to unrelated readers"
        rust_owner_token = claims_ctx.set(
            {
                **wisdom_reader_claims,
                "sub": "rust-wisdom-owner",
                "agent_id": "rust-wisdom-owner",
            }
        )
        try:
            owner_rust_node = wisdom_engine.read.get_nodes(ids=[rust_private_node_id])
        finally:
            claims_ctx.reset(rust_owner_token)
        assert len(owner_rust_node) == 1
        assert isinstance(owner_rust_node[0], PureChromaNode)

        wisdom_query = "deterministic semantic parity query"
        wisdom_read_token = claims_ctx.set(wisdom_reader_claims)
        try:
            python_pure_seed_read = wisdom_engine.read.get_nodes(ids=[pure_seed.id])
            python_pure_edge_read = wisdom_engine.read.get_edges(ids=[pure_edge.id])
            assert [item.id for item in python_pure_seed_read] == [pure_seed.id], (
                "Python ACL graph did not expose its public pure Wisdom node"
            )
            assert [item.id for item in python_pure_edge_read] == [pure_edge.id], (
                "Python ACL graph did not expose its public pure Wisdom edge"
            )
            python_wisdom = GraphQuery(wisdom_engine).semantic_seed_then_expand_text(
                wisdom_query, top_k=10, hops=2
            )
        finally:
            claims_ctx.reset(wisdom_read_token)
        rust_wisdom = rust_mcp_tool(
            "wisdom.semantic_seed_then_expand",
            {"text": wisdom_query, "top_k": 10, "hops": 2},
            wisdom_reader_claims,
        )
        python_wisdom_ids, python_wisdom_layers = semantic_ids(python_wisdom)
        rust_wisdom_ids, rust_wisdom_layers = semantic_ids(rust_wisdom)
        assert rust_wisdom_ids == python_wisdom_ids, (
            f"Rust/Python Wisdom seed order differs: rust={rust_wisdom_ids}, "
            f"python={python_wisdom_ids}"
        )
        assert rust_wisdom_layers == python_wisdom_layers
        assert wisdom_seed.id in rust_wisdom_ids
        assert wisdom_edge.id in set().union(
            *(layer["edges"] for layer in rust_wisdom_layers)
        )
        assert pure_seed.id in rust_wisdom_ids
        assert pure_related.id in set().union(
            *(layer["nodes"] for layer in rust_wisdom_layers)
        )
        assert pure_edge.id in set().union(
            *(layer["edges"] for layer in rust_wisdom_layers)
        )
        python_pure_seed = wisdom_engine.read.get_nodes(ids=[pure_seed.id])[0]
        python_pure_edge = wisdom_engine.read.get_edges(ids=[pure_edge.id])[0]
        assert isinstance(python_pure_seed, PureChromaNode)
        assert isinstance(python_pure_edge, PureChromaEdge)
        python_expected_endpoint_rows = edge_endpoint_rows(python_pure_edge)
        python_endpoint_projection = wisdom_backend.edge_endpoints_get(
            where={"edge_id": pure_edge.id}, include=["metadatas"]
        )
        python_endpoint_keys = set().union(
            *(row.keys() for row in python_expected_endpoint_rows)
        )
        python_actual_endpoint_rows = [
            {
                key: metadata[key]
                for key in python_endpoint_keys
                if key in metadata
            }
            for metadata in python_endpoint_projection["metadatas"]
        ]
        assert sorted(python_actual_endpoint_rows, key=lambda row: row["id"]) == sorted(
            python_expected_endpoint_rows, key=lambda row: row["id"]
        )

        owner_python_claims = {
            **owner_claims,
            "agent_id": "owner-a",
            "ns": graph_type,
        }
        for principal_id, mcp_claims, python_claims in (
            ("reader", reader_claims, reader_claims),
            ("owner", owner_claims, owner_python_claims),
        ):
            python_token = claims_ctx.set(python_claims)
            try:
                query_text = "deterministic semantic parity query"
                python_vector = engine._iterative_defensive_emb(query_text)
                python_ranked = engine.read.query_nodes(
                    query_embeddings=python_vector,
                    n_results=10,
                )
                python_ranked_ids = [
                    str(node.safe_get_id()) for node in python_ranked[0]
                ] if python_ranked else []
                python_semantic = GraphQuery(engine).semantic_seed_then_expand_text(
                    query_text, top_k=10, hops=1
                )
            finally:
                claims_ctx.reset(python_token)

            rust_vector_response = httpx.post(
                f"{base}/internal/graph/vector-query",
                headers={
                    "Authorization": f"Bearer {_hs256_token(secret, mcp_claims)}"
                },
                json={
                    "entity_kind": "node",
                    "query_embeddings": [python_vector],
                    "metric": "cosine",
                    "truth_graph": graph_type,
                    "n_results": 10,
                    "include": ["distances"],
                },
                timeout=10,
            )
            if rust_vector_response.status_code != 200:
                server_log.flush()
                details = server_log_path.read_text(encoding="utf-8", errors="replace")
                raise AssertionError(
                    f"Rust PostgreSQL vector query failed: {rust_vector_response.text}; "
                    f"server log tail:\n{details[-8000:]}"
                )
            rust_vector_payload = rust_vector_response.json()
            rust_ranked_ids = rust_vector_payload["ids"][0]
            assert rust_ranked_ids == python_ranked_ids, (
                f"{principal_id}: Python/Rust vector candidate order differs; "
                f"python={python_ranked_ids}, rust={rust_ranked_ids}, "
                f"distances={rust_vector_payload.get('distances')}"
            )
            rust_semantic = rust_mcp_tool(
                "kg_semantic_seed_then_expand_text",
                {
                    "text": "deterministic semantic parity query",
                    "top_k": 10,
                    "hops": 1,
                },
                mcp_claims,
            )
            expected_seed_ids, expected_layers = semantic_ids(python_semantic)
            actual_seed_ids, actual_layers = semantic_ids(rust_semantic)
            assert actual_seed_ids == expected_seed_ids, principal_id
            assert expected_seed_ids[0] == visible_target.id
            assert expected_seed_ids[-1] == other_doc_target.id
            assert len(actual_layers) == len(expected_layers), principal_id
            for actual, expected in zip(actual_layers, expected_layers, strict=True):
                assert actual == expected, principal_id
            if principal_id == "reader":
                assert private_target.id not in actual_seed_ids
                assert restricted_evidence_target.id not in actual_seed_ids
                assert private_endpoint_edge.id not in set().union(
                    *(layer["edges"] for layer in actual_layers)
                )
            else:
                assert private_target.id in actual_seed_ids
                assert restricted_evidence_target.id in actual_seed_ids

        def add_scoped_pair(
            *, scope_name: str, scope_value: str
        ) -> tuple[Node, Node, Edge, str]:
            scoped_doc_id = f"{doc_id}-{scope_name}"
            start = build_entity_node(
                node_id=f"{id_prefix}-{scope_name}-start",
                doc_id=scoped_doc_id,
                label=f"{scope_name} scoped start",
                metadata={"visibility": "scope", "security_scope": scope_value},
            )
            target = build_entity_node(
                node_id=f"{id_prefix}-{scope_name}-target",
                doc_id=scoped_doc_id,
                label=f"{scope_name} scoped target",
                metadata={"visibility": "scope", "security_scope": scope_value},
            )
            edge = build_relationship_edge(
                edge_id=f"{id_prefix}-{scope_name}-edge",
                src=start.id,
                tgt=target.id,
                doc_id=scoped_doc_id,
                metadata={"visibility": "scope", "security_scope": scope_value},
            )
            engine.write.add_node(start)
            engine.write.add_node(target)
            engine.write.add_edge(edge)
            return start, target, edge, scoped_doc_id

        owner_token = claims_ctx.set(owner_python_claims)
        try:
            project_scoped_pair = add_scoped_pair(
                scope_name="project-scope", scope_value="project-alpha"
            )
            tenant_scoped_pair = add_scoped_pair(
                scope_name="tenant-scope", scope_value="tenant-alpha"
            )
        finally:
            claims_ctx.reset(owner_token)

        for pair, allowed_claims, denied_claims in (
            (
                project_scoped_pair,
                {**reader_claims, "project": "project-alpha"},
                {**reader_claims, "project": "project-beta"},
            ),
            (
                tenant_scoped_pair,
                {
                    **reader_claims,
                    "tenant": "tenant-alpha",
                    "project": "project-alpha",
                },
                {
                    **reader_claims,
                    "tenant": "tenant-beta",
                    "project": "project-alpha",
                },
            ),
        ):
            scoped_start, scoped_target, scoped_edge, scoped_doc_id = pair
            for claims, should_be_visible in (
                (allowed_claims, True),
                (denied_claims, False),
            ):
                python_token = claims_ctx.set(claims)
                try:
                    python_scoped = GraphQuery(engine).neighbors(
                        scoped_start.id, doc_id=scoped_doc_id
                    )
                finally:
                    claims_ctx.reset(python_token)
                rust_scoped = rust_mcp_tool(
                    "kg_neighbors",
                    {"rid": scoped_start.id, "doc_id": scoped_doc_id},
                    claims,
                )
                assert set(rust_scoped["nodes"]) == python_scoped["nodes"]
                assert set(rust_scoped["edges"]) == python_scoped["edges"]
                if should_be_visible:
                    assert python_scoped == {
                        "nodes": {scoped_target.id},
                        "edges": {scoped_edge.id},
                    }
                else:
                    assert python_scoped == {"nodes": set(), "edges": set()}

        response = httpx.post(
            f"{base}/internal/graph/neighbors",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, reader_claims)}"
            },
            json={"rid": source.id, "doc_id": doc_id, "truth_graph": graph_type},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        rust_payload = response.json()
        assert sorted(rust_payload["nodes"]) == sorted(python_result["nodes"])
        assert sorted(rust_payload["edges"]) == sorted(python_result["edges"])
        assert rust_payload["nodes"] == [visible_target.id]
        assert rust_payload["edges"] == [visible_edge.id]

        for principal_id, mcp_claims, python_claims in (
            ("reader", reader_claims, reader_claims),
            ("owner", owner_claims, owner_python_claims),
        ):
            acl_token = claims_ctx.set(python_claims)
            try:
                python_graph = GraphQuery(engine)
                expected_edges = set(python_graph.find_edges(doc_id=doc_id))
                expected_layers = python_graph.k_hop(
                    [source.id], k=1, doc_id=doc_id
                )
                expected_private_path = python_graph.shortest_path(
                    source.id, private_target.id, doc_id=doc_id
                )
            finally:
                claims_ctx.reset(acl_token)

            mcp_edges = rust_mcp_tool(
                "kg_find_edges", {"doc_id": doc_id}, mcp_claims
            )
            assert set(mcp_edges["edges"]) == expected_edges, principal_id

            mcp_layers = rust_mcp_tool(
                "kg_k_hop",
                {"start_ids": [source.id], "k": 1, "doc_id": doc_id},
                mcp_claims,
            )["layers"]
            assert len(mcp_layers) == len(expected_layers), principal_id
            for actual, expected in zip(mcp_layers, expected_layers, strict=True):
                assert set(actual["nodes"]) == expected["nodes"], principal_id
                assert set(actual["edges"]) == expected["edges"], principal_id

            mcp_private_path = rust_mcp_tool(
                "kg_shortest_path",
                {
                    "src_id": source.id,
                    "dst_id": private_target.id,
                    "doc_id": doc_id,
                },
                mcp_claims,
            )["path"]
            assert mcp_private_path == expected_private_path, principal_id

        assert not rust_mcp_tool(
            "kg_shortest_path",
            {
                "src_id": source.id,
                "dst_id": private_target.id,
                "doc_id": doc_id,
            },
            reader_claims,
        )["path"]
        assert rust_mcp_tool(
            "kg_shortest_path",
            {
                "src_id": source.id,
                "dst_id": private_target.id,
                "doc_id": doc_id,
            },
            owner_claims,
        )["path"] == [source.id, private_endpoint_edge.id, private_target.id]

        rust_mcp_initial = rust_mcp_neighbors(reader_claims)
        assert rust_mcp_initial["nodes"] == rust_payload["nodes"]
        assert rust_mcp_initial["edges"] == rust_payload["edges"]

        writer_claims = owner_claims
        extraction = json.loads(
            (ROOT / "contracts" / "golden" / "graph-extraction-v1.json").read_text(
                encoding="utf-8"
            )
        )
        extraction["doc_id"] = f"D-rust-upsert-{id_prefix}"
        for entity in (*extraction["nodes"], *extraction["edges"]):
            for reference in entity.get("references", []):
                reference["doc_id"] = extraction["doc_id"]
            for mention in entity.get("mentions", []):
                for span in mention.get("spans", []):
                    span["doc_id"] = extraction["doc_id"]
        upsert_response = httpx.post(
            f"{base}/api/graph/upsert",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, writer_claims)}"
            },
            json=extraction,
            timeout=10,
        )
        assert upsert_response.status_code == 200, upsert_response.text
        upsert_result = upsert_response.json()
        assert upsert_result["document_id"] == extraction["doc_id"]
        assert upsert_result["nodes_added"] == 2
        assert upsert_result["edges_added"] == 1
        assert len(upsert_result["node_ids"]) == 2
        assert len(upsert_result["edge_ids"]) == 1

        owner_token = claims_ctx.set(
            {
                **writer_claims,
                "agent_id": "owner-a",
                "sub": "owner-a",
                "ns": graph_type,
                "storage_ns": namespace,
            }
        )
        try:
            python_upsert_nodes = engine.read.get_nodes(ids=upsert_result["node_ids"])
            python_upsert_edges = engine.read.get_edges(ids=upsert_result["edge_ids"])
            python_upsert_neighbors = GraphQuery(engine).neighbors(
                upsert_result["node_ids"][0], doc_id=extraction["doc_id"]
            )
        finally:
            claims_ctx.reset(owner_token)
        assert {node.id for node in python_upsert_nodes} == set(upsert_result["node_ids"])
        assert {edge.id for edge in python_upsert_edges} == set(upsert_result["edge_ids"])
        assert python_upsert_neighbors == {
            "nodes": {upsert_result["node_ids"][1]},
            "edges": {upsert_result["edge_ids"][0]},
        }

        rust_target = build_entity_node(
            node_id=f"{id_prefix}-rust-target",
            doc_id=doc_id,
            embedding=[0.25, 0.5, 0.75],
            metadata={"visibility": "public"},
        )
        target_document, target_metadata = engine.write.node_doc_and_meta(rust_target)
        target_metadata["_class_name"] = type(rust_target).__name__
        rust_edge = build_relationship_edge(
            edge_id=f"{id_prefix}-rust-edge",
            src=source.id,
            tgt=rust_target.id,
            doc_id=doc_id,
            embedding=[0.75, 0.5, 0.25],
            metadata={"visibility": "public"},
        )
        edge_document = rust_edge.model_dump_json(field_mode="backend")
        edge_metadata = engine.write.enrich_edge_meta(rust_edge)

        def mutation(
            *,
            entity_kind: str,
            table: str,
            event_id: str,
            entity: Node | Edge,
            document: str,
            metadata: dict[str, object],
            entity_id: str,
            embedding: list[float] | None,
            op: str = "ADD",
        ) -> dict[str, object]:
            payload = entity.model_dump(
                field_mode="backend", exclude=["embedding"], dump_format="json"
            )
            return {
                "scope": {
                    "namespace": namespace,
                    "workspace_id": None,
                    "graph_space": None,
                },
                "table": table,
                "entity_kind": entity_kind,
                "event_id": event_id,
                "op": op,
                "payload": payload,
                "record": {
                    "id": entity_id,
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                },
                "embedding_dim": len(embedding or []),
            }

        mutations = [
            mutation(
                entity_kind="node",
                table="gke_nodes",
                event_id=f"{id_prefix}-rust-node-event",
                entity=rust_target,
                document=target_document,
                metadata=target_metadata,
                entity_id=rust_target.id,
                embedding=rust_target.embedding or [],
            ),
            mutation(
                entity_kind="edge",
                table="gke_edges",
                event_id=f"{id_prefix}-rust-edge-event",
                entity=rust_edge,
                document=edge_document,
                metadata=edge_metadata,
                entity_id=rust_edge.id,
                embedding=rust_edge.embedding or [],
            ),
        ]
        for entity, grain in ((rust_target, "node"), (rust_edge, "edge")):
            for acl_record in engine.acl._default_acl_records_for_item(
                entity, grain=grain
            ):
                acl_node = engine.acl._build_acl_record_node(acl_record)
                acl_document, acl_metadata = engine.write.node_doc_and_meta(acl_node)
                mutations.append(
                    mutation(
                        entity_kind="node",
                        table="gke_nodes",
                        event_id=f"{id_prefix}-acl-{acl_node.id}",
                        entity=acl_node,
                        document=acl_document,
                        metadata=acl_metadata,
                        entity_id=acl_node.id,
                        embedding=None,
                    )
                )
        mutation_body = {
            "contract_version": 1,
            "atomic": True,
            "mutations": mutations,
        }
        mutation_response = httpx.post(
            f"{base}/internal/graph/mutations",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, writer_claims)}"
            },
            json=mutation_body,
            timeout=10,
        )
        assert mutation_response.status_code == 200, mutation_response.text
        assert mutation_response.json()["results"][0]["mutated"] is True
        assert mutation_response.json()["results"][1]["mutated"] is True
        assert backend.node_get(ids=[rust_target.id])["ids"] == [rust_target.id]
        assert backend.edge_get(ids=[rust_edge.id])["ids"] == [rust_edge.id]
        endpoint_rows = backend.edge_endpoints_get(
            where={"edge_id": rust_edge.id}, include=["documents"]
        )
        assert len(endpoint_rows["ids"]) == 2, endpoint_rows

        reader_token = claims_ctx.set(reader_claims)
        try:
            python_rust_node = engine.read.get_nodes(ids=[rust_target.id])
            python_rust_edge = engine.read.get_edges(ids=[rust_edge.id])
            python_after_rust_write = GraphQuery(engine).neighbors(
                source.id, doc_id=doc_id
            )
        finally:
            claims_ctx.reset(reader_token)
        assert [node.id for node in python_rust_node] == [rust_target.id]
        assert [edge.id for edge in python_rust_edge] == [rust_edge.id]
        assert python_after_rust_write == {
            "nodes": {visible_target.id, rust_target.id},
            "edges": {visible_edge.id, rust_edge.id},
        }

        replacement_edge = rust_edge.model_copy(
            update={"target_ids": [visible_target.id]}
        )
        replacement_doc = replacement_edge.model_dump_json(field_mode="backend")
        replacement_metadata = engine.write.enrich_edge_meta(replacement_edge)
        replacement_mutation = mutation(
            entity_kind="edge",
            table="gke_edges",
            event_id=f"{id_prefix}-rust-edge-replace-event",
            entity=replacement_edge,
            document=replacement_doc,
            metadata=replacement_metadata,
            entity_id=replacement_edge.id,
            embedding=replacement_edge.embedding,
            op="REPLACE",
        )
        replacement_response = httpx.post(
            f"{base}/internal/graph/mutations",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, writer_claims)}"
            },
            json={
                "contract_version": 1,
                "atomic": True,
                "mutations": [replacement_mutation],
            },
            timeout=10,
        )
        assert replacement_response.status_code == 200, replacement_response.text
        replacement_rows = backend.edge_endpoints_get(
            where={"edge_id": rust_edge.id}, include=["documents"]
        )
        replacement_endpoint_ids = {
            json.loads(document)["endpoint_id"]
            for document in replacement_rows["documents"]
        }
        assert replacement_endpoint_ids == {source.id, visible_target.id}

        reader_token = claims_ctx.set(reader_claims)
        try:
            python_after_replace = GraphQuery(engine).neighbors(
                source.id, doc_id=doc_id
            )
        finally:
            claims_ctx.reset(reader_token)
        rust_after_replace_response = httpx.post(
            f"{base}/internal/graph/neighbors",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, reader_claims)}"
            },
            json={"rid": source.id, "doc_id": doc_id, "truth_graph": graph_type},
            timeout=10,
        )
        assert rust_after_replace_response.status_code == 200
        rust_after_replace = rust_after_replace_response.json()
        assert set(rust_after_replace["nodes"]) == python_after_replace["nodes"]
        assert set(rust_after_replace["edges"]) == python_after_replace["edges"]
        assert python_after_replace == {
            "nodes": {visible_target.id},
            "edges": {visible_edge.id, rust_edge.id},
        }
        rust_mcp_after_replace = rust_mcp_neighbors(reader_claims)
        assert set(rust_mcp_after_replace["nodes"]) == python_after_replace["nodes"]
        assert set(rust_mcp_after_replace["edges"]) == python_after_replace["edges"]

        tombstone_edge = replacement_edge.model_copy(
            update={
                "metadata": {
                    **(replacement_edge.metadata or {}),
                    "lifecycle_status": "tombstoned",
                }
            }
        )
        tombstone_mutation = mutation(
            entity_kind="edge",
            table="gke_edges",
            event_id=f"{id_prefix}-rust-edge-tombstone-event",
            entity=tombstone_edge,
            document=tombstone_edge.model_dump_json(field_mode="backend"),
            metadata=engine.write.enrich_edge_meta(tombstone_edge),
            entity_id=tombstone_edge.id,
            embedding=tombstone_edge.embedding,
            op="TOMBSTONE",
        )
        tombstone_response = httpx.post(
            f"{base}/internal/graph/mutations",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, writer_claims)}"
            },
            json={
                "contract_version": 1,
                "atomic": True,
                "mutations": [tombstone_mutation],
            },
            timeout=10,
        )
        assert tombstone_response.status_code == 200, tombstone_response.text
        tombstone_rows = backend.edge_endpoints_get(
            where={"edge_id": rust_edge.id}, include=["documents"]
        )
        assert tombstone_rows["ids"] == []

        reader_token = claims_ctx.set(reader_claims)
        try:
            python_after_tombstone = GraphQuery(engine).neighbors(
                source.id, doc_id=doc_id
            )
        finally:
            claims_ctx.reset(reader_token)
        rust_after_tombstone_response = httpx.post(
            f"{base}/internal/graph/neighbors",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, reader_claims)}"
            },
            json={"rid": source.id, "doc_id": doc_id, "truth_graph": graph_type},
            timeout=10,
        )
        assert rust_after_tombstone_response.status_code == 200
        rust_after_tombstone = rust_after_tombstone_response.json()
        assert set(rust_after_tombstone["nodes"]) == python_after_tombstone["nodes"]
        assert set(rust_after_tombstone["edges"]) == python_after_tombstone["edges"]
        assert python_after_tombstone == {
            "nodes": {visible_target.id},
            "edges": {visible_edge.id},
        }
        rust_mcp_after_tombstone = rust_mcp_neighbors(reader_claims)
        assert set(rust_mcp_after_tombstone["nodes"]) == python_after_tombstone["nodes"]
        assert set(rust_mcp_after_tombstone["edges"]) == python_after_tombstone["edges"]

        conversation_id = f"{id_prefix}-python-written-conversation"
        # Rust stores graph types in its configured schema and partitions them
        # by namespace, so the Python producer must target that same schema.
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "python_persisted_conversation"),
            backend=conversation_backend,
            namespace="conversation",
            kg_graph_type="conversation",
            embedding_function=fixture_embedding,
            embedding_profile=EmbeddingProfile(
                provider="ollama",
                model="fixture-embedding-v1",
                dimension=3,
                similarity_metric="cosine",
                embedding_kind="single_vector",
            ),
            acl_enabled=True,
        )
        conversation_owner = {
            "agent_id": "conversation-owner",
            "sub": "conversation-owner",
            "role": "rw",
            "ns": "conversation",
            "storage_ns": "conversation",
        }
        conversation_write_token = claims_ctx.set(conversation_owner)
        try:
            for turn_index, (role, content) in enumerate(
                (("user", "Python persisted question"), ("assistant", "Python persisted answer"))
            ):
                conversation_engine.write.add_node(
                    ConversationNode(
                        id=f"{id_prefix}-conversation-turn-{turn_index}",
                        label=f"Conversation {role} turn",
                        type="entity",
                        summary=content,
                        role=role,
                        turn_index=turn_index,
                        conversation_id=conversation_id,
                        user_id="conversation-owner",
                        metadata={
                            "entity_type": "conversation_turn",
                            "in_conversation_chain": True,
                            "in_ui_chain": True,
                            "level_from_root": 0,
                            "visibility": "public",
                        },
                        mentions=[
                            mk_document_grounding(f"conversation:{conversation_id}")
                        ],
                    )
                )
        finally:
            claims_ctx.reset(conversation_write_token)

        conversation_read_token = claims_ctx.set(conversation_owner)
        try:
            python_conversation_nodes = conversation_engine.read.get_nodes(
                where={"conversation_id": conversation_id}
            )
        finally:
            claims_ctx.reset(conversation_read_token)
        python_transcript = [
            {
                "node_id": node.id,
                "turn_index": node.turn_index,
                "role": node.role,
                "content": node.summary,
                "entity_type": node.metadata["entity_type"],
            }
            for node in sorted(
                python_conversation_nodes,
                key=lambda node: (node.turn_index, node.id),
            )
            if node.metadata.get("entity_type")
            in {"conversation_turn", "assistant_turn"}
        ]
        rust_transcript = rust_syscall(
            "invoke_tool",
            {
                "tool_name": "list_transcript",
                "conversation_id": conversation_id,
            },
            {
                "sub": "conversation-owner",
                "role": "rw",
                "roles": ["reader"],
                "ns": "conversation",
                "storage_ns": "conversation",
                "capabilities": ["invoke_tool", "read_graph"],
                "exp": 4_102_444_800,
            },
        )
        assert rust_transcript["version"] == "v1"
        assert rust_transcript["op"] == "invoke_tool"
        assert rust_transcript["status"] == "ok"
        assert rust_transcript["result"] == {
            "tool_name": "list_transcript",
            "conversation_id": conversation_id,
            "messages": python_transcript,
        }

        rust_conversation_id = f"{id_prefix}-rust-written-conversation"
        conversation_claims = {
            "sub": "conversation-owner",
            "role": "rw",
            "roles": ["reader"],
            "ns": "conversation",
            "storage_ns": "conversation",
            "capabilities": ["spawn_process", "read_graph"],
            "exp": 4_102_444_800,
        }
        rust_create = httpx.post(
            f"{base}/api/conversations",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, conversation_claims)}"
            },
            json={
                "user_id": "conversation-owner",
                "conversation_id": rust_conversation_id,
            },
            timeout=10,
        )
        assert rust_create.status_code == 200, rust_create.text
        assert rust_create.json()["conversation_id"] == rust_conversation_id
        rust_start_node_id = rust_create.json()["start_node_id"]

        workflow_id = "debug.rag.v1"
        workflow_contract = json.loads(
            (ROOT / "contracts/golden/answer-workflow-runtime-contract-v1.json")
            .read_text(encoding="utf-8")
        )
        workflow_spec = workflow_contract["workflows"][workflow_id]
        workflow_claims = {
            "sub": "conversation-owner",
            "role": "rw",
            "roles": ["reader"],
            "ns": "workflow",
            "storage_ns": "workflow",
            "capabilities": ["workflow.design.write", "workflow.design.inspect"],
            "exp": 4_102_444_800,
        }
        python_worker_operations = {
            "start",
            "aa_debug_select_top_candidates",
            "end",
        }
        rust_native_operations = {
            "aa_prepare",
            "aa_get_view_and_question",
            "aa_retrieve_candidates",
            "aa_debug_answer_from_nodes",
            "aa_project_pointers",
            "aa_persist_response",
            "link_assistant_turn",
        }
        assert set(workflow_spec["required_ops"]) == (
            python_worker_operations | rust_native_operations
        )
        for operation in workflow_spec["required_ops"]:
            executor_target = (
                "python-worker"
                if operation in python_worker_operations
                else "rust-native"
                if operation in rust_native_operations
                else None
            )
            metadata = {"wf_version": workflow_spec["python_op_version"]}
            if executor_target is not None:
                metadata["wf_executor_target"] = executor_target
            design_response = httpx.post(
                f"{base}/api/workflow/design/{workflow_id}/nodes",
                headers={
                    "Authorization": f"Bearer {_hs256_token(secret, workflow_claims)}"
                },
                json={
                    "designer_id": "conversation-owner",
                    "node_id": f"node-{operation}",
                    "label": operation,
                    "op": operation,
                    "start": operation == "start",
                    "terminal": operation == "end",
                    "metadata": metadata,
                },
                timeout=10,
            )
            assert design_response.status_code == 200, design_response.text
        for index, route in enumerate(workflow_spec["required_routes"]):
            design_response = httpx.post(
                f"{base}/api/workflow/design/{workflow_id}/edges",
                headers={
                    "Authorization": f"Bearer {_hs256_token(secret, workflow_claims)}"
                },
                json={
                    "designer_id": "conversation-owner",
                    "edge_id": f"edge-{index}",
                    "src": f"node-{route['from']}",
                    "dst": f"node-{route['to']}",
                    "predicate": route["predicate"],
                    "priority": route["priority"],
                    "is_default": route["default"],
                    "multiplicity": route["multiplicity"],
                },
                timeout=10,
            )
            assert design_response.status_code == 200, design_response.text

        turn_claims = {
            **conversation_claims,
            "ns": ["conversation", "workflow"],
            "capabilities": [
                "spawn_process",
                "send_message",
                "write_graph",
                "read_graph",
                "workflow.run.write",
            ],
        }
        rust_turn = httpx.post(
            f"{base}/api/conversations/{rust_conversation_id}/turns:answer",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
            json={"text": "Rust persisted question", "workflow_id": workflow_id},
            timeout=10,
        )
        assert rust_turn.status_code == 202, rust_turn.text
        rust_turn_payload = rust_turn.json()
        assert rust_turn_payload["workflow_id"] == workflow_id
        assert rust_turn_payload["status"] == "queued"
        rust_user_turn_id = rust_turn_payload["user_turn_node_id"]

        from kogwistar.conversation.resolvers import default_resolver

        monkeypatch.setattr(
            conversation_engine.write,
            "add_node",
            lambda *_args, **_kwargs: pytest.fail(
                "Rust-owned answer worker must not write nodes through Python"
            ),
        )
        monkeypatch.setattr(
            conversation_engine.write,
            "add_edge",
            lambda *_args, **_kwargs: pytest.fail(
                "Rust-owned answer worker must not write edges through Python"
            ),
        )
        with RustRuntimeWorker.from_step_resolver(
            base_url=base,
            worker_id="python-answer-worker",
            journal_path=tmp_path / "rust-admitted-answer-worker.sqlite",
            step_resolver=default_resolver,
            dependency_provider=lambda _work: {
                "agent": SimpleNamespace(config=SimpleNamespace(max_used=3))
            },
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
        ) as answer_worker:
            processed_ops: list[str] = []
            worker_deadline = time.monotonic() + 30
            while time.monotonic() < worker_deadline:
                owner_context = claims_ctx.set(conversation_owner)
                try:
                    claimed = answer_worker.claim(
                        run_id=rust_turn_payload["run_id"], limit=1
                    )
                    if claimed:
                        processed_ops.append(claimed[0]["payload"]["op"])
                        result = answer_worker.process(claimed[0])
                        assert result["event_seq"] > 0
                finally:
                    claims_ctx.reset(owner_context)
                current_run = httpx.get(
                    f"{base}/api/workflow/runs/{rust_turn_payload['run_id']}",
                    headers={
                        "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
                    },
                    timeout=10,
                )
                assert current_run.status_code == 200, current_run.text
                current_status = current_run.json()["status"]
                if current_status == "succeeded":
                    break
                assert current_status not in {"failed", "cancelled"}, (
                    current_run.json().get("error")
                )
                if not claimed:
                    time.sleep(0.05)
            else:
                pytest.fail("Rust/Python answer workflow did not finish in 30 seconds")
            assert processed_ops == [
                "start",
                "aa_debug_select_top_candidates",
                "end",
            ]
        rust_answer_run = httpx.get(
            f"{base}/api/workflow/runs/{rust_turn_payload['run_id']}",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
            timeout=10,
        )
        assert rust_answer_run.status_code == 200, rust_answer_run.text
        answer_run_payload = rust_answer_run.json()
        assert answer_run_payload["status"] == "succeeded", answer_run_payload.get(
            "error"
        )
        rust_answer_checkpoints = httpx.get(
            f"{base}/api/workflow/runs/{rust_turn_payload['run_id']}/checkpoints",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
            timeout=10,
        )
        assert rust_answer_checkpoints.status_code == 200, rust_answer_checkpoints.text
        rust_answer_state = rust_answer_checkpoints.json()["checkpoints"][-1]["state"]
        assert rust_answer_state["answer"]["mode"] == "debug_rag"
        assert rust_answer_state["agentic_answering_result"]["assistant_text"]
        rust_assistant_turn_id = rust_answer_state["agentic_answering_result"][
            "assistant_turn_node_id"
        ]

        rust_syscall_turn = rust_mcp_tool(
            "conversation.ask",
            {
                "conversation_id": rust_conversation_id,
                "text": "Rust persisted syscall question",
                "workflow_id": workflow_id,
            },
            turn_claims,
            endpoint="/mcp/conversation",
        )
        assert rust_syscall_turn["run_id"]
        assert rust_syscall_turn["turn_node_id"]
        rust_syscall_turn_id = rust_syscall_turn["turn_node_id"]
        with RustRuntimeWorker.from_step_resolver(
            base_url=base,
            worker_id="python-mcp-answer-worker",
            journal_path=tmp_path / "rust-admitted-mcp-answer-worker.sqlite",
            step_resolver=default_resolver,
            dependency_provider=lambda _work: {
                "agent": SimpleNamespace(config=SimpleNamespace(max_used=3))
            },
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
        ) as mcp_answer_worker:
            processed_mcp_ops: list[str] = []
            worker_deadline = time.monotonic() + 30
            while time.monotonic() < worker_deadline:
                owner_context = claims_ctx.set(conversation_owner)
                try:
                    claimed = mcp_answer_worker.claim(
                        run_id=rust_syscall_turn["run_id"], limit=1
                    )
                    if claimed:
                        processed_mcp_ops.append(claimed[0]["payload"]["op"])
                        result = mcp_answer_worker.process(claimed[0])
                        assert result["event_seq"] > 0
                finally:
                    claims_ctx.reset(owner_context)
                mcp_run = httpx.get(
                    f"{base}/api/workflow/runs/{rust_syscall_turn['run_id']}",
                    headers={
                        "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
                    },
                    timeout=10,
                )
                assert mcp_run.status_code == 200, mcp_run.text
                mcp_status = mcp_run.json()["status"]
                if mcp_status == "succeeded":
                    break
                assert mcp_status not in {"failed", "cancelled"}, (
                    mcp_run.json().get("error")
                )
                if not claimed:
                    time.sleep(0.05)
            else:
                pytest.fail("Rust/Python MCP answer workflow did not finish in 30 seconds")
            assert processed_mcp_ops == [
                "start",
                "aa_debug_select_top_candidates",
                "end",
            ]
        mcp_answer_checkpoints = httpx.get(
            f"{base}/api/workflow/runs/{rust_syscall_turn['run_id']}/checkpoints",
            headers={
                "Authorization": f"Bearer {_hs256_token(secret, turn_claims)}"
            },
            timeout=10,
        )
        assert mcp_answer_checkpoints.status_code == 200, mcp_answer_checkpoints.text
        mcp_answer_state = mcp_answer_checkpoints.json()["checkpoints"][-1]["state"]
        assert mcp_answer_state["answer"]["mode"] == "debug_rag"
        assert mcp_answer_state["agentic_answering_result"]["assistant_text"]
        mcp_assistant_edge_id = mcp_answer_state["assistant_turn_edge_id"]
        mcp_assistant_turn_id = mcp_answer_state["agentic_answering_result"][
            "assistant_turn_node_id"
        ]

        conversation_read_token = claims_ctx.set(conversation_owner)
        try:
            python_rust_nodes = conversation_engine.read.get_nodes(
                where={"conversation_id": rust_conversation_id}
            )
            rest_next_edges = conversation_engine.read.get_edges(
                ids=[rust_answer_state["assistant_turn_edge_id"]], edge_type=Edge
            )
            mcp_next_edges = conversation_engine.read.get_edges(
                ids=[mcp_assistant_edge_id], edge_type=Edge
            )
        finally:
            claims_ctx.reset(conversation_read_token)
        assert any(node.id == rust_start_node_id for node in python_rust_nodes)
        assert len(rest_next_edges) == 1
        assert rest_next_edges[0].source_ids == [rust_user_turn_id]
        assert rest_next_edges[0].target_ids == [rust_assistant_turn_id]
        assert rest_next_edges[0].relation == "next_turn"
        assert len(mcp_next_edges) == 1
        assert mcp_next_edges[0].source_ids == [rust_syscall_turn_id]
        assert mcp_next_edges[0].target_ids == [mcp_assistant_turn_id]
        assert mcp_next_edges[0].relation == "next_turn"
        python_rust_transcript = [
            {
                "node_id": node.id,
                "turn_index": node.turn_index,
                "role": node.role,
                "content": node.summary,
                "entity_type": node.metadata["entity_type"],
            }
            for node in sorted(
                python_rust_nodes,
                key=lambda node: (
                    node.turn_index if isinstance(node.turn_index, int) else -1,
                    node.id,
                ),
            )
            if node.metadata.get("entity_type")
            in {"conversation_turn", "assistant_turn"}
        ]
        expected_python_rust_transcript = [
            {
                "node_id": rust_user_turn_id,
                "turn_index": 0,
                "role": "user",
                "content": "Rust persisted question",
                "entity_type": "conversation_turn",
            },
            {
                "node_id": rust_assistant_turn_id,
                "turn_index": 1,
                "role": "assistant",
                "content": rust_answer_state["agentic_answering_result"][
                    "assistant_text"
                ],
                "entity_type": "assistant_turn",
            },
            {
                "node_id": rust_syscall_turn_id,
                "turn_index": 2,
                "role": "user",
                "content": "Rust persisted syscall question",
                "entity_type": "conversation_turn",
            },
            {
                "node_id": mcp_assistant_turn_id,
                "turn_index": 3,
                "role": "assistant",
                "content": mcp_answer_state["agentic_answering_result"][
                    "assistant_text"
                ],
                "entity_type": "assistant_turn",
            },
        ]
        assert python_rust_transcript == expected_python_rust_transcript, (
            "conversation transcript mismatch after completing REST and MCP-admitted "
            "runs: "
            f"actual={python_rust_transcript!r}; "
            f"expected={expected_python_rust_transcript!r}; "
            f"mcp_initial_turn_index={mcp_answer_state.get('turn_index')!r}; "
            f"mcp_run_node_id={mcp_answer_state.get('run_node_id')!r}; "
            f"rest_assistant_id={rust_assistant_turn_id!r}; "
            f"mcp_assistant_id={mcp_assistant_turn_id!r}"
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        server_log.close()
        embedding_server.shutdown()
        embedding_server.server_close()
        embedding_thread.join(timeout=2)
