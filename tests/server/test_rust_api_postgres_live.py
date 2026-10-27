from __future__ import annotations

import base64
import hashlib
import hmac
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


def test_rust_postgres_graph_neighbors_match_python_over_shared_projection(
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

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Edge, Node
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
    from kogwistar.graph_query import GraphQuery
    from kogwistar.server.auth_middleware import claims_ctx
    from tests._helpers.embeddings import build_test_embedding_function
    from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

    namespace = "alpha"
    graph_type = "knowledge"
    doc_id = "D-neighbor-parity"
    id_prefix = uuid.uuid4().hex
    source = build_entity_node(
        node_id=f"{id_prefix}-source",
        doc_id=doc_id,
        metadata={"visibility": "public"},
    )
    visible_target = build_entity_node(
        node_id=f"{id_prefix}-visible-target",
        doc_id=doc_id,
        metadata={"visibility": "public"},
    )
    private_target = build_entity_node(
        node_id=f"{id_prefix}-private-target",
        doc_id=doc_id,
        metadata={"visibility": "private", "owner_agent_id": "owner-a"},
    )
    other_doc_target = build_entity_node(
        node_id=f"{id_prefix}-other-doc-target",
        doc_id="D-neighbor-parity-other",
        metadata={"visibility": "public"},
    )
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

    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=pg_schema)
    backend.ensure_schema()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "shared_postgres_graph"),
        backend=backend,
        namespace=namespace,
        kg_graph_type=graph_type,
        embedding_function=build_test_embedding_function("constant", dim=3),
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
        for node in (source, visible_target, private_target, other_doc_target):
            engine.write.add_node(node)
        for edge in (visible_edge, private_endpoint_edge, other_doc_edge):
            engine.write.add_edge(edge)
    finally:
        claims_ctx.reset(owner_token)

    reader_claims: dict[str, object] = {
        "sub": "reader-b",
        "role": "ro",
        "roles": ["reader"],
        "ns": [graph_type],
        "storage_ns": namespace,
        "capabilities": ["read_graph"],
        "exp": 4_102_444_800,
    }
    reader_token = claims_ctx.set(reader_claims)
    try:
        python_result = GraphQuery(engine).neighbors(source.id, doc_id=doc_id)
    finally:
        claims_ctx.reset(reader_token)

    port = _port()
    secret = uuid.uuid4().hex
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "AUTH_MODE": "jwt",
            "JWT_SECRET": secret,
            "JWT_ALG": "HS256",
            "KOGWISTAR_BACKEND": "pg",
            "KOGWISTAR_PG_DSN": pg_dsn,
            "KOGWISTAR_PG_SCHEMA_BASE": pg_schema,
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
        writer_claims = {
            **reader_claims,
            "role": "rw",
            "roles": ["reader", "rw"],
            "capabilities": ["read_graph", "write_graph"],
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
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
