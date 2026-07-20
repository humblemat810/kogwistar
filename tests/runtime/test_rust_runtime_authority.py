from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import RunResult, WorkflowRuntime
from kogwistar.runtime.rust_runtime_authority import (
    RustRuntimeAuthority,
    RustRuntimeAuthorityError,
    freeze_runtime_plan,
)


ROOT = Path(__file__).resolve().parents[2]
RUST_SERVER = (
    ROOT
    / "rust"
    / "target"
    / "debug"
    / ("kogwistar-server.exe" if os.name == "nt" else "kogwistar-server")
)


def test_public_sync_runtime_dispatches_to_configured_rust_authority(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", "http://rust")
    expected = RunResult(
        run_id="run-1",
        final_state={"answer": 42},
        mq=__import__("queue").Queue(),
    )
    calls: list[dict[str, Any]] = []

    def dispatch(_runtime: WorkflowRuntime, **kwargs: Any) -> RunResult:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.run_with_rust_authority",
        dispatch,
    )
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = SimpleNamespace(_state_schema={})

    result = runtime.run(
        workflow_id="wf-1",
        conversation_id="conv-1",
        turn_node_id="turn-1",
        initial_state={"seed": 1},
        run_id="run-1",
        cache_dir="cache",
    )

    assert result is expected
    assert calls == [
        {
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "turn_node_id": "turn-1",
            "initial_state": {"seed": 1},
            "run_id": "run-1",
            "cache_dir": "cache",
        }
    ]


def test_public_sync_runtime_rust_mode_without_authority_url_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.delenv("KOGWISTAR_RUST_RUNTIME_URL", raising=False)
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = SimpleNamespace(_state_schema={})

    with pytest.raises(RustRuntimeAuthorityError, match="URL is not configured"):
        runtime.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={},
            run_id="run-1",
        )


def test_authority_pumps_only_target_run_and_restores_process_local_deps(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR", str(tmp_path))
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"entry": 0},
    )
    node = SimpleNamespace(
        id="entry",
        label="Entry",
        op="start",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "start"},
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"entry": node}, {"entry": []}),
    )

    resolver = MappingStepResolver()
    deps = {"injected": object()}

    @resolver.register("start")
    def start(ctx):
        assert ctx.state_view["_deps"] is deps
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"answer": 42})],
        )

    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=2,
        cancel_requested=None,
    )
    state = {"done": False, "claimed": False}
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "POST" and request.url.path == "/api/workflow/runs":
            assert isinstance(body, dict)
            assert body["run_id"] == "run-1"
            assert body["start_node_id"] == "entry"
            assert body["node_ops"] == {"entry": "start"}
            assert body["initial_state"] == {"seed": 1}
            return httpx.Response(
                202,
                json={"run_id": "run-1", "admission": "accepted"},
                request=request,
            )
        if request.method == "GET" and request.url.path == "/api/workflow/runs/run-1":
            status = "succeeded" if state["done"] else "queued"
            return httpx.Response(
                200,
                json={"run_id": "run-1", "status": status, "error": None},
                request=request,
            )
        if request.method == "POST" and request.url.path == "/internal/runtime/claim":
            assert isinstance(body, dict)
            assert body == {
                "claimed_by": body["claimed_by"],
                "limit": 2,
                "lease_seconds": 60,
                "run_id": "run-1",
            }
            if state["claimed"]:
                work: list[dict[str, Any]] = []
            else:
                state["claimed"] = True
                work = [
                    {
                        "message_id": "lane-1",
                        "claimed_by": body["claimed_by"],
                        "run_id": "run-1",
                        "step_id": "entry",
                        "correlation_id": "run-1",
                        "payload": {
                            "run_id": "run-1",
                            "workflow_id": "wf-1",
                            "conversation_id": "conv-1",
                            "turn_node_id": "turn-1",
                            "node_id": "entry",
                            "op": "start",
                            "step_seq": 0,
                            "token_id": "run-1",
                            "parent_token_id": None,
                            "state": {"seed": 1},
                            "runtime_routes": [],
                        },
                    }
                ]
            return httpx.Response(200, json={"work": work}, request=request)
        if request.method == "POST" and request.url.path == "/internal/runtime/results":
            assert isinstance(body, dict)
            assert body["effect"]["state_update"] == [["u", {"answer": 42}]]
            state["done"] = True
            return httpx.Response(200, json={"event_seq": 2}, request=request)
        if (
            request.method == "GET"
            and request.url.path == "/api/workflow/runs/run-1/checkpoints"
        ):
            return httpx.Response(
                200,
                json={
                    "run_id": "run-1",
                    "checkpoints": [
                        {"step_seq": 0, "event_seq": 2, "state": {"seed": 1, "answer": 42}}
                    ],
                },
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://rust",
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    with client:
        result = authority.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={"seed": 1, "_deps": deps},
            run_id="run-1",
        )

    assert result.status == "succeeded"
    assert result.final_state == {"seed": 1, "answer": 42, "_deps": deps}
    claim_bodies = [
        body
        for method, path, body in requests
        if method == "POST" and path.endswith("/claim") and body is not None
    ]
    assert claim_bodies and all(body["run_id"] == "run-1" for body in claim_bodies)


@pytest.mark.integration
def test_public_authority_runs_through_real_rust_sqlite_server(
    monkeypatch,
    tmp_path,
) -> None:
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live authority test")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    node = SimpleNamespace(
        id="entry",
        label="Entry",
        op="start",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "start"},
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"entry": node}, {"entry": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"entry": 0},
    )
    resolver = MappingStepResolver()

    @resolver.register("start")
    def start(ctx):
        assert ctx.state_view["seed"] == 1
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"answer": 42})],
        )

    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.workflow_engine = object()
    runtime.step_resolver = resolver
    runtime.predicate_registry = {}
    runtime.max_workers = 1
    runtime.cancel_requested = None
    sqlite_path = tmp_path / "authority.sqlite3"
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "sqlite",
            "KOGWISTAR_META_SQLITE_PATH": str(sqlite_path),
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "",
        }
    )
    process = subprocess.Popen(
        [str(RUST_SERVER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise AssertionError(f"Rust server exited before ready: {stderr}")
            try:
                if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise AssertionError("Rust SQLite server did not become ready")
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", base_url)
        monkeypatch.setenv(
            "KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR",
            str(tmp_path / "journal"),
        )
        result = runtime.run(
            workflow_id="wf-live",
            conversation_id="conv-live",
            turn_node_id="turn-live",
            initial_state={"seed": 1},
            run_id="run-live",
            cache_dir=tmp_path,
        )
        assert result.status == "succeeded"
        assert result.final_state["seed"] == 1
        assert result.final_state["answer"] == 42
        assert result.final_state["_rt_join"] == {
            "pending": [],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        }
        status = httpx.get(f"{base_url}/api/workflow/runs/run-live", timeout=10)
        assert status.status_code == 200
        assert status.json()["status"] == "succeeded"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def test_submit_retries_once_after_lost_response_with_same_identity(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    attempts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        attempts.append(body)
        if len(attempts) == 1:
            raise httpx.ReadError("response lost", request=request)
        return httpx.Response(
            202,
            json={"run_id": "run-1", "admission": "accepted", "idempotent": True},
            request=request,
        )

    runtime = SimpleNamespace(
        step_resolver=lambda _op: None,
        predicate_registry={},
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://rust",
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    with client:
        response = authority._submit_run(
            {"run_id": "run-1", "workflow_id": "wf-1"}
        )

    assert response["idempotent"] is True
    assert attempts == [attempts[0], attempts[0]]


def test_authority_propagates_cancellation_without_running_other_work(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    state = {"cancelled": False}
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/api/workflow/runs/run-1":
            return httpx.Response(
                200,
                json={
                    "run_id": "run-1",
                    "status": "cancelled" if state["cancelled"] else "queued",
                    "error": None,
                },
                request=request,
            )
        if request.method == "POST" and request.url.path == "/api/workflow/runs/run-1/cancel":
            state["cancelled"] = True
            return httpx.Response(200, json={"cancel_requested": True}, request=request)
        if request.method == "POST" and request.url.path == "/internal/runtime/claim":
            assert isinstance(body, dict)
            assert body["run_id"] == "run-1"
            return httpx.Response(200, json={"work": []}, request=request)
        if (
            request.method == "GET"
            and request.url.path == "/api/workflow/runs/run-1/checkpoints"
        ):
            return httpx.Response(200, json={"checkpoints": []}, request=request)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    runtime = SimpleNamespace(
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        max_workers=1,
        cancel_requested=lambda run_id: run_id == "run-1",
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://rust",
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        result = authority._pump_existing_run(
            run_id="run-1",
            durable_initial={"seed": 1},
            dependencies={},
        )
    finally:
        authority.close()
        client.close()

    assert result.status == "cancelled"
    assert result.final_state == {"seed": 1}
    assert sum(path.endswith("/cancel") for _method, path, _body in requests) == 1


def test_new_authority_instance_continues_existing_run_after_scheduler_restart(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    state = {"done": False, "claimed": False}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        if request.method == "GET" and request.url.path == "/api/workflow/runs/run-1":
            return httpx.Response(
                200,
                json={
                    "run_id": "run-1",
                    "status": "succeeded" if state["done"] else "queued",
                    "error": None,
                },
                request=request,
            )
        if request.method == "POST" and request.url.path == "/internal/runtime/claim":
            assert isinstance(body, dict)
            assert body["run_id"] == "run-1"
            if state["claimed"]:
                work: list[dict[str, Any]] = []
            else:
                state["claimed"] = True
                work = [
                    {
                        "message_id": "lane-recovered",
                        "claimed_by": body["claimed_by"],
                        "run_id": "run-1",
                        "step_id": "finish",
                        "correlation_id": "run-1",
                        "payload": {
                            "run_id": "run-1",
                            "workflow_id": "wf-1",
                            "conversation_id": "conv-1",
                            "node_id": "finish",
                            "op": "finish",
                            "step_seq": 1,
                            "token_id": "token-1",
                            "parent_token_id": None,
                            "state": {"before_restart": True},
                            "runtime_routes": [],
                        },
                    }
                ]
            return httpx.Response(200, json={"work": work}, request=request)
        if request.method == "POST" and request.url.path == "/internal/runtime/results":
            assert isinstance(body, dict)
            assert body["effect"]["state_update"] == [
                ["u", {"after_restart": True}]
            ]
            state["done"] = True
            return httpx.Response(200, json={"event_seq": 4}, request=request)
        if (
            request.method == "GET"
            and request.url.path == "/api/workflow/runs/run-1/checkpoints"
        ):
            return httpx.Response(
                200,
                json={
                    "checkpoints": [
                        {
                            "step_seq": 1,
                            "event_seq": 4,
                            "state": {
                                "before_restart": True,
                                "after_restart": True,
                            },
                        }
                    ]
                },
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    resolver = MappingStepResolver()

    @resolver.register("finish")
    def finish(ctx):
        assert ctx.state_view["before_restart"] is True
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"after_restart": True})],
        )

    runtime = SimpleNamespace(
        step_resolver=resolver,
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://rust",
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        result = authority._pump_existing_run(
            run_id="run-1",
            durable_initial={"before_restart": True},
            dependencies={},
        )
    finally:
        authority.close()
        client.close()

    assert result.status == "succeeded"
    assert result.final_state == {
        "before_restart": True,
        "after_restart": True,
    }


def test_freeze_runtime_plan_preserves_route_alias_fanout_and_join_mask(
    monkeypatch,
) -> None:
    start = SimpleNamespace(
        id="wf|start",
        label="Start",
        op="choose",
        fanout=True,
        metadata={"wf_start": True, "wf_op": "choose"},
    )
    join = SimpleNamespace(
        id="wf|join",
        label="Merge",
        op="join",
        fanout=False,
        metadata={"wf_join": True, "wf_op": "join"},
    )
    edge = SimpleNamespace(
        id="edge-1",
        label="to-merge",
        source_ids=["wf|start"],
        target_ids=["wf|join"],
        predicate="ready",
        multiplicity="many",
        is_default=False,
        priority=7,
        safe_get_id=lambda: "edge-1",
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (
            start,
            {"wf|start": start, "wf|join": join},
            {"wf|start": [edge], "wf|join": []},
        ),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"wf|start": 1, "wf|join": 1},
    )
    runtime = SimpleNamespace(
        workflow_engine=object(),
        predicate_registry={"ready": lambda *_args: True},
        step_resolver=lambda _op: None,
    )

    plan = freeze_runtime_plan(runtime, "wf-1")

    assert plan["start_node_id"] == "wf|start"
    assert plan["start_join_mask"] == 1
    assert plan["join_node_ids"] == ["wf|join"]
    assert plan["node_ops"] == {"wf|start": "choose", "wf|join": "join"}
    assert plan["runtime_routes"] == [
        {
            "edge_id": "edge-1",
            "source_node_id": "wf|start",
            "target_node_id": "wf|join",
            "aliases": ["to-merge", "wf|join", "join", "Merge"],
            "join_mask": 1,
            "predicate": "ready",
            "multiplicity": "many",
            "is_default": False,
            "priority": 7,
            "source_fanout": True,
        }
    ]


def test_async_runtime_fails_closed_when_rust_authority_is_selected(
    monkeypatch,
) -> None:
    from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", "http://rust")
    runtime = AsyncWorkflowRuntime.__new__(AsyncWorkflowRuntime)

    async def invoke() -> None:
        await runtime.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={},
        )

    try:
        asyncio.run(invoke())
    except NotImplementedError as error:
        assert "async resolver callbacks" in str(error)
    else:
        raise AssertionError("async Rust authority must fail closed")


def test_async_runtime_rust_mode_without_authority_url_fails_closed(
    monkeypatch,
) -> None:
    from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.delenv("KOGWISTAR_RUST_RUNTIME_URL", raising=False)
    runtime = AsyncWorkflowRuntime.__new__(AsyncWorkflowRuntime)

    async def invoke() -> None:
        await runtime.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={},
        )

    with pytest.raises(NotImplementedError, match="async resolver callbacks"):
        asyncio.run(invoke())


def test_public_resume_dispatches_to_rust_authority(monkeypatch) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", "http://rust")
    expected = RunResult(
        run_id="run-1",
        final_state={"approved": True},
        mq=__import__("queue").Queue(),
    )
    calls: list[dict[str, Any]] = []

    class FakeAuthority:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def resume(self, **kwargs: Any) -> RunResult:
            calls.append(kwargs)
            return expected

        def close(self) -> None:
            calls.append({"closed": True})

    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.RustRuntimeAuthority",
        FakeAuthority,
    )
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    result = RunSuccess(conversation_node_id=None, state_update=[])

    resumed = runtime.resume_run(
        run_id="run-1",
        suspended_node_id="gate",
        suspended_token_id="token-1",
        client_result=result,
        workflow_id="wf-1",
        conversation_id="conv-1",
        turn_node_id="turn-1",
    )

    assert resumed is expected
    assert calls[0] == {
        "run_id": "run-1",
        "suspended_node_id": "gate",
        "suspended_token_id": "token-1",
        "client_result": result,
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "turn_node_id": "turn-1",
    }
    assert calls[1] == {"closed": True}


def test_resume_effect_uses_existing_update_and_route_semantics(
    monkeypatch,
    tmp_path,
) -> None:
    start = SimpleNamespace(
        id="gate",
        label="Gate",
        op="gate",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "gate"},
    )
    end = SimpleNamespace(
        id="done",
        label="Done",
        op="done",
        fanout=False,
        metadata={"wf_terminal": True, "wf_op": "done"},
    )
    edge = SimpleNamespace(
        id="edge-go",
        label="go",
        source_ids=["gate"],
        target_ids=["done"],
        predicate=None,
        multiplicity="one",
        is_default=False,
        priority=1,
        safe_get_id=lambda: "edge-go",
    )
    graph = (start, {"gate": start, "done": end}, {"gate": [edge], "done": []})
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: graph,
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"gate": 0, "done": 0},
    )
    resolver = MappingStepResolver()
    resolver.set_state_schema({"items": "a"})
    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    client = httpx.Client(base_url="http://rust")
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        success = authority._client_result_effect(
            workflow_id="wf-1",
            suspended_node_id="gate",
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("a", {"items": "approved"})],
                _route_next=["go"],
            ),
            state={"items": []},
        )
        failure = authority._client_result_effect(
            workflow_id="wf-1",
            suspended_node_id="gate",
            client_result=RunFailure(
                conversation_node_id=None,
                state_update=[("u", {"failed": True})],
                errors=["denied"],
            ),
            state={"items": []},
        )
    finally:
        client.close()

    assert success["status"] == "success"
    assert success["successors"] == [{"node_id": "done", "join_mask": 0}]
    assert success["route_next"] == ["go"]
    assert success["result"]["final_state"]["items"] == ["approved"]
    assert failure["status"] == "failed"
    assert failure["errors"] == ["denied"]


def test_resume_effect_rejects_runtime_plumbing_state() -> None:
    authority = RustRuntimeAuthority.__new__(RustRuntimeAuthority)
    authority.runtime = SimpleNamespace()

    for key in ("_deps", "dream_deps", "_rt_node_ops", "_rt_join"):
        try:
            authority._client_result_effect(
                workflow_id="wf-1",
                suspended_node_id="gate",
                client_result=RunSuccess(
                    conversation_node_id=None,
                    state_update=[("u", {key: {"forged": True}})],
                ),
                state={},
            )
        except Exception as error:
            assert "runtime plumbing keys" in str(error)
        else:
            raise AssertionError(f"resume accepted reserved key {key!r}")


def test_resume_rejects_current_graph_drift_before_state_change(
    monkeypatch,
    tmp_path,
) -> None:
    start = SimpleNamespace(
        id="gate",
        label="Gate",
        op="gate",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "gate"},
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (start, {"gate": start}, {"gate": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"gate": 0},
    )
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path.endswith("/resume-contract"):
            return httpx.Response(
                200,
                json={
                    "run_id": "run-1",
                    "status": "suspended",
                    "suspended": [["gate", 0, "token-1", None]],
                    "runtime_routes": [
                        {
                            "source_node_id": "gate",
                            "target_node_id": "old-target",
                        }
                    ],
                    "node_ops": {"gate": "old-op"},
                },
                request=request,
            )
        raise AssertionError("graph drift must fail before another request")

    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://rust",
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        try:
            authority.resume(
                run_id="run-1",
                suspended_node_id="gate",
                suspended_token_id="token-1",
                client_result=RunSuccess(
                    conversation_node_id=None,
                    state_update=[],
                ),
                workflow_id="wf-1",
                conversation_id="conv-1",
                turn_node_id="turn-1",
            )
        except Exception as error:
            assert "frozen resume contract" in str(error)
        else:
            raise AssertionError("resume accepted a different current workflow")
    finally:
        authority.close()
        client.close()

    assert requests == [("GET", "/api/workflow/runs/run-1/resume-contract")]
