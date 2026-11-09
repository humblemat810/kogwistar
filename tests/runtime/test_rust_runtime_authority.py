from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended
from kogwistar.runtime.resolvers import AsyncMappingStepResolver, MappingStepResolver
from kogwistar.runtime.runtime import RunResult, WorkflowRuntime
from kogwistar.runtime.rust_worker import (
    AsyncRustRuntimeWorker,
    RustRuntimeWorker,
    RustWorkerError,
)
from kogwistar.runtime.rust_runtime_authority import (
    AsyncRustRuntimeAuthority,
    RustRuntimeAuthority,
    RustRuntimeAuthorityError,
    _native_operation_for_target,
    _server_managed_inline_worker_enabled,
    _server_advertises_inline_native_worker,
    _python_worker_required_for_plan,
    _runtime_plan_requires_python_worker,
    _validate_resume_plan,
    _json_state,
    freeze_runtime_plan,
)
from kogwistar.implementation_selection import capture_implementation_selection

ROOT = Path(__file__).resolve().parents[2]
RUST_SERVER = (
    ROOT
    / "rust"
    / "target"
    / "debug"
    / ("kogwistar-server.exe" if os.name == "nt" else "kogwistar-server")
)


class _LoseAsyncResultAcknowledgementAfterCommit(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.event_seq: int | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if request.url.path == "/internal/runtime/results" and self.event_seq is None:
            body = await response.aread()
            result = json.loads(body)
            event_seq = result.get("event_seq") if isinstance(result, dict) else None
            if not isinstance(event_seq, int) or event_seq <= 0:
                raise AssertionError("committed async result lacks a positive event_seq")
            self.event_seq = event_seq
            await response.aclose()
            raise httpx.ReadError(
                "simulated async result acknowledgement loss after commit",
                request=request,
            )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def test_native_operation_inventory_routes_through_shared_python_contract() -> None:
    fixture = json.loads(
        (
            ROOT
            / "contracts"
            / "golden"
            / "runtime-native-operation-inventory-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schema_version"] == 1
    assert fixture["registry_scope"] == "builtins"
    operations = fixture["operations"]
    assert operations
    for entry in operations:
        operation = entry["operation"]
        assert entry["version"]
        assert _native_operation_for_target(operation, "rust-native") == operation


def test_rust_authority_excludes_noncheckpointed_runtime_bucket() -> None:
    state = _json_state(
        {
            "durable": {"answer": "ok"},
            "_rt": {"view": object()},
            "_rt_join": {"pending": ["token-1"]},
        }
    )

    assert state == {
        "durable": {"answer": "ok"},
        "_rt_join": {"pending": ["token-1"]},
    }


def _assert_worker_dependencies(
    state: dict[str, Any], expected: dict[str, object]
) -> None:
    actual = state.get("_deps")
    assert isinstance(actual, dict)
    assert actual.get("_rust_worker_effects") is True
    assert all(actual.get(key) is value for key, value in expected.items())


def _test_hs256_token(secret: str, claims: dict[str, object]) -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signing_input = f'{encode({"alg": "HS256", "typ": "JWT"})}.{encode(claims)}'
    signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def _mcp_response_payload(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        payload = response.json()
    else:
        assert content_type.startswith("text/event-stream"), content_type
        data_lines = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert len(data_lines) == 1, response.text
        payload = json.loads(data_lines[0])
    assert isinstance(payload, dict)
    return payload


def _start_fake_chroma_collections() -> tuple[
    ThreadingHTTPServer, str, list[str], list[dict[str, Any]]
]:
    requests: list[str] = []
    payloads: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else {}
            requests.append(self.path)
            payloads.append(payload)
            name = payload.get("name", "unexpected")
            response = (
                {"id": f"collection-{name}", "metadata": {"hnsw:space": "cosine"}}
                if "/collections" in self.path and "/upsert" not in self.path
                else {}
            )
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}", requests, payloads


def _start_fake_ollama_embeddings() -> tuple[
    ThreadingHTTPServer, str, list[dict[str, Any]]
]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else {}
            requests.append(payload)
            inputs = payload.get("input", [])
            body = json.dumps(
                {"embeddings": [[0.6, 0.8, 0.0] for _ in inputs]}
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}", requests


@pytest.mark.parametrize(
    ("plan", "requires_python"),
    [
        ({"node_ops": {"a": "rust-native/increment-counter"}}, False),
        (
            {
                "node_ops": {"a": "increment-counter"},
                "node_executor_targets": {"a": "rust-native"},
            },
            False,
        ),
        (
            {
                "node_ops": {"a": "increment-counter", "b": "python-op"},
                "node_executor_targets": {"a": "rust-native"},
            },
            True,
        ),
        ({"node_ops": {}}, True),
        ({"node_ops": {"a": "increment-counter"}}, True),
    ],
)
def test_native_server_worker_bypass_is_limited_to_all_native_plans(
    plan: dict[str, Any],
    requires_python: bool,
) -> None:
    assert _runtime_plan_requires_python_worker(plan) is requires_python


@pytest.mark.parametrize(
    ("plan", "client_opt_in", "server_capability", "poll_required"),
    [
        ({"node_ops": {"a": "rust-native/increment-counter"}}, True, True, False),
        ({"node_ops": {"a": "rust-native/increment-counter"}}, True, False, True),
        ({"node_ops": {"a": "rust-native/increment-counter"}}, False, True, True),
        ({"node_ops": {"a": "python-op"}}, True, True, True),
    ],
)
def test_python_worker_polling_requires_plan_and_server_capability_agreement(
    plan: dict[str, Any],
    client_opt_in: bool,
    server_capability: bool,
    poll_required: bool,
) -> None:
    assert _python_worker_required_for_plan(
        plan,
        inline_mode_requested=client_opt_in,
        server_advertised_inline_worker=server_capability,
    ) is poll_required


@pytest.mark.parametrize(
    ("health", "expected"),
    [
        ({"native_inline_worker_enabled": True}, True),
        ({"native_inline_worker_enabled": False}, False),
        ({}, False),
        ({"native_inline_worker_enabled": "true"}, False),
    ],
)
def test_server_inline_worker_health_capability_requires_boolean_true(
    health: dict[str, Any],
    expected: bool,
) -> None:
    assert _server_advertises_inline_native_worker(health) is expected


def test_resume_validation_returns_the_single_validated_plan(monkeypatch) -> None:
    plan = {
        "runtime_routes": [],
        "node_ops": {"node": "op"},
        "node_versions": {"node": "v1"},
        "node_environment_ids": {},
    }
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.freeze_runtime_plan",
        lambda *_args: plan,
    )
    contract = {
        "runtime_routes": [],
        "node_ops": {"node": "op"},
        "node_versions": {"node": "v1"},
        "node_environment_ids": {},
        "node_executor_targets": {},
    }

    result = _validate_resume_plan(SimpleNamespace(), "wf", contract)

    assert result is plan


def test_server_managed_worker_client_mode_is_opt_in_and_matches_server_flag(
    monkeypatch,
) -> None:
    monkeypatch.delenv("KOGWISTAR_RUNTIME_NATIVE_INLINE", raising=False)
    assert _server_managed_inline_worker_enabled(None) is False
    monkeypatch.setenv("KOGWISTAR_RUNTIME_NATIVE_INLINE", "yes")
    assert _server_managed_inline_worker_enabled(None) is True
    assert _server_managed_inline_worker_enabled(False) is False
    monkeypatch.setenv("KOGWISTAR_RUNTIME_NATIVE_INLINE", "sometimes")
    with pytest.raises(RustRuntimeAuthorityError, match="must be a boolean"):
        _server_managed_inline_worker_enabled(None)


@pytest.mark.asyncio
async def test_async_authority_can_wait_without_polling_python_worker(
    tmp_path: Path,
) -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/api/workflow/runs/run-native":
            return httpx.Response(
                200,
                json={"run_id": "run-native", "status": "succeeded", "error": None},
                request=request,
            )
        if request.url.path.endswith("/checkpoints"):
            return httpx.Response(
                200,
                json={"checkpoints": []},
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(
        base_url="http://rust",
        transport=httpx.MockTransport(handler),
    )
    runtime = SimpleNamespace(
        step_resolver=AsyncMappingStepResolver(),
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    authority = AsyncRustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
        server_managed_native_worker=True,
    )

    async def unexpected_poll(**_kwargs: Any) -> int:
        raise AssertionError("server-managed native execution must not poll Python")

    authority.worker.poll_once = unexpected_poll
    try:
        result = await authority._pump_existing_run(
            run_id="run-native",
            durable_initial={"seed": 1},
            dependencies={},
            python_worker_required=False,
        )
    finally:
        await authority.aclose()
        await client.aclose()

    assert result.status == "succeeded"
    assert result.final_state == {"seed": 1}
    assert requests == [
        "/api/workflow/runs/run-native",
        "/api/workflow/runs/run-native/checkpoints",
    ]


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
    runtime: Any = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = SimpleNamespace(_state_schema={})
    runtime.workflow_engine = SimpleNamespace(metadata=None)
    runtime.conversation_engine = SimpleNamespace(
        metadata=None,
        read=SimpleNamespace(get_nodes=lambda **_kwargs: []),
    )

    def unexpected_python_admission(**_kwargs: Any) -> None:
        pytest.fail("Rust-selected runs must not use Python run admission")

    runtime._terminal_run_result = unexpected_python_admission
    runtime._claim_workflow_run = unexpected_python_admission

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
            "initial_state": {
                "seed": 1,
                "_wf_invocation_path": ["wf-1"],
                "_wf_current_run_id": "run-1",
            },
            "run_id": "run-1",
            "cache_dir": "cache",
        }
    ]


def test_sync_rust_authority_passes_parent_selector_snapshot_to_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    snapshot = capture_implementation_selection()
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    observed: dict[str, Any] = {}

    def build_worker(_cls: object, **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        RustRuntimeWorker, "from_step_resolver", classmethod(build_worker)
    )
    runtime = SimpleNamespace(
        step_resolver=lambda _op: None,
        predicate_registry={},
        _implementation_selection_snapshot=snapshot,
    )
    client = httpx.Client(
        base_url="http://rust",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={}, request=request)
        ),
    )
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        assert observed["implementation_selection_snapshot"] == snapshot
    finally:
        authority.close()
        client.close()


@pytest.mark.asyncio
async def test_async_rust_authority_passes_parent_selector_snapshot_to_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    snapshot = capture_implementation_selection()
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    observed: dict[str, Any] = {}

    async def close_worker() -> None:
        return None

    def build_worker(_cls: object, **kwargs: Any) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(aclose=close_worker)

    monkeypatch.setattr(
        AsyncRustRuntimeWorker, "from_step_resolver", classmethod(build_worker)
    )
    runtime = SimpleNamespace(
        step_resolver=lambda _op: None,
        predicate_registry={},
        _implementation_selection_snapshot=snapshot,
    )
    client = httpx.AsyncClient(
        base_url="http://rust",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={}, request=request)
        ),
    )
    authority = AsyncRustRuntimeAuthority(
        runtime=runtime,
        base_url="http://rust",
        cache_dir=tmp_path,
        client=client,
    )
    try:
        assert observed["implementation_selection_snapshot"] == snapshot
    finally:
        await authority.aclose()
        await client.aclose()


def test_rust_authority_scopes_live_python_dependencies_by_run() -> None:
    authority = object.__new__(RustRuntimeAuthority)
    authority._init_dependency_registry()
    first_service = object()
    second_service = object()

    authority._register_dependencies("run-a", {"service": first_service})
    authority._register_dependencies("run-b", {"service": second_service})

    assert authority._dependencies({"run_id": "run-a"}) == {"service": first_service}
    assert authority._dependencies({"run_id": "run-b"}) == {"service": second_service}
    with pytest.raises(RustRuntimeAuthorityError, match="already registered"):
        authority._register_dependencies("run-a", {})
    with pytest.raises(RustRuntimeAuthorityError, match="no live Python dependencies"):
        authority._dependencies({"run_id": "unknown"})

    authority._release_dependencies("run-a")
    with pytest.raises(RustRuntimeAuthorityError, match="no live Python dependencies"):
        authority._dependencies({"run_id": "run-a"})

    authority._register_dependencies("run-suspended", {"service": first_service})
    suspended = RunResult(
        run_id="run-suspended",
        final_state={},
        mq=__import__("queue").Queue(),
        status="suspended",
    )
    assert authority._release_if_terminal("run-suspended", suspended) is suspended
    assert (
        authority._dependencies({"run_id": "run-suspended"})["service"] is first_service
    )

    terminal = RunResult(
        run_id="run-suspended",
        final_state={},
        mq=__import__("queue").Queue(),
        status="succeeded",
    )
    authority._release_if_terminal("run-suspended", terminal)
    with pytest.raises(RustRuntimeAuthorityError, match="no live Python dependencies"):
        authority._dependencies({"run_id": "run-suspended"})


def test_run_dependencies_survive_authority_recreation_on_same_runtime() -> None:
    runtime = SimpleNamespace()
    first = object.__new__(RustRuntimeAuthority)
    first._init_dependency_registry(runtime)
    services = {"service": object()}
    first._register_dependencies("run-suspended", services)

    resumed_facade = object.__new__(RustRuntimeAuthority)
    resumed_facade._init_dependency_registry(runtime)
    assert resumed_facade._dependencies({"run_id": "run-suspended"}) is services

    isolated_facade = object.__new__(RustRuntimeAuthority)
    isolated_facade._init_dependency_registry(SimpleNamespace())
    with pytest.raises(RustRuntimeAuthorityError, match="no live Python dependencies"):
        isolated_facade._dependencies({"run_id": "run-suspended"})

    resumed_facade._release_dependencies("run-suspended")
    with pytest.raises(RustRuntimeAuthorityError, match="no live Python dependencies"):
        first._dependencies({"run_id": "run-suspended"})


def test_concurrent_authority_creation_shares_one_runtime_dependency_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import kogwistar.runtime.rust_runtime_authority as rust_authority

    original_init = rust_authority._RunDependencyStore.__init__
    init_lock = threading.Lock()
    first_init_entered = threading.Event()
    second_init_entered = threading.Event()
    release_first_init = threading.Event()
    init_count = 0

    def delayed_init(store: Any) -> None:
        nonlocal init_count
        with init_lock:
            init_count += 1
            current_init = init_count
        if current_init == 1:
            first_init_entered.set()
            assert release_first_init.wait(timeout=5)
        else:
            second_init_entered.set()
        original_init(store)

    monkeypatch.setattr(rust_authority._RunDependencyStore, "__init__", delayed_init)
    runtime = SimpleNamespace()

    def create_facade() -> RustRuntimeAuthority:
        facade = object.__new__(RustRuntimeAuthority)
        facade._init_dependency_registry(runtime)
        return facade

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(create_facade)
            assert first_init_entered.wait(timeout=5)
            second_started = threading.Event()

            def create_second_facade() -> RustRuntimeAuthority:
                second_started.set()
                return create_facade()

            second = pool.submit(create_second_facade)
            assert second_started.wait(timeout=5)
            assert not second_init_entered.wait(timeout=0.1)
            release_first_init.set()
            first_facade = first.result(timeout=5)
            second_facade = second.result(timeout=5)
    finally:
        release_first_init.set()

    assert init_count == 1
    assert first_facade._dependencies_by_run is second_facade._dependencies_by_run
    dependencies = {"opaque": object()}
    first_facade._register_dependencies("run-1", dependencies)
    assert second_facade._run_dependencies("run-1") is dependencies


def test_public_sync_runtime_rust_mode_without_authority_url_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.delenv("KOGWISTAR_RUST_RUNTIME_URL", raising=False)
    runtime: Any = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = SimpleNamespace(_state_schema={})
    runtime.workflow_engine = SimpleNamespace(metadata=None)
    runtime.conversation_engine = SimpleNamespace(
        metadata=None,
        read=SimpleNamespace(get_nodes=lambda **_kwargs: []),
    )

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
        _assert_worker_dependencies(ctx.state_view, deps)
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
                        {
                            "step_seq": 0,
                            "event_seq": 2,
                            "state": {"seed": 1, "answer": 42},
                        }
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
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
def test_sync_runtime_client_resume_uses_real_rust_authority(
    monkeypatch,
    tmp_path,
    backend: str,
) -> None:
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live authority test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for Rust/PostgreSQL test")
    pg_schema = f"kw_sync_resume_{os.urandom(8).hex()}" if backend == "pg" else None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    def node(node_id: str, op: str, **metadata: Any) -> SimpleNamespace:
        return SimpleNamespace(
            id=node_id,
            label=node_id,
            op=op,
            fanout=False,
            metadata={"wf_version": "v1", **metadata},
            safe_get_id=lambda: node_id,
        )

    start = node("start", "start", wf_start=True)
    gate = node("gate", "gate")
    end = node("end", "end", wf_terminal=True)

    def edge(edge_id: str, target_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=edge_id,
            label=edge_id,
            target_ids=[target_id],
            predicate=None,
            multiplicity="one",
            is_default=True,
            priority=100,
            safe_get_id=lambda: edge_id,
        )

    nodes = {item.id: item for item in (start, gate, end)}
    adjacency = {
        "start": [edge("start-gate", "gate")],
        "gate": [edge("gate-end", "end")],
        "end": [],
    }
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (start, nodes, adjacency),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {node_id: 0 for node_id in nodes},
    )
    resolver = MappingStepResolver()
    visited: list[str] = []
    dependencies = {"opaque_service": object()}

    @resolver.register("start")
    def start_step(ctx):
        visited.append("start")
        _assert_worker_dependencies(ctx.state_view, dependencies)
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("gate")
    def gate_step(_ctx):
        visited.append("gate")
        return RunSuspended(
            conversation_node_id=None,
            resume_payload={"prompt": "approve?"},
        )

    @resolver.register("end")
    def end_step(ctx):
        visited.append("end")
        assert ctx.state_view["approved"] is True
        _assert_worker_dependencies(ctx.state_view, dependencies)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"resumed_end": True})],
        )

    runtime: Any = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.workflow_engine = object()
    runtime.conversation_engine = SimpleNamespace(
        read=SimpleNamespace(get_nodes=lambda **_kwargs: [])
    )
    runtime.step_resolver = resolver
    runtime.predicate_registry = {}
    runtime.max_workers = 1
    runtime.cancel_requested = None
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "pg" if backend == "pg" else "sqlite",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "",
        }
    )
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        env["KOGWISTAR_PG_DSN"] = pg_dsn
        env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    else:
        env["KOGWISTAR_META_SQLITE_PATH"] = str(tmp_path / "authority.sqlite3")
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
            raise AssertionError(f"Rust {backend} server did not become ready")
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", base_url)
        monkeypatch.setenv(
            "KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR",
            str(tmp_path / "journal"),
        )
        result = runtime.run(
            workflow_id="wf-sync-resume-live",
            conversation_id="conv-sync-resume-live",
            turn_node_id="turn-sync-resume-live",
            initial_state={"seed": 1, "_deps": dependencies},
            run_id="run-sync-resume-live",
            cache_dir=tmp_path,
        )
        assert result.status == "suspended", result.errors
        checkpoint = httpx.get(
            f"{base_url}/api/workflow/runs/run-sync-resume-live/checkpoints",
            timeout=10,
        )
        checkpoint.raise_for_status()
        latest_state = checkpoint.json()["checkpoints"][-1]["state"]
        assert "_deps" not in latest_state
        token_id = latest_state["_rt_join"]["suspended"][0][2]
        if backend in {"sqlite", "pg"}:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            process = subprocess.Popen(
                [str(RUST_SERVER)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(100):
                if process.poll() is not None:
                    stderr = process.stderr.read() if process.stderr is not None else ""
                    raise AssertionError(
                        f"Rust server failed to restart for resume: {stderr}"
                    )
                try:
                    if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise AssertionError("Rust SQLite server did not recover for resume")
        resumed = runtime.resume_run(
            run_id="run-sync-resume-live",
            suspended_node_id="gate",
            suspended_token_id=token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"approved": True})],
            ),
            workflow_id="wf-sync-resume-live",
            conversation_id="conv-sync-resume-live",
            turn_node_id="turn-sync-resume-live",
            cache_dir=tmp_path,
        )
        assert resumed.status == "succeeded"
        assert resumed.final_state["approved"] is True
        assert resumed.final_state["resumed_end"] is True
        assert resumed.final_state["_deps"] is dependencies
        assert visited == ["start", "gate", "end"]
        status = httpx.get(
            f"{base_url}/api/workflow/runs/run-sync-resume-live", timeout=10
        )
        assert status.status_code == 200
        assert status.json()["status"] == "succeeded", status.text
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
def test_python_facade_leaves_all_native_work_to_inline_rust_server(
    monkeypatch,
    tmp_path: Path,
    backend: str,
) -> None:
    """The facade must not poll the Python worker for a fully native plan."""
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live native-server test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for live PostgreSQL test")
    monkeypatch.setenv("KOGWISTAR_RUNTIME_NATIVE_INLINE", "true")
    suffix = os.urandom(8).hex()
    run_id = f"native-facade-{suffix}"
    pg_schema = f"kw_native_facade_{suffix}" if backend == "pg" else None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    def node(node_id: str, *, terminal: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            id=node_id,
            label=node_id,
            op="rust-native/increment-counter",
            fanout=False,
            metadata={
                "wf_version": "v1",
                "wf_executor_target": "rust-native",
                **({"wf_terminal": True} if terminal else {}),
            },
            safe_get_id=lambda: node_id,
        )

    first = node("first")
    second = node("second", terminal=True)
    nodes = {item.id: item for item in (first, second)}
    route = SimpleNamespace(
        id="first-second",
        label="first-second",
        target_ids=["second"],
        predicate=None,
        multiplicity="one",
        is_default=True,
        priority=100,
        safe_get_id=lambda: "first-second",
    )
    adjacency = {"first": [route], "second": []}
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (first, nodes, adjacency),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {node_id: 0 for node_id in nodes},
    )

    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    server_env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        server_env.pop(name, None)
    server_env.update(
        {
            "KOGWISTAR_BACKEND": "pg" if backend == "pg" else "sqlite",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
            "KOGWISTAR_RUNTIME_NATIVE_INLINE": "true",
        }
    )
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        server_env["KOGWISTAR_PG_DSN"] = pg_dsn
        server_env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    else:
        server_env["KOGWISTAR_META_SQLITE_PATH"] = str(
            tmp_path / "facade-native.sqlite3"
        )
    process = subprocess.Popen(
        [str(RUST_SERVER)],
        cwd=ROOT,
        env=server_env,
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
            raise AssertionError("inline Rust server did not become ready")

        requests: list[tuple[str, str]] = []
        client = httpx.Client(
            base_url=base_url,
            headers={"x-kogwistar-roles": "reader"},
            event_hooks={
                "request": [lambda request: requests.append((request.method, request.url.path))]
            },
            timeout=10,
        )
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["native_inline_worker_enabled"] is True
        authority = RustRuntimeAuthority(
            runtime=runtime,
            base_url=base_url,
            cache_dir=tmp_path,
            client=client,
        )
        try:
            result = authority.run(
                workflow_id="rust-native-facade-workflow",
                conversation_id="rust-native-facade-conversation",
                turn_node_id=None,
                initial_state={"counter": 0},
                run_id=run_id,
            )
        finally:
            authority.close()
            client.close()

        assert result.status == "succeeded"
        assert result.final_state["counter"] == 2
        assert ("POST", "/api/workflow/runs") in requests
        assert any(path == f"/api/workflow/runs/{run_id}" for _, path in requests)
        assert not any(path == "/internal/runtime/claim" for _, path in requests)
        assert not any(path == "/internal/runtime/native/execute" for _, path in requests)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
def test_async_python_facade_uses_server_inline_native_worker(
    monkeypatch,
    tmp_path: Path,
    backend: str,
) -> None:
    """Async authority also skips Python claims only after server attestation."""
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live native-server test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for live PostgreSQL test")
    monkeypatch.setenv("KOGWISTAR_RUNTIME_NATIVE_INLINE", "true")
    suffix = os.urandom(8).hex()
    run_id = f"async-native-facade-{suffix}"
    pg_schema = f"kw_async_native_facade_{suffix}" if backend == "pg" else None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    def node(node_id: str, *, start: bool = False, terminal: bool = False):
        return SimpleNamespace(
            id=node_id,
            label=node_id,
            op="rust-native/increment-counter",
            fanout=False,
            metadata={
                "wf_version": "v1",
                "wf_executor_target": "rust-native",
                **({"wf_start": True} if start else {}),
                **({"wf_terminal": True} if terminal else {}),
            },
            safe_get_id=lambda: node_id,
        )

    first = node("first", start=True)
    second = node("second", terminal=True)
    nodes = {item.id: item for item in (first, second)}
    route = SimpleNamespace(
        id="first-second",
        label="first-second",
        target_ids=["second"],
        predicate=None,
        multiplicity="one",
        is_default=True,
        priority=100,
        safe_get_id=lambda: "first-second",
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (first, nodes, {"first": [route], "second": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {node_id: 0 for node_id in nodes},
    )
    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=AsyncMappingStepResolver(),
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    server_env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        server_env.pop(name, None)
    server_env.update(
        {
            "KOGWISTAR_BACKEND": "pg" if backend == "pg" else "sqlite",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "",
            "KOGWISTAR_RUNTIME_NATIVE_INLINE": "true",
        }
    )
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        server_env["KOGWISTAR_PG_DSN"] = pg_dsn
        server_env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    else:
        server_env["KOGWISTAR_META_SQLITE_PATH"] = str(
            tmp_path / "async-facade-native.sqlite3"
        )
    process = subprocess.Popen(
        [str(RUST_SERVER)],
        cwd=ROOT,
        env=server_env,
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
            raise AssertionError(f"async inline Rust {backend} server did not become ready")

        requests: list[tuple[str, str]] = []

        async def record_request(request: httpx.Request) -> None:
            requests.append((request.method, request.url.path))

        async def invoke() -> RunResult:
            client = httpx.AsyncClient(
                base_url=base_url,
                event_hooks={"request": [record_request]},
                timeout=10,
            )
            authority = AsyncRustRuntimeAuthority(
                runtime=runtime,
                base_url=base_url,
                cache_dir=tmp_path,
                client=client,
            )
            try:
                health = await client.get("/health")
                assert health.status_code == 200, health.text
                assert health.json()["native_inline_worker_enabled"] is True
                return await authority.run(
                    workflow_id="async-rust-native-facade-workflow",
                    conversation_id="async-rust-native-facade-conversation",
                    turn_node_id=None,
                    initial_state={"counter": 0},
                    run_id=run_id,
                )
            finally:
                await authority.aclose()
                await client.aclose()

        result = asyncio.run(invoke())
        assert result.status == "succeeded"
        assert result.final_state["counter"] == 2
        assert ("POST", "/api/workflow/runs") in requests
        assert not any(path == "/internal/runtime/claim" for _, path in requests)
        assert not any(path == "/internal/runtime/native/execute" for _, path in requests)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "sqlite-chroma", "pg"])
def test_python_worker_and_rust_native_step_interoperate_over_real_server(
    tmp_path: Path, backend: str
) -> None:
    """Exercise mixed execution and graph-effect replay on Rust storage paths."""
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live authority test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for live PostgreSQL test")
    pg_schema = (
        f"kw_worker_replay_{os.urandom(8).hex()}" if backend == "pg" else None
    )
    graph_enabled = backend in {"pg", "sqlite-chroma"}
    graph_namespace = (
        f"worker-graph-{os.urandom(8).hex()}" if graph_enabled else None
    )
    graph_entity_id = (
        f"worker-effect-{os.urandom(8).hex()}" if graph_enabled else None
    )
    graph_sqlalchemy_engine = None
    chroma_server = None
    chroma_requests: list[str] = []
    chroma_endpoint = None
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        import sqlalchemy as sa

        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        graph_sqlalchemy_engine = sa.create_engine(pg_dsn, future=True)
        PgVectorBackend(
            engine=graph_sqlalchemy_engine,
            embedding_dim=3,
            schema=f"{pg_schema}_knowledge",
        ).ensure_schema()
    elif backend == "sqlite-chroma":
        chroma_server, chroma_endpoint, chroma_requests, _ = (
            _start_fake_chroma_collections()
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "sqlite" if backend == "sqlite-chroma" else backend,
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
        }
    )
    if backend != "pg":
        env["KOGWISTAR_META_SQLITE_PATH"] = str(tmp_path / "mixed-language.sqlite3")
    worker_token = None
    if graph_enabled:
        jwt_secret = f"test-{os.urandom(24).hex()}"
        env["JWT_SECRET"] = jwt_secret
        env["JWT_ALG"] = "HS256"
        assert graph_namespace is not None
        if backend == "pg":
            assert pg_dsn is not None and pg_schema is not None
            env["KOGWISTAR_PG_DSN"] = pg_dsn
            env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
        else:
            assert chroma_endpoint is not None
            env["AUTH_MODE"] = "jwt"
            env["KOGWISTAR_CHROMA_ENDPOINT"] = chroma_endpoint
            env["KOGWISTAR_CHROMA_NODE_COLLECTION"] = "worker-nodes"
            env["KOGWISTAR_CHROMA_EDGE_COLLECTION"] = "worker-edges"
        worker_token = _test_hs256_token(
            jwt_secret,
            {
                "sub": "worker-replay-test",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": graph_namespace,
                "capabilities": ["write_graph"],
                **(
                    {"workspace": "worker-workspace", "graph_space": "knowledge"}
                    if backend == "pg"
                    else {}
                ),
                "exp": int(time.time()) + 3600,
            },
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

        resolver = MappingStepResolver()
        callbacks: list[str] = []

        @resolver.register("application.before")
        def before(ctx):
            callbacks.append("before")
            assert ctx.state_view["counter"] == 4
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"python_before_seen": True})],
            )

        @resolver.register("application.after")
        def after(ctx):
            callbacks.append("after")
            assert ctx.state_view["counter"] == 5
            assert ctx.state_view["python_before_seen"] is True
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"python_after_seen": True})],
            )

        headers = (
            {"authorization": f"Bearer {worker_token}"}
            if worker_token is not None
            else {"x-kogwistar-roles": "reader"}
        )
        submitted = httpx.post(
            f"{base_url}/api/workflow/runs",
            headers=headers,
            json={
                "run_id": "mixed-live-run",
                "workflow_id": "mixed-live-workflow",
                "conversation_id": "mixed-live-conversation",
                "initial_state": {
                    "counter": 4,
                    **(
                        {
                            "run_id": "mixed-live-run",
                            "conversation_id": "mixed-live-conversation",
                            "run_step_seq": 1,
                            "attempt_seq": 0,
                            "question": "snapshot smoke question",
                            "system_prompt": "snapshot smoke system",
                            "candidates": [],
                        }
                        if backend == "sqlite-chroma"
                        else {}
                    ),
                },
                "runtime_kind": "sync",
                "start_node_id": "python-before",
                "node_ops": {
                    "python-before": "application.before",
                    "native-middle": "increment-counter",
                    "python-after": "application.after",
                },
                "node_executor_targets": {"native-middle": "rust-native"},
                "runtime_routes": [
                    {
                        "source_node_id": "python-before",
                        "target_node_id": "native-middle",
                        "join_mask": 0,
                    },
                    {
                        "source_node_id": "native-middle",
                        "target_node_id": "python-after",
                        "join_mask": 0,
                    },
                ],
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text

        journal_path = tmp_path / "python-worker.sqlite"
        callback_marker = tmp_path / "python-worker-callbacks.txt"
        work_record = tmp_path / "python-worker-work.json"
        worker_probe = [
            sys.executable,
            "-m",
            "tests.runtime.worker_restart_probe",
            "--base-url",
            base_url,
            "--journal",
            str(journal_path),
            "--marker",
            str(callback_marker),
            "--work-record",
            str(work_record),
        ]
        worker_env = os.environ.copy()
        if graph_enabled:
            assert worker_token is not None
            assert graph_namespace is not None and graph_entity_id is not None
            worker_probe.extend(["--graph-namespace", graph_namespace])
            if backend == "sqlite-chroma":
                worker_probe.append("--snapshot-effect")
            else:
                worker_probe.extend(["--graph-entity-id", graph_entity_id])
                if backend == "pg":
                    worker_probe.extend(
                        ["--graph-default-acl-truth-graph", "knowledge"]
                    )
            worker_probe.extend(["--auth-token", worker_token])
            worker_env["JWT_SECRET"] = env["JWT_SECRET"]
            worker_env["JWT_ALG"] = "HS256"
        if backend == "pg":
            import sqlite3

            rejected_journal = tmp_path / "unresolved-postgres-worker.sqlite"
            checkpoints_before_rejection = httpx.get(
                f"{base_url}/api/workflow/runs/mixed-live-run/checkpoints",
                headers=headers,
                timeout=10,
            )
            assert checkpoints_before_rejection.status_code == 200
            with RustRuntimeWorker.from_step_resolver(
                base_url=base_url,
                worker_id="python-worker-unresolved",
                journal_path=rejected_journal,
                step_resolver=MappingStepResolver(),
                headers=headers,
            ) as rejecting_worker:
                rejected_work = rejecting_worker.claim(
                    run_id="mixed-live-run", lease_seconds=1
                )
                assert len(rejected_work) == 1
                assert rejected_work[0]["step_id"] == "python-before"
                with pytest.raises(
                    RustWorkerError,
                    match="cannot resolve frozen op 'application.before'",
                ):
                    rejecting_worker.process(rejected_work[0])

            with sqlite3.connect(rejected_journal) as connection:
                assert connection.execute(
                    "SELECT COUNT(*) FROM rust_worker_results"
                ).fetchone()[0] == 0
            checkpoints_after_rejection = httpx.get(
                f"{base_url}/api/workflow/runs/mixed-live-run/checkpoints",
                headers=headers,
                timeout=10,
            )
            assert checkpoints_after_rejection.status_code == 200
            assert (
                checkpoints_after_rejection.json()["checkpoints"]
                == checkpoints_before_rejection.json()["checkpoints"]
            )
            assert graph_sqlalchemy_engine is not None
            from kogwistar.engine_core.rust_postgres_session import (
                RustEnginePostgresMetaStore,
            )

            graph_store = RustEnginePostgresMetaStore(
                dsn=pg_dsn, schema=f"{pg_schema}_knowledge"
            )
            assert list(
                graph_store.iter_entity_events(namespace=graph_namespace)
            ) == []
            time.sleep(1.2)
        prepare = subprocess.run(
            [*worker_probe, "--worker-id", "python-worker-a", "--mode", "prepare"],
            cwd=ROOT,
            env=worker_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert prepare.returncode == 0, prepare.stderr
        # Worker A exits with a completed local journal but an unsubmitted
        # result. Restart the durable server/scheduler before worker B reclaims
        # the expired claim, proving recovery is not tied to scheduler memory.
        if backend == "sqlite":
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            process = subprocess.Popen(
                [str(RUST_SERVER)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(100):
                if process.poll() is not None:
                    stderr = process.stderr.read() if process.stderr is not None else ""
                    raise AssertionError(
                        f"Rust server failed to restart: {stderr}"
                    )
                try:
                    if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise AssertionError(f"Rust {backend} server did not recover after restart")
            resumed = httpx.get(
                f"{base_url}/api/workflow/runs/mixed-live-run",
                headers=headers,
                timeout=10,
            )
            assert resumed.status_code == 200, resumed.text
            assert resumed.json()["status"] != "succeeded"
        # The claim lease is one second; wait after server readiness so reclaim
        # exercises durable lease recovery, not an in-memory active claim.
        time.sleep(1.1)
        replay = subprocess.run(
            [
                *worker_probe,
                "--worker-id",
                "python-worker-b",
                "--mode",
                "replay",
                "--lease-seconds",
                "30",
            ],
            cwd=ROOT,
            env=worker_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert replay.returncode == 0, replay.stderr
        # Worker B exits after the server commits but its duplicate-result ACK
        # is lost. Worker C reopens the same journal and resubmits the saved
        # claim/result, proving process-restart recovery in the commit/ACK gap.
        recovered = subprocess.run(
            [*worker_probe, "--worker-id", "python-worker-b", "--mode", "recover"],
            cwd=ROOT,
            env=worker_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert recovered.returncode == 0, recovered.stderr
        assert callback_marker.read_text(encoding="utf-8").splitlines() == ["callback"]
        if graph_enabled:
            assert graph_namespace is not None and graph_entity_id is not None
            if backend == "pg":
                assert pg_dsn is not None and pg_schema is not None
                from kogwistar.engine_core.rust_postgres_session import (
                    RustEnginePostgresMetaStore,
                )

                graph_store = RustEnginePostgresMetaStore(
                    dsn=pg_dsn, schema=f"{pg_schema}_knowledge"
                )
                graph_events = list(
                    graph_store.iter_entity_events(namespace=graph_namespace)
                )
                graph_jobs = graph_store.list_index_jobs(namespace=graph_namespace)
            else:
                import sqlite3

                database_path = tmp_path / "mixed-language.sqlite3"
                with sqlite3.connect(
                    f"file:{database_path.as_posix()}?mode=ro", uri=True
                ) as connection:
                    graph_events = connection.execute(
                        """
                        SELECT namespace, entity_kind, entity_id, op, payload_json
                        FROM entity_events WHERE namespace = ?
                        """,
                        (graph_namespace,),
                    ).fetchall()
                    graph_jobs = connection.execute(
                        """
                        SELECT namespace, entity_id, index_kind, status
                        FROM index_jobs WHERE namespace = ?
                        """,
                        (graph_namespace,),
                    ).fetchall()
            if backend == "pg":
                assert len(graph_events) == 4
                graph_event = next(
                    event
                    for event in graph_events
                    if event[1] == "node" and event[2] == graph_entity_id
                )
                graph_payload = json.loads(graph_event[4])
                assert graph_payload["doc_id"] == f"source-doc-{graph_entity_id}"
                assert graph_payload["mentions"][0]["spans"][0] == {
                    "doc_id": f"source-doc-{graph_entity_id}",
                    "start_char": 0,
                    "end_char": len("worker effect"),
                }
                assert graph_payload["metadata"]["source_revision_id"] == "revision-1"
                acl_payloads = [
                    json.loads(event[4])
                    for event in graph_events
                    if json.loads(event[4])
                    .get("metadata", {})
                    .get("entity_type")
                    == "acl_record"
                ]
                assert len(acl_payloads) == 3
                assert {
                    payload["metadata"]["acl_target_grain"]
                    for payload in acl_payloads
                } == {"node", "grounding", "span"}
                assert all(
                    payload["metadata"]["owner_id"] == "worker-replay-test"
                    and payload["metadata"]["acl_mode"] == "private"
                    and payload["metadata"]["acl_target_entity_id"]
                    == graph_entity_id
                    for payload in acl_payloads
                )
            else:
                assert len(graph_events) == 1
            snapshot_effect = backend == "sqlite-chroma"
            if backend == "pg":
                assert graph_event[1:4] == ("node", graph_entity_id, "ADD")
            else:
                assert graph_events[0][0:4] == (
                    graph_namespace,
                    "node",
                    graph_events[0][2] if snapshot_effect else graph_entity_id,
                    "ADD",
                )
                if snapshot_effect:
                    from kogwistar.id_provider import stable_id

                    snapshot_id = str(
                        stable_id(
                            "conversation.context_snapshot",
                            "mixed-live-conversation",
                            "mixed-live-run",
                            "select_used_evidence",
                            "1",
                            "0",
                        )
                    )
                    assert graph_events[0][2] == snapshot_id
                    snapshot_payload = json.loads(graph_events[0][4])
                    assert snapshot_payload["metadata"]["entity_type"] == "context_snapshot"
            if backend == "pg":
                assert any(job.entity_id == graph_entity_id for job in graph_jobs)
            else:
                if snapshot_effect:
                    assert {job[2] for job in graph_jobs} >= {"node_docs", "node_refs"}
                    assert all(job[0] == graph_namespace for job in graph_jobs)
                    assert all(job[1] == graph_events[0][2] for job in graph_jobs)
                else:
                    assert {
                        (job[0], job[1], job[2], job[3]) for job in graph_jobs
                    } == {
                        (graph_namespace, graph_entity_id, "node_embedding", "PENDING"),
                        (graph_namespace, graph_entity_id, "node_docs", "PENDING"),
                    }
            if backend == "pg":
                assert graph_sqlalchemy_engine is not None
                from kogwistar.engine_core.postgres_backend import PgVectorBackend

                projected = PgVectorBackend(
                    engine=graph_sqlalchemy_engine,
                    embedding_dim=3,
                    schema=f"{pg_schema}_knowledge",
                ).node_get(
                    ids=[graph_entity_id], include=["documents", "metadatas"]
                )
                assert projected["ids"] == [graph_entity_id]
                assert projected["documents"] == ["worker effect"]
                assert projected["metadatas"][0]["workspace_id"] == "worker-workspace"
                assert projected["metadatas"][0]["graph_space"] == "knowledge"
            else:
                assert chroma_requests
                assert all("/collections" in path for path in chroma_requests)
        callbacks.append("before")

        with RustRuntimeWorker.from_step_resolver(
            base_url=base_url,
            worker_id="python-worker",
            journal_path=journal_path,
            step_resolver=resolver,
            headers=headers,
        ) as worker:
            native_claim = httpx.post(
                f"{base_url}/internal/runtime/claim",
                headers=headers,
                json={
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"],
                    "run_id": "mixed-live-run",
                },
                timeout=10,
            )
            assert native_claim.status_code == 200, native_claim.text
            native_work = native_claim.json()["work"]
            assert len(native_work) == 1
            native_item = native_work[0]
            assert native_item["step_id"] == "native-middle"
            assert native_item["payload"]["state"]["python_before_seen"] is True
            native_result = httpx.post(
                f"{base_url}/internal/runtime/native/execute",
                headers=headers,
                json={
                    "handoff": {
                        "message_id": native_item["message_id"],
                        "claimed_by": native_item["claimed_by"],
                        "run_id": native_item["run_id"],
                        "step_id": native_item["step_id"],
                        "correlation_id": native_item["correlation_id"],
                    },
                    "payload": native_item["payload"],
                },
                timeout=10,
            )
            assert native_result.status_code == 200, native_result.text

            python_after_work = worker.claim(run_id="mixed-live-run")
            assert len(python_after_work) == 1
            assert python_after_work[0]["step_id"] == "python-after"
            assert python_after_work[0]["payload"]["state"]["counter"] == 5
            assert (
                python_after_work[0]["payload"]["state"]["python_before_seen"] is True
            )
            final_result = worker.process(python_after_work[0])
            assert final_result["event_seq"] > 0

        assert callbacks == ["before", "after"]
        status = httpx.get(
            f"{base_url}/api/workflow/runs/mixed-live-run",
            headers=headers,
            timeout=10,
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded", status.text
        checkpoints = httpx.get(
            f"{base_url}/api/workflow/runs/mixed-live-run/checkpoints",
            headers=headers,
            timeout=10,
        )
        assert checkpoints.status_code == 200, checkpoints.text
        final_state = checkpoints.json()["checkpoints"][-1]["state"]
        assert final_state["counter"] == 5
        assert final_state["python_before_seen"] is True
        assert final_state["python_after_seen"] is True
        digest_state = {
            key: final_state[key]
            for key in ("counter", "python_after_seen", "python_before_seen")
        }
        final_state_digest = hashlib.sha256(
            json.dumps(digest_state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert final_state_digest == (
            "f39046db1b17d0df597ecbd2a6d9bed05b70563740a20dc37862eda40e789279"
        )

        if backend in {"sqlite", "sqlite-chroma", "pg"}:
            failure_run_id = "mixed-failure-run"
            failure_submitted = httpx.post(
                f"{base_url}/api/workflow/runs",
                headers=headers,
                json={
                    "run_id": failure_run_id,
                    "workflow_id": "mixed-failure-workflow",
                    "conversation_id": "mixed-failure-conversation",
                    "initial_state": {},
                    "runtime_kind": "sync",
                    "start_node_id": "python-failure",
                    "node_ops": {"python-failure": "application.before"},
                    "runtime_routes": [],
                },
                timeout=10,
            )
            assert failure_submitted.status_code == 202, failure_submitted.text
            failure_journal = tmp_path / "python-worker-failure.sqlite"
            failure_marker = tmp_path / "python-worker-failure-callbacks.txt"
            failure_record = tmp_path / "python-worker-failure-work.json"
            failure_probe = [
                sys.executable,
                "-m",
                "tests.runtime.worker_restart_probe",
                "--base-url",
                base_url,
                "--worker-id",
                "python-failure-worker",
                "--journal",
                str(failure_journal),
                "--marker",
                str(failure_marker),
                "--work-record",
                str(failure_record),
                "--run-id",
                failure_run_id,
                "--step-id",
                "python-failure",
                "--failure-result",
            ]
            if worker_token is not None:
                failure_probe.extend(["--auth-token", worker_token])
            failed = subprocess.run(
                [*failure_probe, "--mode", "failure"],
                cwd=ROOT,
                env=worker_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert failed.returncode == 0, failed.stderr
            replayed_failure = subprocess.run(
                [*failure_probe, "--mode", "recover"],
                cwd=ROOT,
                env=worker_env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            assert replayed_failure.returncode == 0, replayed_failure.stderr
            assert failure_marker.read_text(encoding="utf-8").splitlines() == [
                "callback"
            ]
            failure_status = httpx.get(
                f"{base_url}/api/workflow/runs/{failure_run_id}",
                headers=headers,
                timeout=10,
            )
            assert failure_status.status_code == 200, failure_status.text
            assert failure_status.json()["status"] == "failed"
            failure_events = httpx.get(
                f"{base_url}/api/workflow/runs/{failure_run_id}/events?after_seq=0",
                headers=headers,
                timeout=10,
            )
            assert failure_events.status_code == 200, failure_events.text
            terminal_failures = []
            for frame in failure_events.text.split("\n\n"):
                data_line = next(
                    (line[6:] for line in frame.splitlines() if line.startswith("data: ")),
                    None,
                )
                if data_line is None:
                    continue
                event_payload = json.loads(data_line)
                run_status = event_payload.get("reduced", {}).get("run_status", {})
                if run_status.get("terminal_node_id") == f"wf_failed|{failure_run_id}":
                    terminal_failures.append(event_payload)
            assert len(terminal_failures) == 1

            async_failure_run_id = "mixed-async-failure-run"
            async_failure_submitted = httpx.post(
                f"{base_url}/api/workflow/runs",
                headers=headers,
                json={
                    "run_id": async_failure_run_id,
                    "workflow_id": "mixed-async-failure-workflow",
                    "conversation_id": "mixed-async-failure-conversation",
                    "initial_state": {},
                    "runtime_kind": "async",
                    "start_node_id": "python-async-failure",
                    "node_ops": {
                        "python-async-failure": "application.async_failure"
                    },
                    "runtime_routes": [],
                },
                timeout=10,
            )
            assert async_failure_submitted.status_code == 202, async_failure_submitted.text
            async_resolver = AsyncMappingStepResolver()
            async_callbacks: list[str] = []

            @async_resolver.register("application.async_failure")
            async def fail_async(_ctx):
                async_callbacks.append("callback")
                await asyncio.sleep(0)
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[("u", {"async_failure_probe_seen": True})],
                    errors=["expected async worker failure"],
                )

            async_journal = tmp_path / "python-async-worker-failure.sqlite"
            async_transport = _LoseAsyncResultAcknowledgementAfterCommit()

            async def run_async_failure_and_replay() -> None:
                async_client = httpx.AsyncClient(
                    transport=async_transport,
                    base_url=base_url,
                    headers=headers,
                )
                worker = AsyncRustRuntimeWorker.from_step_resolver(
                    base_url=base_url,
                    worker_id="python-async-failure-worker",
                    journal_path=async_journal,
                    step_resolver=async_resolver,
                    client=async_client,
                )
                try:
                    claimed = await worker.claim(run_id=async_failure_run_id)
                    assert len(claimed) == 1
                    work_item = claimed[0]
                    assert work_item["step_id"] == "python-async-failure"
                    with pytest.raises(
                        RustWorkerError,
                        match="simulated async result acknowledgement loss after commit",
                    ):
                        await worker.process(work_item)
                finally:
                    await worker.aclose()
                    await async_client.aclose()

                assert async_transport.event_seq is not None
                restarted_worker = AsyncRustRuntimeWorker.from_step_resolver(
                    base_url=base_url,
                    worker_id="python-async-failure-worker",
                    journal_path=async_journal,
                    step_resolver=async_resolver,
                    headers=headers,
                )
                try:
                    replayed = await restarted_worker.process(work_item)
                finally:
                    await restarted_worker.aclose()
                assert replayed["event_seq"] == async_transport.event_seq

            asyncio.run(run_async_failure_and_replay())
            assert async_callbacks == ["callback"]
            async_failure_status = httpx.get(
                f"{base_url}/api/workflow/runs/{async_failure_run_id}",
                headers=headers,
                timeout=10,
            )
            assert async_failure_status.status_code == 200, async_failure_status.text
            assert async_failure_status.json()["status"] == "failed"
            async_failure_events = httpx.get(
                f"{base_url}/api/workflow/runs/{async_failure_run_id}/events?after_seq=0",
                headers=headers,
                timeout=10,
            )
            assert async_failure_events.status_code == 200, async_failure_events.text
            async_terminal_failures = []
            for frame in async_failure_events.text.split("\n\n"):
                data_line = next(
                    (line[6:] for line in frame.splitlines() if line.startswith("data: ")),
                    None,
                )
                if data_line is None:
                    continue
                event_payload = json.loads(data_line)
                run_status = event_payload.get("reduced", {}).get("run_status", {})
                if run_status.get("terminal_node_id") == f"wf_failed|{async_failure_run_id}":
                    async_terminal_failures.append(event_payload)
            assert len(async_terminal_failures) == 1

        async_callbacks: list[str] = []
        async_resolver = AsyncMappingStepResolver()

        @async_resolver.register("application.async_before")
        async def async_before(ctx):
            async_callbacks.append("before")
            assert ctx.state_view["counter"] == 8
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"async_python_before_seen": True})],
            )

        @async_resolver.register("application.async_after")
        async def async_after(ctx):
            async_callbacks.append("after")
            assert ctx.state_view["counter"] == 9
            assert ctx.state_view["async_python_before_seen"] is True
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"async_python_after_seen": True})],
            )

        async_submitted = httpx.post(
            f"{base_url}/api/workflow/runs",
            headers=headers,
            json={
                "run_id": "mixed-async-live-run",
                "workflow_id": "mixed-async-live-workflow",
                "conversation_id": "mixed-async-live-conversation",
                "initial_state": {"counter": 8},
                "runtime_kind": "async",
                "start_node_id": "python-before",
                "node_ops": {
                    "python-before": "application.async_before",
                    "native-middle": "increment-counter",
                    "python-after": "application.async_after",
                },
                "node_executor_targets": {"native-middle": "rust-native"},
                "runtime_routes": [
                    {
                        "source_node_id": "python-before",
                        "target_node_id": "native-middle",
                        "join_mask": 0,
                    },
                    {
                        "source_node_id": "native-middle",
                        "target_node_id": "python-after",
                        "join_mask": 0,
                    },
                ],
            },
            timeout=10,
        )
        assert async_submitted.status_code == 202, async_submitted.text

        async def execute_async_python_workers() -> None:
            worker = AsyncRustRuntimeWorker.from_step_resolver(
                base_url=base_url,
                worker_id="async-python-worker",
                journal_path=tmp_path / "async-python-worker.sqlite",
                step_resolver=async_resolver,
                headers=headers,
            )
            try:
                async_before_work = await worker.claim(run_id="mixed-async-live-run")
                assert len(async_before_work) == 1
                assert async_before_work[0]["step_id"] == "python-before"
                await worker.process(async_before_work[0])

                async with httpx.AsyncClient(
                    base_url=base_url, headers=headers, timeout=10
                ) as client:
                    native_claim_response = await client.post(
                        "/internal/runtime/claim",
                        json={
                            "claimed_by": "rust-native-worker",
                            "limit": 1,
                            "lease_seconds": 60,
                            "recipient_ids": ["rust-native-worker"],
                            "run_id": "mixed-async-live-run",
                        },
                    )
                    assert native_claim_response.status_code == 200
                    native_work = native_claim_response.json()["work"]
                    assert len(native_work) == 1
                    native_item = native_work[0]
                    assert native_item["step_id"] == "native-middle"
                    assert (
                        native_item["payload"]["state"]["async_python_before_seen"]
                        is True
                    )
                    native_response = await client.post(
                        "/internal/runtime/native/execute",
                        json={
                            "handoff": {
                                "message_id": native_item["message_id"],
                                "claimed_by": native_item["claimed_by"],
                                "run_id": native_item["run_id"],
                                "step_id": native_item["step_id"],
                                "correlation_id": native_item["correlation_id"],
                            },
                            "payload": native_item["payload"],
                        },
                    )
                    assert native_response.status_code == 200, native_response.text

                async_after_work = await worker.claim(run_id="mixed-async-live-run")
                assert len(async_after_work) == 1
                assert async_after_work[0]["step_id"] == "python-after"
                async_after_state = async_after_work[0]["payload"]["state"]
                assert async_after_state["counter"] == 9
                assert async_after_state["async_python_before_seen"] is True
                await worker.process(async_after_work[0])
            finally:
                await worker.aclose()

        asyncio.run(execute_async_python_workers())
        assert async_callbacks == ["before", "after"]
        async_status = httpx.get(
            f"{base_url}/api/workflow/runs/mixed-async-live-run",
            headers=headers,
            timeout=10,
        )
        assert async_status.status_code == 200, async_status.text
        assert async_status.json()["status"] == "succeeded"
        async_checkpoints = httpx.get(
            f"{base_url}/api/workflow/runs/mixed-async-live-run/checkpoints",
            headers=headers,
            timeout=10,
        )
        assert async_checkpoints.status_code == 200, async_checkpoints.text
        async_final_state = async_checkpoints.json()["checkpoints"][-1]["state"]
        assert async_final_state["counter"] == 9
        assert async_final_state["async_python_before_seen"] is True
        assert async_final_state["async_python_after_seen"] is True
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()
        if chroma_server is not None:
            chroma_server.shutdown()
            chroma_server.server_close()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite-chroma", "pg"])
@pytest.mark.parametrize("persistence_target", ["python-worker", "rust-native"])
def test_python_agentic_answer_workflow_commits_graph_effects_through_rust_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    persistence_target: str,
) -> None:
    """Compare Python-worker and Rust-native persistence of Python answers."""
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live authority test")

    from kogwistar.conversation.agentic_answering import AgenticAnsweringAgent
    from kogwistar.conversation.conversation_context import (
        ContextItem,
        ContextMessage,
        PromptContext,
    )
    from kogwistar.conversation.models import ConversationEdge, ConversationNode
    from kogwistar.conversation.resolvers import default_resolver
    from kogwistar.conversation.service import ConversationService
    import kogwistar.conversation.resolvers as resolver_module
    import sqlite3

    assistant_contract = json.loads(
        (ROOT / "contracts/golden/assistant-turn-graph-v1.json").read_text(
            encoding="utf-8"
        )
    )

    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for live PostgreSQL test")
    pg_schema = f"python_answer_{os.urandom(8).hex()}" if backend == "pg" else None
    database_path = tmp_path / "python-answer-rust-authority.sqlite3"
    auth_database_path = tmp_path / "python-answer-auth.sqlite3"
    graph_namespace = f"python-answer-{os.urandom(8).hex()}"
    chroma_server = None
    chroma_endpoint = None
    chroma_requests: list[str] = []
    chroma_payloads: list[dict[str, Any]] = []
    if backend == "sqlite-chroma":
        chroma_server, chroma_endpoint, chroma_requests, chroma_payloads = (
            _start_fake_chroma_collections()
        )
    else:
        import sqlalchemy as sa

        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        assert pg_dsn is not None and pg_schema is not None
        graph_engine = sa.create_engine(pg_dsn, future=True)
        try:
            PgVectorBackend(
                engine=graph_engine,
                embedding_dim=3,
                schema=pg_schema,
            ).ensure_schema()
        finally:
            graph_engine.dispose()
    embedding_server, embedding_endpoint, embedding_requests = (
        _start_fake_ollama_embeddings()
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    jwt_secret = f"test-{os.urandom(24).hex()}"
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "sqlite" if backend == "sqlite-chroma" else "pg",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
            "AUTH_MODE": "jwt",
            "JWT_SECRET": jwt_secret,
            "JWT_ALG": "HS256",
            "KOGWISTAR_GRAPH_ROLE": "conversation",
            "KOGWISTAR_EMBEDDING_WORKER_INLINE": "true",
            "KOGWISTAR_EMBEDDING_PROVIDER": "ollama",
            "KOGWISTAR_EMBEDDING_MODEL": "fixture-embedding-model",
            "KOGWISTAR_EMBEDDING_DIMENSION": "3",
            "KOGWISTAR_EMBEDDING_ENDPOINT": embedding_endpoint,
        }
    )
    if backend == "sqlite-chroma":
        env["KOGWISTAR_META_SQLITE_PATH"] = str(database_path)
        assert chroma_endpoint is not None
        env.update(
            {
                "KOGWISTAR_CHROMA_ENDPOINT": chroma_endpoint,
                "KOGWISTAR_CHROMA_NODE_COLLECTION": "python-answer-nodes",
                "KOGWISTAR_CHROMA_EDGE_COLLECTION": "python-answer-edges",
            }
        )
    else:
        assert pg_dsn is not None and pg_schema is not None
        env["KOGWISTAR_PG_DSN"] = pg_dsn
        env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    env["AUTH_DB_URL"] = f"sqlite:///{auth_database_path.as_posix()}"
    with sqlite3.connect(auth_database_path) as auth_db:
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
                user_id, email, display_name, is_active, global_role,
                global_ns, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (
                "python-answer-owner",
                "python-answer-owner@example.test",
                "Python answer owner",
                True,
                "rw",
                graph_namespace,
            ),
        )
    token = _test_hs256_token(
        jwt_secret,
        {
            "sub": "python-answer-owner",
            "role": "rw",
            "roles": ["reader", "rw"],
            "ns": ["workflow", "conversation", graph_namespace],
            "storage_ns": graph_namespace,
            "capabilities": ["read_graph", "write_graph"],
            "exp": int(time.time()) + 3600,
        },
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
            raise AssertionError("Rust SQLite+Chroma server did not become ready")

        class _ConversationGraph:
            namespace = graph_namespace
            persistence_mode = "single_stage"
            _phase1_enable_index_jobs = True
            _disable_event_log = False
            backend = SimpleNamespace(
                embedding_dim=3,
                edge_get=lambda **_kwargs: {"ids": [], "metadatas": []},
            )
            pre_add_node_hooks: list[Any] = []
            pre_add_edge_hooks: list[Any] = []

            @staticmethod
            def iterative_defensive_emb(_text: str) -> list[float]:
                return [0.6, 0.8, 0.0]

            @staticmethod
            def _node_projection(node: Any) -> tuple[str, dict[str, Any]]:
                return (
                    node.model_dump_json(field_mode="backend", exclude=["embedding"]),
                    {**node.metadata, "namespace": graph_namespace},
                )

            @staticmethod
            def _edge_projection(edge: Any) -> tuple[str, dict[str, Any]]:
                return (
                    edge.model_dump_json(field_mode="backend", exclude=["embedding"]),
                    {**edge.metadata, "namespace": graph_namespace},
                )

        graph = _ConversationGraph()
        graph.embed = SimpleNamespace(iterative_defensive_emb=graph.iterative_defensive_emb)
        graph.write = SimpleNamespace(
            node_doc_and_meta=graph._node_projection,
            edge_doc_and_meta=graph._edge_projection,
            add_node=lambda *_args, **_kwargs: pytest.fail(
                "Rust-owned workflow must not write graph nodes through Python"
            ),
            add_edge=lambda *_args, **_kwargs: pytest.fail(
                "Rust-owned workflow must not write graph edges through Python"
            ),
        )
        graph.indexing = SimpleNamespace(
            canonical_source_payload_for_projection=lambda **_kwargs: "{}"
        )
        service = ConversationService.__new__(ConversationService)
        service.conversation_engine = graph
        monkeypatch.setattr(resolver_module, "_chat_service", lambda _deps: service)

        class _AnswerAgent:
            cache_dir = str(tmp_path / "answer-cache")
            config = SimpleNamespace()
            conversation_engine = graph

            @staticmethod
            def _generate_answer_with_citations(agent: Any, **kwargs: Any) -> dict[str, Any]:
                assert agent.conversation_engine is graph
                assert kwargs["question"] == "How does the workflow cross the boundary?"
                assert kwargs["evidence_pack"] == {
                    "nodes": [{"node_id": "evidence-node"}],
                    "edges": [],
                }
                return {
                    "text": "The Rust runtime commits the Python worker's graph effects.",
                    "reasoning": "",
                    "claims": [],
                }

            def rehydrate_evidence_pack_from_digest(self, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "evidence_pack": {
                        "nodes": [{"node_id": "evidence-node"}],
                        "edges": [],
                    }
                }

            def _prepare_assistant_turn(self, **kwargs: Any) -> Any:
                return AgenticAnsweringAgent._prepare_assistant_turn(self, **kwargs)

            def _prepare_run_to_response_edge(self, **kwargs: Any) -> Any:
                return AgenticAnsweringAgent._prepare_run_to_response_edge(
                    self, **kwargs
                )

        agent = _AnswerAgent()

        resolver = MappingStepResolver()

        @resolver.register("python.answer.generate")
        def generate(ctx: Any) -> Any:
            with ctx.state_write as state:
                state["_rt"] = {
                    "view": PromptContext(
                        conversation_id="python-answer-conversation",
                        purpose="answer",
                        token_budget=64,
                        tokens_used=3,
                        items=(
                            ContextItem(
                                kind="tail_turn",
                                role="user",
                                text="How does the workflow cross the boundary?",
                                source="conversation",
                                node_id="user-turn",
                                token_cost=3,
                            ),
                        ),
                        messages=(
                            ContextMessage(
                                role="user",
                                content="How does the workflow cross the boundary?",
                                source="conversation",
                            ),
                        ),
                    ),
                    "evidence_pack": {
                        "nodes": [{"node_id": "evidence-node"}],
                        "edges": [],
                    },
                }
            return default_resolver.resolve("aa_generate_answer_with_citations")(ctx)

        @resolver.register("python.answer.persist")
        def persist(ctx: Any) -> Any:
            return default_resolver.resolve("aa_persist_response")(ctx)

        headers = {"authorization": f"Bearer {token}"}
        submitted = httpx.post(
            f"{base_url}/api/workflow/runs",
            headers=headers,
            json={
                "run_id": "python-answer-run",
                "workflow_id": "python-answer-workflow",
                "conversation_id": "python-answer-conversation",
                "initial_state": {
                    "run_id": "python-answer-run",
                    "run_node_id": "python-answer-run-node",
                    "conversation_id": "python-answer-conversation",
                    "turn_index": 0,
                    "question": "How does the workflow cross the boundary?",
                    "system_prompt": "Answer using the provided evidence.",
                    "used_node_ids": ["evidence-node"],
                    "evidence_pack_digest": {"evidence_pack_hash": "evidence-1"},
                    "run_step_seq": 1,
                    "attempt_seq": 0,
                    "prev_turn_meta_summary": {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                        "tail_turn_index": 0,
                    },
                },
                "runtime_kind": "sync",
                "start_node_id": "answer-generate",
                "node_ops": {
                    "answer-generate": "python.answer.generate",
                    "answer-persist": (
                        "python.answer.persist"
                        if persistence_target == "python-worker"
                        else "aa_persist_response"
                    ),
                },
                "node_executor_targets": (
                    {"answer-persist": "rust-native"}
                    if persistence_target == "rust-native"
                    else {}
                ),
                "runtime_routes": [
                    {
                        "source_node_id": "answer-generate",
                        "target_node_id": "answer-persist",
                        "join_mask": 0,
                    }
                ],
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text

        worker = RustRuntimeWorker.from_step_resolver(
            base_url=base_url,
            worker_id="python-answer-worker",
            journal_path=tmp_path / "python-answer-worker.sqlite",
            step_resolver=resolver,
            dependency_provider=lambda _work: {
                "agent": agent,
                "conversation_engine": graph,
                "conversation_service": service,
                "_chat_service": lambda _deps: service,
            },
            headers=headers,
        )
        with worker:
            for _ in range(3):
                claimed = worker.claim(run_id="python-answer-run", limit=1)
                if not claimed:
                    break
                result = worker.process(claimed[0])
                assert result["event_seq"] > 0
        if persistence_target == "rust-native":
            native_claim = httpx.post(
                f"{base_url}/internal/runtime/claim",
                headers=headers,
                json={
                    "claimed_by": "rust-native-worker",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["rust-native-worker"],
                    "run_id": "python-answer-run",
                },
                timeout=10,
            )
            assert native_claim.status_code == 200, native_claim.text
            native_work = native_claim.json()["work"]
            assert len(native_work) == 1
            native_item = native_work[0]
            assert native_item["payload"]["executor_target"] == "rust-native"
            assert native_item["payload"]["op"] == "aa_persist_response"
            native_result = httpx.post(
                f"{base_url}/internal/runtime/native/execute",
                headers=headers,
                json={
                    "handoff": {
                        key: native_item[key]
                        for key in (
                            "message_id",
                            "claimed_by",
                            "run_id",
                            "step_id",
                            "correlation_id",
                        )
                    },
                    "payload": native_item["payload"],
                },
                timeout=10,
            )
            assert native_result.status_code == 200, native_result.text
        status = httpx.get(
            f"{base_url}/api/workflow/runs/python-answer-run",
            headers=headers,
            timeout=10,
        )
        run_deadline = time.monotonic() + 20
        while status.json()["status"] == "running" and time.monotonic() < run_deadline:
            time.sleep(0.05)
            status = httpx.get(
                f"{base_url}/api/workflow/runs/python-answer-run",
                headers=headers,
                timeout=10,
            )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded", status.text
        run_events_response = httpx.post(
            f"{base_url}/mcp/workflow",
            headers={
                **headers,
                "accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": "python-answer-run-events",
                "method": "tools/call",
                "params": {
                    "name": "workflow.run_events",
                    "arguments": {
                        "run_id": "python-answer-run",
                        "after_seq": 0,
                        "limit": 100,
                    },
                },
            },
            timeout=10,
        )
        assert run_events_response.status_code == 200, run_events_response.text
        run_events_result = _mcp_response_payload(run_events_response)["result"]
        run_events = run_events_result["structuredContent"]["events"]
        python_worker_traces = [
            event["payload"]["trace"]
            for event in run_events
            if event["event_type"] == "workflow.trace.v1"
            and event.get("payload", {}).get("trace", {}).get("event")
            in {
                "runtime.python_worker.result_prepared",
                "runtime.python_worker.result_submitted",
            }
        ]
        assert python_worker_traces
        for worker_trace in python_worker_traces:
            assert worker_trace["run_id"] == "python-answer-run"
            assert worker_trace["worker_id"] == "python-answer-worker"
            assert worker_trace["message_id"]
            assert worker_trace["step_id"]
            assert worker_trace["correlation_id"] == "python-answer-run"
        assert {
            trace["event"] for trace in python_worker_traces
        } == {
            "runtime.python_worker.result_prepared",
            "runtime.python_worker.result_submitted",
        }
        worker_trace_bindings = {
            event_name: {
                (
                    trace["message_id"],
                    trace["run_id"],
                    trace["step_id"],
                    trace["correlation_id"],
                )
                for trace in python_worker_traces
                if trace["event"] == event_name
            }
            for event_name in (
                "runtime.python_worker.result_prepared",
                "runtime.python_worker.result_submitted",
            )
        }
        assert worker_trace_bindings["runtime.python_worker.result_prepared"]
        assert (
            worker_trace_bindings["runtime.python_worker.result_prepared"]
            == worker_trace_bindings["runtime.python_worker.result_submitted"]
        )
        checkpoints = httpx.get(
            f"{base_url}/api/workflow/runs/python-answer-run/checkpoints",
            headers=headers,
            timeout=10,
        )
        assert checkpoints.status_code == 200, checkpoints.text
        final_state = checkpoints.json()["checkpoints"][-1]["state"]
        assert final_state["answer"]["text"].startswith("The Rust runtime commits")
        assistant_id = final_state["agentic_answering_result"]["assistant_turn_node_id"]

        # SQLite+Chroma writes the canonical event/job first, then projects a
        # freshly computed vector. PostgreSQL persists the same vector through
        # either the Python graph effect or the configured Rust embedder.
        if backend == "sqlite-chroma":
            deadline = time.monotonic() + 15
            assistant_embedding_done = False
            while time.monotonic() < deadline:
                with sqlite3.connect(
                    f"file:{database_path.as_posix()}?mode=ro", uri=True
                ) as connection:
                    status_row = connection.execute(
                        "SELECT status FROM index_jobs "
                        "WHERE namespace = ? AND entity_id = ? AND index_kind = 'node_embedding'",
                        (graph_namespace, assistant_id),
                    ).fetchone()
                if status_row is not None and status_row[0] == "DONE":
                    assistant_embedding_done = True
                    break
                time.sleep(0.1)
            assert assistant_embedding_done, "Rust embedding worker did not project the assistant node"
            assert embedding_requests
        else:
            import sqlalchemy as sa

            assert pg_dsn is not None and pg_schema is not None
            with sa.create_engine(pg_dsn, future=True).connect() as connection:
                vector_row = connection.execute(
                    sa.text(
                        f'SELECT embedding::text, metadata FROM "{pg_schema}".gke_nodes '
                        "WHERE id = :entity_id"
                    ),
                    {"entity_id": assistant_id},
                ).first()
            assert vector_row is not None
            assert vector_row[0] == "[0.6,0.8,0]"
            assert vector_row[1]["namespace"] == graph_namespace
            if persistence_target == "python-worker":
                assert not embedding_requests, (
                    "PostgreSQL should persist the Python-supplied vector directly"
                )
            else:
                assert any(
                    assistant_id in item
                    for request in embedding_requests
                    for item in request.get("input", [])
                ), "Rust-native persistence should embed the assistant node through the configured provider"
        if backend == "sqlite-chroma":
            assistant_upserts = [
                payload
                for path, payload in zip(chroma_requests, chroma_payloads, strict=True)
                if "/upsert" in path and assistant_id in payload.get("ids", [])
            ]
            assert len(assistant_upserts) == 1
            assert assistant_upserts[0]["embeddings"] == [[0.6, 0.8, 0.0]]
            assert assistant_upserts[0]["metadatas"][0]["namespace"] == graph_namespace

        # Reopen the canonical store after Rust has committed the worker effects.
        process.terminate()
        process.wait(timeout=10)
        if backend == "sqlite-chroma":
            with sqlite3.connect(
                f"file:{database_path.as_posix()}?mode=ro", uri=True
            ) as connection:
                rows = connection.execute(
                    "SELECT entity_kind, entity_id, payload_json FROM entity_events "
                    "WHERE namespace = ?",
                    (graph_namespace,),
                ).fetchall()
                jobs = connection.execute(
                    "SELECT entity_id, index_kind, status FROM index_jobs "
                    "WHERE namespace = ?",
                    (graph_namespace,),
                ).fetchall()
        else:
            import sqlalchemy as sa

            assert pg_dsn is not None and pg_schema is not None
            with sa.create_engine(pg_dsn, future=True).connect() as connection:
                rows = connection.execute(
                    sa.text(
                        f'SELECT entity_kind, entity_id, payload_json FROM "{pg_schema}".entity_events '
                        "WHERE namespace = :namespace"
                    ),
                    {"namespace": graph_namespace},
                ).all()
                jobs = connection.execute(
                    sa.text(
                        f'SELECT entity_id, index_kind, status FROM "{pg_schema}".index_jobs '
                        "WHERE namespace = :namespace"
                    ),
                    {"namespace": graph_namespace},
                ).all()
        assert any(
            kind == "node"
            and entity_id == assistant_id
            and "assistant_turn" in payload
            and "The Rust runtime commits" in payload
            for kind, entity_id, payload in rows
        )
        assistant_payloads = [
            payload
            for kind, entity_id, payload in rows
            if kind == "node" and entity_id == assistant_id
        ]
        assert len(assistant_payloads) == 1
        python_assistant = ConversationNode.model_validate_json(assistant_payloads[0])
        assistant_payload = json.loads(assistant_payloads[0])
        for field in assistant_contract["assistant_node"]["required_fields"]:
            assert field in assistant_payload
        for path, expected in assistant_contract["assistant_node"]["field_values"].items():
            value: Any = assistant_payload
            for part in path.split("."):
                value = value[part]
            assert value == expected
        assert python_assistant.id == assistant_id
        assert python_assistant.summary.startswith("The Rust runtime commits")
        assert python_assistant.role == "assistant"
        assert python_assistant.metadata["entity_type"] == "assistant_turn"
        assert any("context_snapshot" in payload for _, _, payload in rows)
        generated_edge_payloads = [
            payload
            for kind, _, payload in rows
            if kind == "edge"
            and '"relation":"generated"' in payload
            and assistant_id in payload
        ]
        assert len(generated_edge_payloads) == 1
        python_generated_edge = ConversationEdge.model_validate_json(
            generated_edge_payloads[0]
        )
        generated_edge_payload = json.loads(generated_edge_payloads[0])
        for field in assistant_contract["generated_edge"]["required_fields"]:
            assert field in generated_edge_payload
        for path, expected in assistant_contract["generated_edge"]["field_values"].items():
            value = generated_edge_payload
            for part in path.split("."):
                value = value[part]
            assert value == expected
        assert python_generated_edge.relation == "generated"
        assert assistant_id in python_generated_edge.target_ids
        if backend == "pg":
            from kogwistar.engine_core.embedding_profile import EmbeddingProfile
            from kogwistar.engine_core.engine import GraphKnowledgeEngine
            from kogwistar.engine_core.postgres_backend import PgVectorBackend

            assert pg_dsn is not None and pg_schema is not None
            import sqlalchemy as sa

            python_storage_engine = sa.create_engine(pg_dsn, future=True)
            python_graph: GraphKnowledgeEngine | None = None
            try:
                python_backend = PgVectorBackend(
                    engine=python_storage_engine,
                    embedding_dim=3,
                    schema=pg_schema,
                )
                python_graph = GraphKnowledgeEngine(
                    backend=python_backend,
                    namespace=graph_namespace,
                    kg_graph_type="conversation",
                    embedding_function=lambda texts: [
                        [0.0, 0.0, 1.0] for _ in texts
                    ],
                    embedding_profile=EmbeddingProfile(
                        provider="ollama",
                        model="fixture-embedding-model",
                        dimension=3,
                        similarity_metric="cosine",
                        embedding_kind="single_vector",
                    ),
                )
                read_assistant = python_graph.read.get_nodes(
                    ids=[assistant_id], node_type=ConversationNode
                )
                read_generated_edge = python_graph.read.get_edges(
                    ids=[python_generated_edge.id], edge_type=ConversationEdge
                )
                assert len(read_assistant) == 1
                assert read_assistant[0].summary == python_assistant.summary
                assert len(read_generated_edge) == 1
                assert read_generated_edge[0].relation == "generated"
                assert assistant_id in read_generated_edge[0].target_ids
            finally:
                if python_graph is not None:
                    python_graph.close()
                else:
                    python_storage_engine.dispose()
        assistant_jobs = {
            (index_kind, status)
            for entity_id, index_kind, status in jobs
            if entity_id == assistant_id
        }
        expected_assistant_jobs = {
            ("node_docs", "PENDING"),
            ("node_refs", "PENDING"),
        }
        if backend == "sqlite-chroma" or persistence_target == "rust-native":
            expected_assistant_jobs.add(("node_embedding", "DONE"))
        assert assistant_jobs == expected_assistant_jobs
        if backend == "pg" and persistence_target == "python-worker":
            assert not embedding_requests
        if backend == "sqlite-chroma":
            created_collections = {
                payload["name"] for payload in chroma_payloads if "name" in payload
            }
            assert created_collections == {
                "python-answer-nodes",
                "python-answer-edges",
                "nodes_index",
            }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if chroma_server is not None:
            chroma_server.shutdown()
            chroma_server.server_close()
        embedding_server.shutdown()
        embedding_server.server_close()
        if backend == "pg" and pg_dsn is not None and pg_schema is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE'))
                    connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}_wisdom" CASCADE'))
            finally:
                cleanup_engine.dispose()


@pytest.mark.integration
def test_async_authority_runs_async_v2_through_real_rust_sqlite_server(
    monkeypatch,
    tmp_path,
) -> None:
    """Admission and continuation lanes retain async-v2 through real SQLite."""
    from kogwistar.runtime.resolvers import AsyncMappingStepResolver

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
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def start(ctx):
        await asyncio.sleep(0)
        assert ctx.state_view["seed"] == 1
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"async_answer": 42})],
        )

    runtime = SimpleNamespace(
        workflow_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=1,
        cancel_requested=None,
    )
    sqlite_path = tmp_path / "async-authority.sqlite3"
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

        async def invoke() -> RunResult:
            authority = AsyncRustRuntimeAuthority(
                runtime=runtime,
                base_url=base_url,
                cache_dir=tmp_path,
            )
            try:
                return await authority.run(
                    workflow_id="wf-async-live",
                    conversation_id="conv-async-live",
                    turn_node_id="turn-async-live",
                    initial_state={"seed": 1},
                    run_id="run-async-live",
                )
            finally:
                await authority.aclose()

        result = asyncio.run(invoke())
        assert result.status == "succeeded"
        assert result.final_state["async_answer"] == 42
        claimed = httpx.post(
            f"{base_url}/internal/runtime/claim",
            json={"claimed_by": "inspection", "limit": 1, "lease_seconds": 1},
            timeout=10,
        )
        assert claimed.status_code == 200
        assert claimed.json()["work"] == []
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
def test_async_runtime_client_resume_uses_real_rust_authority(
    monkeypatch,
    tmp_path,
    backend: str,
) -> None:
    """Exercise async callbacks and client resume through Rust HTTP/storage."""
    from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live authority test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for Rust/PostgreSQL test")
    pg_schema = f"kw_resume_{os.urandom(8).hex()}" if backend == "pg" else None
    workflow_id = "wf-async-resume-live"

    def node(node_id: str, op: str, **metadata: Any) -> SimpleNamespace:
        return SimpleNamespace(
            id=node_id,
            label=node_id,
            op=op,
            fanout=False,
            metadata={"wf_version": "v1", **metadata},
            safe_get_id=lambda: node_id,
        )

    start = node("start", "start", wf_start=True)
    gate = node("gate", "gate")
    end = node("end", "end", wf_terminal=True)

    def edge(edge_id: str, target_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=edge_id,
            label=edge_id,
            target_ids=[target_id],
            predicate=None,
            multiplicity="one",
            is_default=True,
            priority=100,
            safe_get_id=lambda: edge_id,
        )

    nodes = {item.id: item for item in (start, gate, end)}
    adjacency = {
        "start": [edge("start-gate", "gate")],
        "gate": [edge("gate-end", "end")],
        "end": [],
    }
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (start, nodes, adjacency),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {node_id: 0 for node_id in nodes},
    )

    resolver = AsyncMappingStepResolver()
    visited: list[str] = []
    dependencies = {"opaque_service": object()}

    @resolver.register("start")
    async def start_step(ctx):
        visited.append("start")
        _assert_worker_dependencies(ctx.state_view, dependencies)
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("gate")
    async def gate_step(_ctx):
        visited.append("gate")
        return RunSuspended(
            conversation_node_id=None,
            resume_payload={"prompt": "approve?"},
        )

    @resolver.register("end")
    async def end_step(ctx):
        visited.append("end")
        assert ctx.state_view["approved"] is True
        _assert_worker_dependencies(ctx.state_view, dependencies)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"resumed_end": True})],
        )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "pg" if backend == "pg" else "sqlite",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "",
        }
    )
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        env["KOGWISTAR_PG_DSN"] = pg_dsn
        env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    else:
        env["KOGWISTAR_META_SQLITE_PATH"] = str(tmp_path / "async-resume.sqlite3")
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
            raise AssertionError(f"Rust {backend} server did not become ready")

        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", base_url)
        monkeypatch.setenv(
            "KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR", str(tmp_path / "journal")
        )
        runtime = AsyncWorkflowRuntime(
            workflow_engine=object(),
            conversation_engine=object(),
            step_resolver=resolver,
            predicate_registry={},
            max_workers=1,
            trace=False,
        )

        async def execute() -> tuple[RunResult, RunResult]:
            suspended = await runtime.run(
                workflow_id=workflow_id,
                conversation_id="conv-async-resume-live",
                turn_node_id="turn-async-resume-live",
                initial_state={"seed": 1, "_deps": dependencies},
                run_id="run-async-resume-live",
                cache_dir=str(tmp_path),
            )
            checkpoint = httpx.get(
                f"{base_url}/api/workflow/runs/run-async-resume-live/checkpoints",
                timeout=10,
            )
            checkpoint.raise_for_status()
            latest_state = checkpoint.json()["checkpoints"][-1]["state"]
            token_id = latest_state["_rt_join"]["suspended"][0][2]
            resumed = await runtime.resume_run(
                run_id="run-async-resume-live",
                suspended_node_id="gate",
                suspended_token_id=token_id,
                client_result=RunSuccess(
                    conversation_node_id=None,
                    state_update=[("u", {"approved": True})],
                ),
                workflow_id=workflow_id,
                conversation_id="conv-async-resume-live",
                turn_node_id="turn-async-resume-live",
                cache_dir=str(tmp_path),
            )
            return suspended, resumed

        suspended, resumed = asyncio.run(execute())
        assert suspended.status == "suspended"
        assert resumed.status == "succeeded"
        assert resumed.final_state["approved"] is True
        assert resumed.final_state["resumed_end"] is True
        assert resumed.final_state["_deps"] is dependencies
        assert visited == ["start", "gate", "end"]
        status = httpx.get(
            f"{base_url}/api/workflow/runs/run-async-resume-live", timeout=10
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_engine = sa.create_engine(pg_dsn, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()


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
        response = authority._submit_run({"run_id": "run-1", "workflow_id": "wf-1"})

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
        if (
            request.method == "POST"
            and request.url.path == "/api/workflow/runs/run-1/cancel"
        ):
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
            assert body["effect"]["state_update"] == [["u", {"after_restart": True}]]
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
        metadata={
            "wf_start": True,
            "wf_op": "choose",
            "wf_executor_target": "rust-native",
        },
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
    assert plan["node_versions"] == {"wf|start": "v1", "wf|join": "v1"}
    assert plan["node_environment_ids"] == {}
    assert plan["node_executor_targets"] == {"wf|start": "rust-native"}
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


def test_freeze_runtime_plan_rejects_mismatched_operation_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SimpleNamespace(
        id="wf|start",
        op="process",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "process", "wf_version": "v2"},
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"wf|start": node}, {"wf|start": []}),
    )
    runtime = SimpleNamespace(
        workflow_engine=object(),
        predicate_registry={},
        step_resolver=lambda _op: None,
    )

    with pytest.raises(
        RustRuntimeAuthorityError,
        match="requires 'process' version 'v2'.*registers 'v1'",
    ):
        freeze_runtime_plan(runtime, "wf-1")


def test_freeze_runtime_plan_carries_registered_operation_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SimpleNamespace(
        id="wf|start",
        op="process",
        fanout=False,
        metadata={"wf_start": True, "wf_op": "process", "wf_version": "v2"},
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"wf|start": node}, {"wf|start": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"wf|start": 0},
    )
    resolver = lambda _op: None
    resolver.op_versions = {"process": "v2"}
    runtime = SimpleNamespace(
        workflow_engine=object(), predicate_registry={}, step_resolver=resolver
    )

    plan = freeze_runtime_plan(runtime, "wf-1")

    assert plan["node_versions"] == {"wf|start": "v2"}


def test_freeze_runtime_plan_keeps_historical_operation_version_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.runtime.models import RunSuccess
    from kogwistar.runtime.resolvers import MappingStepResolver

    node = SimpleNamespace(
        id="wf|start",
        op="process",
        fanout=False,
        metadata={
            "wf_start": True,
            "wf_op": "process",
            "wf_version": "v1",
            "wf_environment_id": "process-env-v1",
        },
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"wf|start": node}, {"wf|start": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"wf|start": 0},
    )
    resolver = MappingStepResolver()

    @resolver.register("process", version="v2", environment_id="process-env-v2")
    def current(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register_versioned(
        "process", version="v1", environment_id="process-env-v1"
    )
    def historical(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    runtime = SimpleNamespace(
        workflow_engine=object(), predicate_registry={}, step_resolver=resolver
    )

    plan = freeze_runtime_plan(runtime, "wf-1")

    assert plan["node_versions"] == {"wf|start": "v1"}
    assert plan["node_environment_ids"] == {"wf|start": "process-env-v1"}


def test_freeze_runtime_plan_requires_matching_worker_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SimpleNamespace(
        id="wf|start",
        op="numerical",
        fanout=False,
        metadata={
            "wf_start": True,
            "wf_op": "numerical",
            "wf_environment_id": "science-cpu-v3",
        },
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"wf|start": node}, {"wf|start": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"wf|start": 0},
    )
    resolver = lambda _op: None
    resolver.op_environment_ids = {"numerical": "science-cpu-v3"}
    runtime = SimpleNamespace(
        workflow_engine=object(), predicate_registry={}, step_resolver=resolver
    )

    plan = freeze_runtime_plan(runtime, "wf-1")

    assert plan["node_environment_ids"] == {"wf|start": "science-cpu-v3"}
    resolver.op_environment_ids = {"numerical": "science-cpu-v2"}
    with pytest.raises(RustRuntimeAuthorityError, match="registers 'science-cpu-v2'"):
        freeze_runtime_plan(runtime, "wf-1")


def test_freeze_runtime_plan_pins_python_worker_identity_and_rejects_environment_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "schema_version": 1,
        "environment_id": "science-cpu-v3",
        "artifact_sha256": "a" * 64,
        "dependency_lock_sha256": "b" * 64,
    }
    node = SimpleNamespace(
        id="wf|start",
        op="numerical",
        fanout=False,
        metadata={
            "wf_start": True,
            "wf_op": "numerical",
            "wf_environment_id": "science-cpu-v3",
            "wf_worker_identity": identity,
        },
    )
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.validate_workflow_design",
        lambda **_kwargs: (node, {"wf|start": node}, {"wf|start": []}),
    )
    monkeypatch.setattr(
        "kogwistar.runtime.runtime._compute_may_reach_join_bitsets",
        lambda **_kwargs: {"wf|start": 0},
    )
    resolver = lambda _op: None
    resolver.op_environment_ids = {"numerical": "science-cpu-v3"}
    runtime = SimpleNamespace(
        workflow_engine=object(), predicate_registry={}, step_resolver=resolver
    )

    plan = freeze_runtime_plan(runtime, "wf-1")

    assert plan["node_worker_identities"] == {"wf|start": identity}
    node.metadata["wf_worker_identity"] = {**identity, "environment_id": "stale-env"}
    with pytest.raises(RustRuntimeAuthorityError, match="identity environment"):
        freeze_runtime_plan(runtime, "wf-1")


def test_async_runtime_dispatches_to_async_rust_authority_when_selected(
    monkeypatch,
) -> None:
    from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", "http://rust")
    runtime = AsyncWorkflowRuntime.__new__(AsyncWorkflowRuntime)
    runtime.workflow_engine = None
    runtime.conversation_engine = None
    expected = RunResult(
        run_id="run-1", final_state={"answer": 42}, mq=__import__("queue").Queue()
    )
    calls: list[dict[str, Any]] = []

    async def dispatch(_runtime: Any, **kwargs: Any) -> RunResult:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.run_with_rust_authority_async",
        dispatch,
    )

    async def invoke() -> RunResult:
        return await runtime.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={"seed": 1},
            run_id="run-1",
            cache_dir="cache",
        )

    assert asyncio.run(invoke()) is expected
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


def test_async_runtime_rust_mode_without_authority_url_fails_closed(
    monkeypatch,
) -> None:
    from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.delenv("KOGWISTAR_RUST_RUNTIME_URL", raising=False)
    runtime = AsyncWorkflowRuntime.__new__(AsyncWorkflowRuntime)
    runtime.workflow_engine = None
    runtime.conversation_engine = None

    async def invoke() -> None:
        await runtime.run(
            workflow_id="wf-1",
            conversation_id="conv-1",
            turn_node_id="turn-1",
            initial_state={},
        )

    with pytest.raises(RustRuntimeAuthorityError, match="URL is not configured"):
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
    resolver = MappingStepResolver(
        default=lambda _ctx: RunSuccess(conversation_node_id=None)
    )
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
    validated_plan = {
        "runtime_routes": [
            {"source_node_id": "gate", "target_node_id": "done", "join_mask": 0}
        ]
    }
    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.freeze_runtime_plan",
        lambda *_args: pytest.fail("resume effect refroze an already validated plan"),
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
            runtime_plan=validated_plan,
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
            runtime_plan=validated_plan,
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

    for key in (
        "_deps",
        "dream_deps",
        "_rt_node_ops",
        "_rt_node_executor_targets",
        "_rt_join",
    ):
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
        step_resolver=MappingStepResolver(
            default=lambda _ctx: RunSuccess(conversation_node_id=None)
        ),
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


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["sqlite", "sqlite-chroma", "pg"])
@pytest.mark.parametrize("execution_mode", ["external-worker", "inline-server"])
def test_rust_native_workflow_runs_through_real_server_without_python_worker(
    tmp_path: Path,
    backend: str,
    execution_mode: str,
    record_property,
) -> None:
    """Exercise an all-native workflow over the packaged Rust server HTTP API."""
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live native-server test")
    pg_dsn = next(
        (
            os.getenv(name)
            for name in ("GKE_PG_DSN", "PG_DSN", "DATABASE_URL")
            if os.getenv(name)
        ),
        None,
    )
    if backend == "pg" and not pg_dsn:
        pytest.skip("set GKE_PG_DSN, PG_DSN, or DATABASE_URL for live PostgreSQL test")

    suffix = os.urandom(8).hex()
    workflow_contract = json.loads(
        (
            ROOT
            / "contracts"
            / "golden"
            / "runtime-native-increment-workflow-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_id = f"native-http-{suffix}"
    pg_schema = f"kw_native_http_{suffix}" if backend == "pg" else None
    chroma_server = None
    chroma_requests: list[str] = []
    chroma_endpoint = None
    if backend == "sqlite-chroma":
        chroma_server, chroma_endpoint, chroma_requests, _ = (
            _start_fake_chroma_collections()
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "pg" if backend == "pg" else "sqlite",
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
            "KOGWISTAR_RUNTIME_NATIVE_INLINE": str(execution_mode == "inline-server").lower(),
        }
    )
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        env["KOGWISTAR_PG_DSN"] = pg_dsn
        env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
    else:
        env["KOGWISTAR_META_SQLITE_PATH"] = str(tmp_path / "native-http.sqlite3")
        if backend == "sqlite-chroma":
            assert chroma_endpoint is not None
            env["KOGWISTAR_CHROMA_ENDPOINT"] = chroma_endpoint
            env["KOGWISTAR_CHROMA_NODE_COLLECTION"] = "native-runtime-nodes"
            env["KOGWISTAR_CHROMA_EDGE_COLLECTION"] = "native-runtime-edges"

    process = subprocess.Popen(
        [str(RUST_SERVER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    headers = {"x-kogwistar-roles": "reader"}
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
            raise AssertionError(f"Rust {backend} server did not become ready")

        run_started_at = time.perf_counter()
        submitted = httpx.post(
            f"{base_url}/api/workflow/runs",
            headers=headers,
            json={
                "run_id": run_id,
                "workflow_id": workflow_contract["workflow_id"],
                "conversation_id": "rust-native-http-conversation",
                "initial_state": workflow_contract["initial_state"],
                "runtime_kind": "sync",
                "start_node_id": workflow_contract["nodes"][0]["id"],
                "node_ops": {
                    node["id"]: f"rust-native/{node['operation']}"
                    for node in workflow_contract["nodes"]
                },
                "node_executor_targets": {
                    node["id"]: "rust-native" for node in workflow_contract["nodes"]
                },
                "runtime_routes": workflow_contract["routes"],
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text

        if execution_mode == "external-worker":
            executed_steps: list[str] = []
            expected_step_ids = [node["id"] for node in workflow_contract["nodes"]]
            for expected_step in expected_step_ids:
                claimed = httpx.post(
                    f"{base_url}/internal/runtime/claim",
                    headers=headers,
                    json={
                        "claimed_by": "rust-native-worker",
                        "limit": 1,
                        "lease_seconds": 60,
                        "recipient_ids": ["rust-native-worker"],
                        "run_id": run_id,
                    },
                    timeout=10,
                )
                assert claimed.status_code == 200, claimed.text
                work_items = claimed.json()["work"]
                assert len(work_items) == 1
                work = work_items[0]
                assert work["step_id"] == expected_step
                payload = work["payload"]
                assert payload["executor_target"] == "rust-native"
                assert payload["op"] == "rust-native/increment-counter"
                executed = httpx.post(
                    f"{base_url}/internal/runtime/native/execute",
                    headers=headers,
                    json={
                        "handoff": {
                            key: work[key]
                            for key in (
                                "message_id",
                                "claimed_by",
                                "run_id",
                                "step_id",
                                "correlation_id",
                            )
                        },
                        "payload": payload,
                    },
                    timeout=10,
                )
                assert executed.status_code == 200, executed.text
                executed_steps.append(work["step_id"])
            assert executed_steps == expected_step_ids
            execution_completed_at = time.perf_counter()
        else:
            for _ in range(200):
                status = httpx.get(
                    f"{base_url}/api/workflow/runs/{run_id}",
                    headers=headers,
                    timeout=10,
                )
                assert status.status_code == 200, status.text
                if status.json()["status"] == "succeeded":
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("inline Rust server did not finish the native run")
            execution_completed_at = time.perf_counter()
            python_run_id = f"python-http-{suffix}"
            python_submitted = httpx.post(
                f"{base_url}/api/workflow/runs",
                headers=headers,
                json={
                    "run_id": python_run_id,
                    "workflow_id": "python-worker-compatibility",
                    "conversation_id": "python-worker-compatibility",
                    "initial_state": {},
                    "runtime_kind": "sync",
                    "start_node_id": "python-step",
                    "node_ops": {"python-step": "custom-operation"},
                    "node_executor_targets": {"python-step": "python-worker"},
                    "runtime_routes": [],
                },
                timeout=10,
            )
            assert python_submitted.status_code == 202, python_submitted.text
            python_claim = httpx.post(
                f"{base_url}/internal/runtime/claim",
                headers=headers,
                json={
                    "claimed_by": "python-worker-test",
                    "limit": 1,
                    "lease_seconds": 60,
                    "recipient_ids": ["python-worker"],
                    "run_id": python_run_id,
                },
                timeout=10,
            )
            assert python_claim.status_code == 200, python_claim.text
            python_work = python_claim.json()["work"]
            assert len(python_work) == 1
            assert python_work[0]["step_id"] == "python-step"
            assert python_work[0]["payload"]["executor_target"] == "python-worker"
        record_property("execution_mode", f"{execution_mode}:{backend}")
        record_property(
            "submit_to_native_completion_ms",
            round((execution_completed_at - run_started_at) * 1000, 3),
        )
        status = httpx.get(
            f"{base_url}/api/workflow/runs/{run_id}", headers=headers, timeout=10
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "succeeded"
        checkpoints = httpx.get(
            f"{base_url}/api/workflow/runs/{run_id}/checkpoints",
            headers=headers,
            timeout=10,
        )
        assert checkpoints.status_code == 200, checkpoints.text
        assert (
            checkpoints.json()["checkpoints"][-1]["state"]["counter"]
            == workflow_contract["expected_final_state"]["counter"]
        )
        steps_response = httpx.get(
            f"{base_url}/api/workflow/runs/{run_id}/steps",
            headers=headers,
            timeout=10,
        )
        assert steps_response.status_code == 200, steps_response.text
        steps = steps_response.json()["steps"]
        assert [step["workflow_node_id"] for step in steps] == [
            node["id"] for node in workflow_contract["nodes"]
        ]
        mcp_events = httpx.post(
            f"{base_url}/mcp/workflow",
            headers={
                **headers,
                "accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": "native-run-events",
                "method": "tools/call",
                "params": {
                    "name": "workflow.run_events",
                    "arguments": {"run_id": run_id, "after_seq": 0, "limit": 100},
                },
            },
            timeout=10,
        )
        assert mcp_events.status_code == 200, mcp_events.text
        mcp_result = _mcp_response_payload(mcp_events)["result"]
        event_list = mcp_result["structuredContent"]["events"]
        assert any(event["event_type"] == "run.created" for event in event_list)
        native_traces = [
            event["payload"]["trace"]
            for event in event_list
            if event["event_type"] == "workflow.trace.v1"
            and "runtime.native.step.completed" in json.dumps(event["payload"])
        ]
        assert len(native_traces) == 2
        expected_completed_steps = workflow_contract["expected_completed_steps"]
        assert [trace["node_id"] for trace in native_traces] == expected_completed_steps
        assert [trace["step_seq"] for trace in native_traces] == workflow_contract[
            "expected_step_sequences"
        ]
        for trace in native_traces:
            assert trace["implementation_authority"] == "rust-native"
            assert trace["seam_crossings"] == []
            assert trace["run_id"] == run_id
        assert len(steps) == len(native_traces)
        for step, trace in zip(steps, native_traces, strict=True):
            assert step["step_seq"] == trace["step_seq"]
            assert step["workflow_node_id"] == trace["node_id"]
            assert step["trace_context"] == {
                "trace_id": trace["trace_id"],
                "span_id": trace["span_id"],
                "parent_span_id": trace["parent_span_id"],
                "attempt": trace["attempt"],
            }
        trace_contract = workflow_contract["trace_contract"]
        assert [trace["attempt"] for trace in native_traces] == trace_contract[
            "expected_attempts"
        ]
        assert all(trace["trace_id"] == run_id for trace in native_traces)
        if trace_contract["require_nonempty_span_ids"]:
            assert all(trace["span_id"] and trace["parent_span_id"] for trace in native_traces)
        if trace_contract["require_distinct_parent_span_ids"]:
            assert all(
                trace["span_id"] != trace["parent_span_id"] for trace in native_traces
            )
        if backend == "sqlite-chroma":
            assert any("/collections" in path for path in chroma_requests)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if pg_schema is not None and pg_dsn is not None:
            import sqlalchemy as sa

            cleanup_url = sa.engine.make_url(pg_dsn)
            if cleanup_url.drivername in {"postgres", "postgresql"}:
                cleanup_url = cleanup_url.set(drivername="postgresql+psycopg")
            cleanup_engine = sa.create_engine(cleanup_url, future=True)
            try:
                with cleanup_engine.begin() as connection:
                    connection.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{pg_schema}" CASCADE')
                    )
            finally:
                cleanup_engine.dispose()
        if chroma_server is not None:
            chroma_server.shutdown()
            chroma_server.server_close()
