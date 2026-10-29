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
import time
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
)
from kogwistar.runtime.rust_runtime_authority import (
    AsyncRustRuntimeAuthority,
    RustRuntimeAuthority,
    RustRuntimeAuthorityError,
    _server_managed_inline_worker_enabled,
    _server_advertises_inline_native_worker,
    _python_worker_required_for_plan,
    _runtime_plan_requires_python_worker,
    _validate_resume_plan,
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
        token_id = latest_state["_rt_join"]["suspended"][0][2]
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
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
def test_python_worker_and_rust_native_step_interoperate_over_real_server(
    tmp_path: Path, backend: str
) -> None:
    """Exercise mixed execution and lost-ack replay on Rust SQLite/PostgreSQL."""
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
    graph_namespace = f"worker-graph-{os.urandom(8).hex()}" if backend == "pg" else None
    graph_entity_id = f"worker-effect-{os.urandom(8).hex()}" if backend == "pg" else None
    graph_sqlalchemy_engine = None
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        import sqlalchemy as sa

        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        graph_sqlalchemy_engine = sa.create_engine(pg_dsn, future=True)
        PgVectorBackend(
            engine=graph_sqlalchemy_engine,
            embedding_dim=3,
            schema=pg_schema,
        ).ensure_schema()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": backend,
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "reader",
        }
    )
    worker_token = None
    if backend == "pg":
        assert pg_dsn is not None and pg_schema is not None
        env["KOGWISTAR_PG_DSN"] = pg_dsn
        env["KOGWISTAR_PG_SCHEMA_BASE"] = pg_schema
        jwt_secret = f"test-{os.urandom(24).hex()}"
        env["JWT_SECRET"] = jwt_secret
        env["JWT_ALG"] = "HS256"
        assert graph_namespace is not None
        worker_token = _test_hs256_token(
            jwt_secret,
            {
                "sub": "worker-replay-test",
                "role": "rw",
                "roles": ["reader", "rw"],
                "ns": "workflow",
                "storage_ns": graph_namespace,
                "capabilities": ["write_graph"],
                "exp": int(time.time()) + 3600,
            },
        )
    else:
        env["KOGWISTAR_META_SQLITE_PATH"] = str(tmp_path / "mixed-language.sqlite3")
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
                "initial_state": {"counter": 4},
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
        ]
        worker_env = os.environ.copy()
        if backend == "pg":
            assert worker_token is not None
            assert graph_namespace is not None and graph_entity_id is not None
            worker_probe.extend(
                [
                    "--graph-namespace",
                    graph_namespace,
                    "--graph-entity-id",
                    graph_entity_id,
                    "--auth-token",
                    worker_token,
                ]
            )
            worker_env["JWT_SECRET"] = env["JWT_SECRET"]
            worker_env["JWT_ALG"] = "HS256"
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
        # result. Worker B must reclaim after lease expiry and replay the result.
        time.sleep(1.1)
        replay = subprocess.run(
            [*worker_probe, "--worker-id", "python-worker-b", "--mode", "replay"],
            cwd=ROOT,
            env=worker_env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert replay.returncode == 0, replay.stderr
        assert callback_marker.read_text(encoding="utf-8").splitlines() == ["callback"]
        if backend == "pg":
            assert pg_dsn is not None and pg_schema is not None
            assert graph_namespace is not None and graph_entity_id is not None
            from kogwistar.engine_core.rust_postgres_session import (
                RustEnginePostgresMetaStore,
            )

            graph_store = RustEnginePostgresMetaStore(dsn=pg_dsn, schema=pg_schema)
            graph_events = list(
                graph_store.iter_entity_events(namespace=graph_namespace)
            )
            assert len(graph_events) == 1
            assert graph_events[0][1:4] == ("node", graph_entity_id, "ADD")
            graph_jobs = graph_store.list_index_jobs(namespace=graph_namespace)
            assert len(graph_jobs) == 1
            assert graph_jobs[0].entity_id == graph_entity_id
            assert graph_sqlalchemy_engine is not None
            from kogwistar.engine_core.postgres_backend import PgVectorBackend

            projected = PgVectorBackend(
                engine=graph_sqlalchemy_engine,
                embedding_dim=3,
                schema=pg_schema,
            ).node_get(ids=[graph_entity_id], include=["documents"])
            assert projected["ids"] == [graph_entity_id]
            assert projected["documents"] == ["worker effect"]
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
        assert status.json()["status"] == "succeeded"
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

        from kogwistar.runtime.resolvers import AsyncMappingStepResolver

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
                if graph_sqlalchemy_engine is not None:
                    graph_sqlalchemy_engine.dispose()


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
@pytest.mark.parametrize("backend", ["sqlite", "pg"])
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
    run_id = f"native-http-{suffix}"
    pg_schema = f"kw_native_http_{suffix}" if backend == "pg" else None
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
                "workflow_id": "rust-native-http-workflow",
                "conversation_id": "rust-native-http-conversation",
                "initial_state": {"counter": 0},
                "runtime_kind": "sync",
                "start_node_id": "first",
                "node_ops": {
                    "first": "rust-native/increment-counter",
                    "second": "rust-native/increment-counter",
                },
                "node_executor_targets": {
                    "first": "rust-native",
                    "second": "rust-native",
                },
                "runtime_routes": [
                    {
                        "source_node_id": "first",
                        "target_node_id": "second",
                        "join_mask": 0,
                    }
                ],
            },
            timeout=10,
        )
        assert submitted.status_code == 202, submitted.text

        if execution_mode == "external-worker":
            executed_steps: list[str] = []
            for expected_step in ("first", "second"):
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
            assert executed_steps == ["first", "second"]
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
        assert checkpoints.json()["checkpoints"][-1]["state"]["counter"] == 2
        steps_response = httpx.get(
            f"{base_url}/api/workflow/runs/{run_id}/steps",
            headers=headers,
            timeout=10,
        )
        assert steps_response.status_code == 200, steps_response.text
        steps = steps_response.json()["steps"]
        assert [step["workflow_node_id"] for step in steps] == ["first", "second"]
        mcp_events = httpx.post(
            f"{base_url}/mcp/workflow",
            headers=headers,
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
        event_list = mcp_events.json()["result"]["events"]
        assert any(event["event_type"] == "run.created" for event in event_list)
        native_traces = [
            event["payload"]["trace"]
            for event in event_list
            if event["event_type"] == "workflow.trace.v1"
            and "runtime.native.step.completed" in json.dumps(event["payload"])
        ]
        assert len(native_traces) == 2
        for trace in native_traces:
            assert trace["implementation_authority"] == "rust-native"
            assert trace["seam_crossings"] == []
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
