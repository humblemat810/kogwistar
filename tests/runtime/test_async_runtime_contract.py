from __future__ import annotations

import asyncio
import time

import pytest

from kogwistar.runtime import (
    AsyncMappingStepResolver,
    AsyncWorkflowRuntime,
    MappingStepResolver,
)
from kogwistar.runtime.async_runtime import _as_sync_step_fn, _SyncResolverAdapter
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.runtime import RunResult, StepContext
import queue

pytestmark = [pytest.mark.ci]


class _DummyCtx:
    run_id = "r1"
    workflow_id = "wf1"
    workflow_node_id = "n1"
    op = "op1"
    token_id = "t1"
    attempt = 1
    step_seq = 1
    conversation_id = "c1"
    turn_node_id = "turn1"
    state_view = {}


def test_async_runtime_exported():
    from kogwistar.runtime import AsyncWorkflowRuntime as Exported
    from kogwistar.runtime import AsyncMappingStepResolver as ExportedAsyncResolver

    assert Exported is AsyncWorkflowRuntime
    assert ExportedAsyncResolver is AsyncMappingStepResolver


def test_default_sync_ops_equal_default_async_ops():
    from kogwistar.conversation.resolvers import default_resolver

    async_resolver = AsyncMappingStepResolver(
        handlers=dict(default_resolver.handlers),
        default=default_resolver.default,
    )
    assert set(default_resolver.ops) == set(async_resolver.ops)


def test_async_adapter_accepts_sync_handler():
    def _sync(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "v"})])

    wrapped = _as_sync_step_fn(_sync)
    out = wrapped(_DummyCtx())
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"k": "v"})]


def test_async_adapter_accepts_async_handler():
    async def _async(_ctx):
        await asyncio.sleep(0.001)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"async_key": "ok"})],
        )

    wrapped = _as_sync_step_fn(_async)
    out = wrapped(_DummyCtx())
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"async_key": "ok"})]


def test_missing_op_behavior_matches_sync_resolver():
    sync_resolver = MappingStepResolver()
    async_resolver = AsyncMappingStepResolver()

    with pytest.raises(KeyError):
        sync_resolver.resolve("missing")
    with pytest.raises(KeyError):
        async_resolver.resolve_async("missing")


def test_sync_runtime_rejects_async_handler_by_runfailure():
    sync_resolver = MappingStepResolver()

    class _AwaitableResult:
        def __await__(self):
            async def _coro():
                return RunSuccess(conversation_node_id=None, state_update=[])

            return _coro().__await__()

    @sync_resolver.register("async_in_sync")
    def _async_like_handler(_ctx):
        return _AwaitableResult()

    out = sync_resolver.resolve("async_in_sync")(_DummyCtx())
    assert isinstance(out, RunFailure)
    assert any("Resolver must return StepRunResult" in e for e in out.errors)


def test_exception_to_runfailure_matches_sync_resolver():
    sync_resolver = MappingStepResolver()
    async_resolver = AsyncMappingStepResolver()

    @sync_resolver.register("boom")
    def _boom_sync(_ctx):
        raise RuntimeError("sync boom")

    @async_resolver.register("boom")
    def _boom_async(_ctx):
        raise RuntimeError("async boom")

    sync_out = sync_resolver.resolve("boom")(_DummyCtx())
    async_out = asyncio.run(async_resolver.resolve_async("boom")(_DummyCtx()))
    assert isinstance(sync_out, RunFailure)
    assert isinstance(async_out, RunFailure)
    assert any("sync boom" in e for e in sync_out.errors)
    assert any("async boom" in e for e in async_out.errors)


def test_registered_async_op_appears_in_async_op_set():
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("async_op")
    async def _async_ok(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "ok"})])

    assert "async_op" in async_resolver.ops
    out = asyncio.run(async_resolver.resolve_async("async_op")(_DummyCtx()))
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"k": "ok"})]


def test_async_resolver_preserves_sandboxed_behavior():
    async_resolver = AsyncMappingStepResolver()

    class _FakeSandbox:
        def run(self, code, state, context):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sandbox_code": code, "sandbox_ctx_op": context.get("op")})],
            )

    async_resolver.set_sandbox(_FakeSandbox())
    async_resolver.sandboxed_ops.add("py")

    @async_resolver.register("py")
    async def _sandboxed(_ctx):
        return "result = {'state_update': []}"

    out = asyncio.run(async_resolver.resolve_async("py")(_DummyCtx()))
    assert isinstance(out, RunSuccess)
    assert out.state_update[0][1]["sandbox_ctx_op"] == "op1"


def test_async_resolver_offloads_blocking_sync_handlers():
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("slow")
    def _slow(_ctx):
        time.sleep(0.20)
        return RunSuccess(conversation_node_id=None, state_update=[])

    async def _run_two():
        fn = async_resolver.resolve_async("slow")
        t0 = time.perf_counter()
        r1, r2 = await asyncio.gather(fn(_DummyCtx()), fn(_DummyCtx()))
        dt = time.perf_counter() - t0
        return r1, r2, dt

    r1, r2, dt = asyncio.run(_run_two())
    assert isinstance(r1, RunSuccess)
    assert isinstance(r2, RunSuccess)
    assert dt < 0.35


def test_async_resolver_runs_two_awaited_handlers_concurrently():
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("awaited")
    async def _awaited(_ctx):
        await asyncio.sleep(0.20)
        return RunSuccess(conversation_node_id=None, state_update=[])

    async def _run_two():
        fn = async_resolver.resolve_async("awaited")
        t0 = time.perf_counter()
        r1, r2 = await asyncio.gather(fn(_DummyCtx()), fn(_DummyCtx()))
        dt = time.perf_counter() - t0
        return r1, r2, dt

    r1, r2, dt = asyncio.run(_run_two())
    assert isinstance(r1, RunSuccess)
    assert isinstance(r2, RunSuccess)
    assert dt < 0.30


def test_async_resolver_state_schema_metadata_available():
    async_resolver = AsyncMappingStepResolver()
    async_resolver.set_state_schema({"events": "a", "answer": "u"})
    assert async_resolver.describe_state() == {"events": "a", "answer": "u"}


def test_async_resolver_deps_available_in_handler():
    async_resolver = AsyncMappingStepResolver()

    class _DepsCtx(_DummyCtx):
        state_view = {"_deps": {"x": 7}}

    @async_resolver.register("use_deps")
    async def _use_deps(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"x": ctx.state_view["_deps"]["x"]})],
        )

    out = asyncio.run(async_resolver.resolve_async("use_deps")(_DepsCtx()))
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"x": 7})]


def test_async_resolver_legacy_update_warning_preserved():
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("legacy")
    async def _legacy(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], update={"k": 1})

    with pytest.warns(RuntimeWarning, match="legacy update detected"):
        out = asyncio.run(async_resolver.resolve_async("legacy")(_DummyCtx()))
    assert isinstance(out, RunSuccess)


def test_async_runtime_step_context_and_result_contract(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.workflow_engine = kwargs["workflow_engine"]
            self.conversation_engine = kwargs["conversation_engine"]
            self.step_resolver = kwargs["step_resolver"]

        def run(self, **kwargs):
            return kwargs

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    async_runtime = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )
    assert async_runtime.step_context_type is StepContext
    assert async_runtime.step_result_type is not None
    assert async_runtime.terminal_status_values == {
        "succeeded",
        "failed",
        "cancelled",
        "suspended",
    }


def test_sync_and_async_runtime_accept_same_workflow_graph_model(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.workflow_engine = kwargs["workflow_engine"]
            self.conversation_engine = kwargs["conversation_engine"]
            self.step_resolver = kwargs["step_resolver"]

        def run(self, **kwargs):
            return kwargs

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    workflow_engine = {"graph_model": "WorkflowSpec-like"}
    conversation_engine = {"graph_model": "ConversationGraph"}
    sync_resolver = MappingStepResolver()
    async_runtime = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=sync_resolver,
        predicate_registry={},
        trace=False,
    )
    assert async_runtime.sync_runtime.workflow_engine is workflow_engine
    assert async_runtime.sync_runtime.conversation_engine is conversation_engine


def test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter():
    resolver = MappingStepResolver()
    resolver.nested_ops.add("answer")
    resolver.set_state_schema({"events": "a"})

    adapter = _SyncResolverAdapter(resolver)
    assert "answer" in adapter.nested_ops
    assert adapter._state_schema == {"events": "a"}


def test_async_runtime_adapter_forwards_close_sandbox_run():
    class _Resolver:
        def __init__(self):
            self.closed: list[str] = []

        def __call__(self, _op: str):
            return lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])

        def close_sandbox_run(self, run_id: str) -> None:
            self.closed.append(run_id)

    resolver = _Resolver()
    adapter = _SyncResolverAdapter(resolver)
    adapter.close_sandbox_run("run-123")
    assert resolver.closed == ["run-123"]


def test_async_runtime_linear_terminal_status_equivalent_to_sync(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            state = dict(kwargs.get("initial_state") or {})
            state["path"] = "linear"
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-linear"),
                final_state=state,
                mq=queue.Queue(),
                status="succeeded",
            )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )
    sync_out = rt.run_sync(
        workflow_id="wf-linear",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={"k": "v"},
        run_id="r-linear",
    )
    async_out = asyncio.run(
        rt.run(
            workflow_id="wf-linear",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={"k": "v"},
            run_id="r-linear",
        )
    )
    assert async_out.status == sync_out.status == "succeeded"
    assert async_out.final_state == sync_out.final_state


def test_async_runtime_branch_join_status_and_state_equivalent_to_sync(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            state = dict(kwargs.get("initial_state") or {})
            branch_values = list(state.get("branch_values") or [])
            state["joined_total"] = sum(int(v) for v in branch_values)
            state["path"] = "branch_join"
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-branch-join"),
                final_state=state,
                mq=queue.Queue(),
                status="succeeded",
            )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )
    initial = {"branch_values": [2, 3, 5]}
    sync_out = rt.run_sync(
        workflow_id="wf-branch-join",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state=initial,
        run_id="r-branch-join",
    )
    async_out = asyncio.run(
        rt.run(
            workflow_id="wf-branch-join",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state=initial,
            run_id="r-branch-join",
        )
    )
    assert async_out.status == sync_out.status == "succeeded"
    assert async_out.final_state == sync_out.final_state
    assert async_out.final_state["joined_total"] == 10


def test_async_runtime_deps_live_but_omitted_from_checkpoint_payload(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.step_resolver = kwargs["step_resolver"]

        def run(self, **kwargs):
            initial_state = dict(kwargs.get("initial_state") or {})
            # Simulate runtime checkpoint serialization rule: _deps is process-local.
            checkpoint_state = {k: v for k, v in initial_state.items() if k != "_deps"}

            # Simulate one live step invocation that can still read _deps.
            class _Ctx:
                state_view = initial_state

            live_out = self.step_resolver("use_deps")(_Ctx())
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-deps"),
                final_state={
                    "live_x": live_out.state_update[0][1]["x"],
                    "checkpoint_state": checkpoint_state,
                },
                mq=queue.Queue(),
                status="succeeded",
            )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )

    resolver = MappingStepResolver()

    @resolver.register("use_deps")
    def _use_deps(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"x": ctx.state_view["_deps"]["x"]})],
        )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-deps",
            conversation_id="c-deps",
            turn_node_id="t-deps",
            initial_state={"_deps": {"x": 9}, "keep": 1},
            run_id="r-deps",
        )
    )
    assert out.status == "succeeded"
    assert out.final_state["live_x"] == 9
    assert "_deps" not in out.final_state["checkpoint_state"]
    assert out.final_state["checkpoint_state"]["keep"] == 1


def test_async_runtime_native_scheduler_linear_success(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        end = _Node("end", "end", terminal=True, fanout=False)
        return start, {"start": start, "end": end}, {"start": [_Edge("end")], "end": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"path": "linear"})])

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-linear",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert out.final_state["path"] == "linear"
    assert out.final_state["done"] is True


def test_async_runtime_native_scheduler_fanout_appends(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"branches": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"branches": "b"})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-fanout",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert sorted(out.final_state.get("branches", [])) == ["a", "b"]


def test_async_runtime_native_scheduler_cancellation_drains_inflight(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=False, fanout=False)
        b = _Node("b", "b", terminal=False, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()
    inflight_counter = {"n": 0}
    started = {"n": 0}

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        inflight_counter["n"] += 1
        started["n"] += 1
        try:
            await asyncio.sleep(0.5)
            return RunSuccess(conversation_node_id=None, state_update=[])
        finally:
            inflight_counter["n"] -= 1

    @resolver.register("b")
    async def _b(_ctx):
        inflight_counter["n"] += 1
        started["n"] += 1
        try:
            await asyncio.sleep(0.5)
            return RunSuccess(conversation_node_id=None, state_update=[])
        finally:
            inflight_counter["n"] -= 1

    cancel_requested = {"flag": False}

    def _cancel(run_id: str) -> bool:
        return cancel_requested["flag"]

    async def _flip_cancel():
        await asyncio.sleep(0.12)
        cancel_requested["flag"] = True

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
        cancel_requested=_cancel,
        max_workers=4,
    )

    async def _run():
        flip = asyncio.create_task(_flip_cancel())
        out = await rt.run(
            workflow_id="wf-native-cancel",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
        await flip
        return out

    out = asyncio.run(_run())
    assert started["n"] >= 1
    assert out.status == "cancelled"
    assert inflight_counter["n"] == 0


def test_async_runtime_native_scheduler_route_next_and_priority(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst, *, label: str, priority: int):
            self.target_ids = [dst]
            self.label = label
            self.metadata = {
                "wf_priority": priority,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        low = _Node("low", "low", terminal=True, fanout=False)
        high = _Node("high", "high", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "low": low, "high": high},
            {
                # lower priority number first
                "start": [
                    _Edge("high", label="to_high", priority=200),
                    _Edge("low", label="to_low", priority=10),
                ],
                "low": [],
                "high": [],
            },
        )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        # route-next should override plain default-edge selection.
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            next_step_names=["to_high"],
        )

    @resolver.register("low")
    async def _low(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"picked": "low"})])

    @resolver.register("high")
    async def _high(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"picked": "high"})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-route-next",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert out.final_state["picked"] == "high"


def test_async_runtime_native_scheduler_token_nesting_and_spawn_events(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"start_token": ctx.token_id})],
        )

    @resolver.register("a")
    async def _a(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"branch_tokens": ctx.token_id})],
        )

    @resolver.register("b")
    async def _b(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"branch_tokens": ctx.token_id})],
        )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-token-spawn",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    start_token = out.final_state["start_token"]
    branch_tokens = list(out.final_state["branch_tokens"])
    assert len(branch_tokens) == 2
    assert len(set(branch_tokens)) == 2
    assert start_token in branch_tokens
    spawn_events = []
    while True:
        try:
            evt = out.mq.get_nowait()
        except queue.Empty:
            break
        if evt.get("type") == "token.spawn":
            spawn_events.append(evt)
    assert len(spawn_events) >= 1
    assert spawn_events[0]["parent_token_id"] == start_token
    assert spawn_events[0]["child_token_id"] in branch_tokens
    assert spawn_events[0]["child_token_id"] != start_token


def test_async_runtime_native_scheduler_join_merge_runs_once(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False, wf_join=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout, "wf_join": wf_join}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=False, fanout=False)
        b = _Node("b", "b", terminal=False, fanout=False)
        j = _Node("join", "join", terminal=False, fanout=False, wf_join=True)
        end = _Node("end", "end", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "a": a, "b": b, "join": j, "end": end},
            {
                "start": [_Edge("a"), _Edge("b")],
                "a": [_Edge("join")],
                "b": [_Edge("join")],
                "join": [_Edge("end")],
                "end": [],
            },
        )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "b"})])

    @resolver.register("join")
    async def _join(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"join_calls": int(ctx.state_view.get("join_calls", 0)) + 1})],
        )

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-join-merge",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert sorted(out.final_state.get("parts", [])) == ["a", "b"]
    assert out.final_state["join_calls"] == 1
    assert out.final_state["done"] is True


def test_async_runtime_native_scheduler_without_join_executes_once_per_token(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def apply_state_update(self, state, result):
            for mode, payload in result.state_update:
                if mode == "u":
                    state.update(payload)
                elif mode == "a":
                    for k, v in payload.items():
                        state.setdefault(k, []).append(v)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=False, fanout=False)
        b = _Node("b", "b", terminal=False, fanout=False)
        c = _Node("c", "c", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "a": a, "b": b, "c": c},
            {"start": [_Edge("a"), _Edge("b")], "a": [_Edge("c")], "b": [_Edge("c")], "c": []},
        )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("c")
    async def _c(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"c_tokens": ctx.token_id})],
        )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-converge-no-join",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    c_tokens = list(out.final_state.get("c_tokens", []))
    assert len(c_tokens) == 2
    assert len(set(c_tokens)) == 2


def test_async_runtime_native_scheduler_respects_many_multiplicity(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst, *, multiplicity="one"):
            self.target_ids = [dst]
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": multiplicity,
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a", multiplicity="many"), _Edge("b", multiplicity="many")], "a": [], "b": []},
        )

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"targets": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"targets": "b"})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-many",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert sorted(out.final_state.get("targets", [])) == ["a", "b"]


def test_async_runtime_native_scheduler_persists_rt_join_frontier_shape(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=False, fanout=False)
        b = _Node("b", "b", terminal=False, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    cancel_requested = {"flag": False}

    def _cancel(_run_id: str) -> bool:
        return cancel_requested["flag"]

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
        cancel_requested=_cancel,
    )

    async def _run():
        cancel_requested["flag"] = True
        return await rt.run(
            workflow_id="wf-native-rt-join",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )

    out = asyncio.run(_run())
    rt_join = out.final_state.get("_rt_join") or {}
    assert set(rt_join.keys()) >= {"join_node_ids", "join_outstanding", "join_waiters", "pending", "suspended"}


def test_async_runtime_resume_run_delegates_to_sync_resume(monkeypatch):
    calls = {}

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            calls["run"] = kwargs
            return RunResult(run_id="run-sync", final_state={"ok": True}, mq=queue.Queue(), status="succeeded")

        def resume_run(self, **kwargs):
            calls["resume"] = kwargs
            return RunResult(run_id="run-resume", final_state={"resumed": True}, mq=queue.Queue(), status="succeeded")

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )

    out = asyncio.run(
        rt.resume_run(
            run_id="run-1",
            suspended_node_id="node-x",
            suspended_token_id="tok-x",
            client_result=RunSuccess(conversation_node_id=None, state_update=[]),
            workflow_id="wf-1",
            conversation_id="c-1",
            turn_node_id="t-1",
        )
    )
    assert out.status == "succeeded"
    assert calls["resume"]["suspended_node_id"] == "node-x"
    assert calls["resume"]["suspended_token_id"] == "tok-x"


def test_async_runtime_run_with_resume_markers_delegates_to_sync_run(monkeypatch):
    calls = {}

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            calls["run"] = kwargs
            return RunResult(run_id="run-sync", final_state={"ok": True}, mq=queue.Queue(), status="succeeded")

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-1",
            conversation_id="c-1",
            turn_node_id="t-1",
            initial_state={},
            _resume_step_seq=7,
            _resume_last_exec_node=object(),
        )
    )
    assert out.status == "succeeded"
    assert calls["run"]["_resume_step_seq"] == 7
    assert calls["run"]["_resume_last_exec_node"] is not None


def test_async_runtime_native_scheduler_persists_cancelled_terminal(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.persisted = []

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def _persist_cancelled_terminal(self, **kwargs):
            self.persisted.append(kwargs)

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        end = _Node("end", "end", terminal=True, fanout=False)
        return start, {"start": start, "end": end}, {"start": [_Edge("end")], "end": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    cancel_requested = {"flag": True}

    def _cancel(_run_id: str) -> bool:
        return cancel_requested["flag"]

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
        cancel_requested=_cancel,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-cancel-persist",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "cancelled"
    assert getattr(rt.sync_runtime, "persisted", [])


def test_async_runtime_native_scheduler_enforces_max_concurrent_tasks(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return RunResult(run_id="sync", final_state=dict(kwargs.get("initial_state") or {}), mq=queue.Queue(), status="succeeded")

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()
    active = {"n": 0}
    max_seen = {"n": 0}

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        active["n"] += 1
        max_seen["n"] = max(max_seen["n"], active["n"])
        try:
            await asyncio.sleep(0.08)
            return RunSuccess(conversation_node_id=None, state_update=[("a", {"seen": "a"})])
        finally:
            active["n"] -= 1

    @resolver.register("b")
    async def _b(_ctx):
        active["n"] += 1
        max_seen["n"] = max(max_seen["n"], active["n"])
        try:
            await asyncio.sleep(0.08)
            return RunSuccess(conversation_node_id=None, state_update=[("a", {"seen": "b"})])
        finally:
            active["n"] -= 1

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
        max_workers=1,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf-native-max-workers",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "succeeded"
    assert max_seen["n"] == 1


def test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order(monkeypatch):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return RunResult(run_id="sync", final_state=dict(kwargs.get("initial_state") or {}), mq=queue.Queue(), status="succeeded")

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=True)
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return start, {"start": start, "a": a, "b": b}, {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    delay = {"a": 0.08, "b": 0.01}

    @resolver.register("a")
    async def _a(_ctx):
        await asyncio.sleep(delay["a"])
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"order": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        await asyncio.sleep(delay["b"])
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"order": "b"})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    out1 = asyncio.run(
        rt.run(
            workflow_id="wf-native-order",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    delay["a"], delay["b"] = delay["b"], delay["a"]
    out2 = asyncio.run(
        rt.run(
            workflow_id="wf-native-order",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out1.status == out2.status == "succeeded"
    assert out1.final_state == out2.final_state
