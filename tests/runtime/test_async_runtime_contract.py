from __future__ import annotations

import asyncio
import copy
import time

import pytest

from kogwistar.runtime import (
    AsyncMappingStepResolver,
    AsyncWorkflowRuntime,
    MappingStepResolver,
)
from kogwistar.runtime.async_runtime import _as_sync_step_fn, _SyncResolverAdapter
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.runtime import (
    RunResult,
    StepContext,
    apply_state_update_inplace,
)
import queue

pytestmark = [pytest.mark.ci, pytest.mark.runtime, pytest.mark.runtime_async]


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


def test_async_resolver_runs_sync_handlers_inline_without_to_thread(monkeypatch):
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("slow")
    def _slow(_ctx):
        time.sleep(0.20)
        return RunSuccess(conversation_node_id=None, state_update=[])

    def _boom(*args, **kwargs):
        raise AssertionError("async resolver must not call to_thread for sync handlers")

    monkeypatch.setattr(asyncio, "to_thread", _boom)

    async def _run_two():
        fn = async_resolver.resolve_async("slow")
        t0 = time.perf_counter()
        r1, r2 = await asyncio.gather(fn(_DummyCtx()), fn(_DummyCtx()))
        dt = time.perf_counter() - t0
        return r1, r2, dt

    r1, r2, dt = asyncio.run(_run_two())
    assert isinstance(r1, RunSuccess)
    assert isinstance(r2, RunSuccess)
    assert dt >= 0.35


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


def test_async_runtime_native_scheduler_sync_handlers_run_inline_without_to_thread(monkeypatch):
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    root = Path(tempfile.gettempdir()) / f"phase3_inline_sync_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(root / "wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(root / "conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    workflow_id = "wf-inline-sync"
    start = _wf_node(workflow_id=workflow_id, node_id="wf|start", op="start", start=True)
    leaf = _wf_node(workflow_id=workflow_id, node_id="wf|leaf", op="leaf", terminal=True)
    workflow_engine.write.add_node(start)
    workflow_engine.write.add_node(leaf)
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id="wf|start->leaf",
            src=start.safe_get_id(),
            dst=leaf.safe_get_id(),
        )
    )

    def _boom(*args, **kwargs):
        raise AssertionError("async runtime must not call to_thread for sync handlers")

    monkeypatch.setattr(asyncio, "to_thread", _boom)

    rt = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=lambda op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[("u", {"last_op": op})])),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )

    out = asyncio.run(
        rt.run(
            workflow_id=workflow_id,
            conversation_id="conv-inline",
            turn_node_id="turn-1",
            initial_state={},
            run_id="run-inline",
        )
    )
    assert out.status == "succeeded"
    assert out.final_state["last_op"] == "leaf"


def test_async_runtime_rejects_disabled_native_scheduler():
    with pytest.raises(ValueError, match="native async scheduler"):
        AsyncWorkflowRuntime(
            workflow_engine=object(),
            conversation_engine=object(),
            step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
            predicate_registry={},
            trace=False,
            experimental_native_scheduler=False,
        )


def test_async_runtime_run_sync_is_not_supported():
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )

    with pytest.raises(NotImplementedError, match="run_sync"):
        rt.run_sync(
            workflow_id="wf-1",
            conversation_id="c-1",
            turn_node_id="t-1",
            initial_state={},
        )


def test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter():
    resolver = MappingStepResolver()
    resolver.nested_ops.add("answer")
    resolver.set_state_schema({"events": "a"})

    adapter = _SyncResolverAdapter(resolver)
    assert "answer" in adapter.nested_ops
    assert adapter._state_schema == {"events": "a"}


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


def test_async_runtime_native_scheduler_linear_success(monkeypatch):
    """Sync mirror: `tests/workflow/test_workflow_join.py` linear smoke path."""
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
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"path": "linear"})])

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


def test_async_runtime_native_scheduler_uses_shared_state_merge_semantics(monkeypatch):
    """Sync mirror: `tests/runtime/test_checkpoint_resume_contract.py` state merge reducer; also `tests/workflow/test_workflow_native_update.py` reducer parity."""
    resolver = AsyncMappingStepResolver()
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )

    sync_state = {
        "plain": "seed",
        "items": ["a"],
        "nested": {"keep": True},
    }
    async_state = copy.deepcopy(sync_state)
    state_update = [
        ("u", {"plain": "step-1"}),
        ("a", {"items": "b"}),
        ("e", {"items": ["c", "d"]}),
    ]

    rt.sync_runtime.apply_state_update(sync_state, state_update)
    apply_state_update_inplace(
        async_state,
        state_update,
        state_schema=getattr(resolver, "_state_schema", None),
    )

    assert sync_state == async_state


def test_async_runtime_native_scheduler_fanout_appends(monkeypatch):
    """Sync mirror: fanout expansion in `tests/workflow/test_workflow_join.py`."""
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
    """Sync mirror: `tests/runtime/test_workflow_cancel_event_sourced.py::test_runtime_event_sourced_cancel_reconciles_and_replay_is_stable`."""
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

    cancel_requested = {"flag": 0}

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
    """Sync mirror: `tests/runtime/test_workflow_invocation_and_route_next.py` route-next priority; also `tests/workflow/test_workflow_join.py`."""
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
            _route_next=["to_high"],
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
    """Sync mirror: token nesting / join ancestry in `tests/workflow/test_workflow_join.py`."""
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
    """Sync mirror: `tests/workflow/test_workflow_join.py` join-barrier merge."""
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


def test_async_runtime_native_scheduler_fanout_merge_join_does_not_deadlock(
    monkeypatch,
):
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

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
            self.metadata = {
                "wf_terminal": terminal,
                "wf_fanout": fanout,
                "wf_join": wf_join,
            }

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
        start = _Node("start", "start", fanout=False)
        fork = _Node("fork", "fork", fanout=True)
        a = _Node("a", "a")
        b = _Node("b", "b")
        join = _Node("join", "join", wf_join=True)
        end = _Node("end", "end", terminal=True)
        return (
            start,
            {
                "start": start,
                "fork": fork,
                "a": a,
                "b": b,
                "join": join,
                "end": end,
            },
            {
                "start": [_Edge("fork")],
                "fork": [_Edge("a"), _Edge("b")],
                "a": [_Edge("join")],
                "b": [_Edge("join")],
                "join": [_Edge("end")],
                "end": [],
            },
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("fork")
    async def _fork(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        await asyncio.sleep(0)
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "a"})])

    @resolver.register("b")
    async def _b(_ctx):
        await asyncio.sleep(0)
        return RunSuccess(conversation_node_id=None, state_update=[("a", {"parts": "b"})])

    @resolver.register("join")
    async def _join(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[
                ("u", {"join_calls": int(ctx.state_view.get("join_calls", 0)) + 1})
            ],
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
    )

    async def _run():
        return await asyncio.wait_for(
            rt.run(
                workflow_id="wf-native-join-no-deadlock",
                conversation_id="c1",
                turn_node_id="t1",
                initial_state={},
                run_id="run-native-join-no-deadlock",
            ),
            timeout=2.0,
        )

    out = asyncio.run(_run())
    assert out.status == "succeeded"
    assert sorted(out.final_state.get("parts", [])) == ["a", "b"]
    assert out.final_state["join_calls"] == 1
    assert out.final_state["done"] is True


def test_async_runtime_native_scheduler_without_join_executes_once_per_token(monkeypatch):
    """Sync mirror: single-token execution path in `tests/workflow/test_workflow_join.py`."""
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
    """Sync mirror: multiplicity rules in `tests/workflow/test_workflow_join.py`."""
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
    """Sync mirror: `tests/runtime/test_checkpoint_resume_contract.py` checkpoint frontier shape; also `tests/runtime/test_workflow_suspend_resume.py` resume frontier shape."""
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


@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_uses_step_uow_boundary(monkeypatch):
    active = {"depth": 0, "enters": 0}
    seen: list[str] = []

    class _FakeUOW:
        def __enter__(self):
            active["depth"] += 1
            active["enters"] += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            active["depth"] -= 1
            return False

    class _ConversationEngine:
        backend_kind = "memory"
        write = type(
            "_Write",
            (),
            {
                "add_node": lambda self, *args, **kwargs: None,
                "add_edge": lambda self, *args, **kwargs: None,
            },
        )()

        def uow(self):
            return _FakeUOW()

    class _WorkflowEngine:
        backend_kind = "memory"

    class _Node:
        def __init__(self, nid, op, terminal=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True)
        return start, {"start": start}, {"start": []}

    async def _start(ctx):
        seen.append(ctx.workflow_node_id)
        assert active["depth"] == 1
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"step_uow_seen": True})],
        )

    class _Resolver:
        nested_ops = set()
        _state_schema = {}

        def __call__(self, op):
            assert op == "start"
            return _start

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    rt = AsyncWorkflowRuntime(
        workflow_engine=_WorkflowEngine(),
        conversation_engine=_ConversationEngine(),
        step_resolver=_Resolver(),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
        transaction_mode="step",
    )

    out = await rt.run(
        workflow_id="wf-step-uow",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={},
    )

    assert out.status == "succeeded"
    assert seen == ["start"]
    assert active["enters"] == 1
    assert active["depth"] == 0


def test_async_runtime_native_scheduler_persists_pending_token_parent_links_on_cancel(monkeypatch):
    """Sync mirror: cancel checkpoint ancestry in `tests/runtime/test_workflow_cancel_event_sourced.py`."""
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
    cancel_requested = {"flag": False}

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    def _cancel(_run_id: str) -> bool:
        cancel_requested["flag"] = cancel_requested["flag"] + 1
        return cancel_requested["flag"] >= 3

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
            workflow_id="wf-native-pending-links",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    rt_join = out.final_state.get("_rt_join") or {}
    pending = list(rt_join.get("pending") or [])
    assert out.status == "cancelled"
    assert len(pending) == 2
    assert pending[0][0] == "a"
    assert pending[0][3] is None
    assert pending[1][0] == "b"
    assert pending[1][3] == pending[0][2]


@pytest.mark.parametrize("terminal_case", ["success", "failure"])
def test_async_runtime_side_by_side_node_edge_and_terminal_parity(terminal_case):
    """Sync mirror: node/edge parity in `tests/workflow/test_workflow_join.py` and `tests/workflow/test_workflow_native_update.py`."""
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import WorkflowCompletedNode, WorkflowEdge, WorkflowFailedNode, WorkflowNode
    from kogwistar.runtime.runtime import WorkflowRuntime
    from kogwistar.typing_interfaces import EmbeddingFunctionLike
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    def _norm_nodes(conv_engine, run_id: str):
        nodes = conv_engine.get_nodes(
            where={"$and": [{"run_id": run_id}]},
            limit=10_000,
        )
        out = []
        for n in nodes:
            md = dict(getattr(n, "metadata", {}) or {})
            et = md.get("entity_type")
            if et not in {
                "workflow_run",
                "workflow_step_exec",
                "workflow_checkpoint",
                "workflow_completed",
                "workflow_failed",
                "workflow_cancelled",
            }:
                continue
            md.pop("created_at_ms", None)
            md.pop("duration_ms", None)
            md.pop("summary", None)
            md.pop("mentions", None)
            out.append((str(getattr(n, "id", "")), et, md))
        return sorted(out)

    def _norm_edges(conv_engine, run_id: str):
        edges = conv_engine.get_edges(limit=10_000)
        out = []
        for e in edges:
            md = dict(getattr(e, "metadata", {}) or {})
            if md.get("run_id") != run_id:
                continue
            md.pop("created_at_ms", None)
            out.append(
                (
                    str(getattr(e, "id", "")),
                    str(getattr(e, "relation", "")),
                    tuple(getattr(e, "source_ids", []) or []),
                    tuple(getattr(e, "target_ids", []) or []),
                    md,
                )
            )
        return sorted(out)

    root = Path(tempfile.gettempdir()) / f"phase4_side_by_side_{terminal_case}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    sync_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    sync_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "async_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "async_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    workflow_id = f"wf-parity-{terminal_case}"
    start = _wf_node(workflow_id=workflow_id, node_id="wf|start", op="start", start=True)
    leaf = _wf_node(
        workflow_id=workflow_id,
        node_id="wf|leaf",
        op="leaf",
        terminal=True,
    )
    for eng in (sync_wf, async_wf):
        eng.write.add_node(start)
        eng.write.add_node(leaf)
        eng.write.add_edge(
            _wf_edge(
                workflow_id=workflow_id,
                edge_id="wf|start->leaf",
                src=start.safe_get_id(),
                dst=leaf.safe_get_id(),
            )
        )

    def _resolve(op: str):
        def _fn(_ctx):
            if terminal_case == "failure":
                return RunFailure(
                    conversation_node_id=None,
                    state_update=[("u", {"last_op": op})],
                    errors=["boom"],
                )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"last_op": op})],
            )

        return _fn

    sync_runtime = WorkflowRuntime(
        workflow_engine=sync_wf,
        conversation_engine=sync_conv,
        step_resolver=_resolve,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    async_runtime = AsyncWorkflowRuntime(
        workflow_engine=async_wf,
        conversation_engine=async_conv,
        step_resolver=_resolve,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    run_id = f"run-parity-{terminal_case}"
    sync_out = sync_runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv-parity",
        turn_node_id="turn-1",
        initial_state={},
        run_id=run_id,
    )
    async_out = asyncio.run(
        async_runtime.run(
            workflow_id=workflow_id,
            conversation_id="conv-parity",
            turn_node_id="turn-1",
            initial_state={},
            run_id=run_id,
        )
    )

    assert sync_out.status == async_out.status
    assert sync_out.final_state == async_out.final_state
    assert _norm_nodes(sync_conv, run_id) == _norm_nodes(async_conv, run_id)
    assert _norm_edges(sync_conv, run_id) == _norm_edges(async_conv, run_id)

    terminal_kind = "workflow_completed" if terminal_case == "success" else "workflow_failed"
    sync_terminal = sync_conv.get_nodes(
        where={"$and": [{"entity_type": terminal_kind}, {"run_id": run_id}]},
        limit=10,
    )
    async_terminal = async_conv.get_nodes(
        where={"$and": [{"entity_type": terminal_kind}, {"run_id": run_id}]},
        limit=10,
    )
    assert len(sync_terminal) == len(async_terminal) == 1
    assert isinstance(sync_terminal[0], WorkflowCompletedNode if terminal_case == "success" else WorkflowFailedNode)
    assert isinstance(async_terminal[0], WorkflowCompletedNode if terminal_case == "success" else WorkflowFailedNode)


@pytest.mark.parametrize("client_status", ["success", "failure"])
def test_async_runtime_suspend_and_resume_roundtrip(client_status):
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py` roundtrip cases."""
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import RunFailure, RunSuspended, RunSuccess, WorkflowEdge, WorkflowNode
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    root = Path(tempfile.gettempdir()) / f"phase4_suspend_resume_{client_status}_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(root / "wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(root / "conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    workflow_id = f"wf-suspend-{client_status}"
    workflow_engine.write.add_node(_wf_node(workflow_id=workflow_id, node_id="wf|start", op="start", start=True))
    workflow_engine.write.add_node(_wf_node(workflow_id=workflow_id, node_id="wf|gate", op="gate"))
    workflow_engine.write.add_node(_wf_node(workflow_id=workflow_id, node_id="wf|end", op="end", terminal=True))
    workflow_engine.write.add_edge(_wf_edge(workflow_id=workflow_id, edge_id="wf|start->gate", src="wf|start", dst="wf|gate"))
    workflow_engine.write.add_edge(_wf_edge(workflow_id=workflow_id, edge_id="wf|gate->end", src="wf|gate", dst="wf|end"))

    rt = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=lambda op: (
            lambda _ctx: (
                RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})])
                if op == "start"
                else RunSuspended(
                    conversation_node_id=None,
                    state_update=[],
                    wait_reason="await_client",
                    resume_payload={"kind": "need_client"},
                )
                if op == "gate"
                else RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])
            )
        ),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )

    run_id = f"run-suspend-{client_status}"
    out1 = asyncio.run(
        rt.run(
            workflow_id=workflow_id,
            conversation_id="conv-suspend",
            turn_node_id="turn-1",
            initial_state={},
            run_id=run_id,
        )
    )
    assert out1.status == "suspended"
    assert out1.final_state.get("started") is True
    suspended = (out1.final_state.get("_rt_join", {}) or {}).get("suspended", [])
    assert len(suspended) == 1
    suspended_token_id = suspended[0][2]

    client_result = (
        RunFailure(conversation_node_id=None, state_update=[], errors=["client fail"])
        if client_status == "failure"
        else RunSuccess(conversation_node_id=None, state_update=[("u", {"resumed": True})])
    )
    with pytest.raises(NotImplementedError, match="resume_run"):
        asyncio.run(
            rt.resume_run(
                run_id=run_id,
                suspended_node_id="wf|gate",
                suspended_token_id=suspended_token_id,
                client_result=client_result,
                workflow_id=workflow_id,
                conversation_id="conv-suspend",
                turn_node_id="turn-1",
            )
        )


def test_async_runtime_nested_workflow_invocation_matches_sync():
    """Sync mirror: `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_synthesized_design_is_persisted_and_used`."""
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowInvocationRequest, WorkflowNode
    from kogwistar.runtime.runtime import WorkflowRuntime
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    root = Path(tempfile.gettempdir()) / f"phase4_nested_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    sync_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    sync_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "async_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "async_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    parent_wf = "wf-parent"
    child_wf = "wf-child"
    for eng in (sync_wf, async_wf):
        eng.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True))
        eng.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True))
        eng.write.add_edge(_wf_edge(workflow_id=parent_wf, edge_id="p|start->end", src="p|start", dst="p|end"))
        eng.write.add_node(_wf_node(workflow_id=child_wf, node_id="c|start", op="child", start=True, terminal=True))

    def _resolver(op: str):
        def _fn(_ctx):
            if op == "spawn":
                return RunSuccess(
                    conversation_node_id=None,
                    state_update=[("u", {"parent_seen": True})],
                    workflow_invocations=[
                        WorkflowInvocationRequest(
                            workflow_id=child_wf,
                            result_state_key="child_result",
                            run_id="child-run",
                        )
                    ],
                )
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"child_value": 7})],
            )

        return _fn

    sync_rt = WorkflowRuntime(
        workflow_engine=sync_wf,
        conversation_engine=sync_conv,
        step_resolver=_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    async_rt = AsyncWorkflowRuntime(
        workflow_engine=async_wf,
        conversation_engine=async_conv,
        step_resolver=_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    run_id = "run-nested"
    sync_out = sync_rt.run(
        workflow_id=parent_wf,
        conversation_id="conv-nested",
        turn_node_id="turn-1",
        initial_state={},
        run_id=run_id,
    )
    async_out = asyncio.run(
        async_rt.run(
            workflow_id=parent_wf,
            conversation_id="conv-nested",
            turn_node_id="turn-1",
            initial_state={},
            run_id=run_id,
        )
    )
    assert sync_out.status == async_out.status == "succeeded"
    assert sync_out.final_state == async_out.final_state
    assert sync_out.final_state["parent_seen"] is True
    assert sync_out.final_state["child_result"]["child_value"] == 7


def test_async_runtime_nested_workflow_child_failure_fails_parent():
    """Sync mirror: `tests/runtime/test_workflow_invocation_and_route_next.py` nested workflow failure semantics; also `tests/runtime/test_workflow_cancel_event_sourced.py`."""
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import RunFailure, RunSuccess, WorkflowEdge, WorkflowInvocationRequest, WorkflowNode
    from kogwistar.runtime.runtime import WorkflowRuntime
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    root = Path(tempfile.gettempdir()) / f"phase4_nested_fail_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    sync_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    sync_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "sync_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "async_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "async_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    parent_wf = "wf-parent"
    child_wf = "wf-child"
    for eng in (sync_wf, async_wf):
        eng.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True))
        eng.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True))
        eng.write.add_edge(_wf_edge(workflow_id=parent_wf, edge_id="p|start->end", src="p|start", dst="p|end"))
        eng.write.add_node(_wf_node(workflow_id=child_wf, node_id="c|start", op="child", start=True, terminal=True))

    def _resolver(op: str):
        def _fn(_ctx):
            if op == "spawn":
                return RunSuccess(
                    conversation_node_id=None,
                    state_update=[("u", {"parent_seen": True})],
                    workflow_invocations=[
                        WorkflowInvocationRequest(
                            workflow_id=child_wf,
                            result_state_key="child_result",
                            run_id="child-run",
                        )
                    ],
                )
            return RunFailure(
                conversation_node_id=None,
                status="failure",
                errors=["child boom"],
                state_update=[],
            )

        return _fn

    sync_rt = WorkflowRuntime(
        workflow_engine=sync_wf,
        conversation_engine=sync_conv,
        step_resolver=_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
    )
    async_rt = AsyncWorkflowRuntime(
        workflow_engine=async_wf,
        conversation_engine=async_conv,
        step_resolver=_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )
    run_id = "run-nested-fail"
    sync_out = sync_rt.run(
        workflow_id=parent_wf,
        conversation_id="conv-nested",
        turn_node_id="turn-1",
        initial_state={},
        run_id=run_id,
    )
    async_out = asyncio.run(
        async_rt.run(
            workflow_id=parent_wf,
            conversation_id="conv-nested",
            turn_node_id="turn-1",
            initial_state={},
            run_id=run_id,
        )
    )
    assert sync_out.status == async_out.status == "failure"
    assert sync_out.final_state == async_out.final_state


def test_async_runtime_parent_cancellation_propagates_to_child():
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py` cancellation propagation; also `tests/runtime/test_workflow_cancel_event_sourced.py`."""
    from pathlib import Path
    import tempfile
    import uuid

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowInvocationRequest, WorkflowNode
    from tests.conftest import FakeEmbeddingFunction
    from tests._helpers.fake_backend import build_fake_backend

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(*, workflow_id: str, node_id: str, op: str, start: bool = False, terminal: bool = False) -> WorkflowNode:
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _wf_edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
        return WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=workflow_id,
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
        )

    root = Path(tempfile.gettempdir()) / f"phase4_parent_cancel_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    ef = FakeEmbeddingFunction(dim=3)
    async_wf = GraphKnowledgeEngine(
        persist_directory=str(root / "async_wf"),
        kg_graph_type="workflow",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )
    async_conv = GraphKnowledgeEngine(
        persist_directory=str(root / "async_conv"),
        kg_graph_type="conversation",
        embedding_function=ef,
        backend_factory=build_fake_backend,
    )

    parent_wf = "wf-parent"
    child_wf = "wf-child"
    async_wf.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True))
    async_wf.write.add_node(_wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True))
    async_wf.write.add_edge(_wf_edge(workflow_id=parent_wf, edge_id="p|start->end", src="p|start", dst="p|end"))
    async_wf.write.add_node(_wf_node(workflow_id=child_wf, node_id="c|start", op="child", start=True))
    async_wf.write.add_node(_wf_node(workflow_id=child_wf, node_id="c|end", op="child-end", terminal=True))
    async_wf.write.add_edge(_wf_edge(workflow_id=child_wf, edge_id="c|start->end", src="c|start", dst="c|end"))

    child_started = asyncio.Event()
    cancel_flags = {"parent-run": False}
    seen_runs: list[str] = []

    def _cancel(run_id: str) -> bool:
        seen_runs.append(str(run_id))
        return bool(cancel_flags.get(str(run_id), False))

    def _resolver(op: str):
        async def _spawn(_ctx):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[],
                workflow_invocations=[
                    WorkflowInvocationRequest(
                        workflow_id=child_wf,
                        result_state_key="child_result",
                        run_id="child-run",
                    )
                ],
            )

        async def _child(_ctx):
            child_started.set()
            await asyncio.sleep(0.15)
            return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_started": True})])

        async def _child_end(_ctx):
            return RunSuccess(conversation_node_id=None, state_update=[("u", {"child_ended": True})])

        return {"spawn": _spawn, "child": _child, "child-end": _child_end}[op]

    rt = AsyncWorkflowRuntime(
        workflow_engine=async_wf,
        conversation_engine=async_conv,
        step_resolver=_resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
        cancel_requested=_cancel,
    )

    async def _run():
        async def _flip():
            await child_started.wait()
            await asyncio.sleep(0.02)
            cancel_flags["parent-run"] = True

        flipper = asyncio.create_task(_flip())
        try:
            return await rt.run(
                workflow_id=parent_wf,
                conversation_id="conv-parent-cancel",
                turn_node_id="turn-1",
                initial_state={},
                run_id="parent-run",
            )
        finally:
            await flipper

    out = asyncio.run(_run())
    assert out.status == "cancelled"
    assert "parent-run" in seen_runs
    assert "child-run" in seen_runs


def test_async_runtime_native_scheduler_cancel_idempotent_terminal_persistence(monkeypatch):
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

    cancel_requested = {"flag": True, "checks": 0}

    def _cancel(_run_id: str) -> bool:
        cancel_requested["checks"] += 1
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
            workflow_id="wf-native-cancel-idempotent",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
        )
    )
    assert out.status == "cancelled"
    assert cancel_requested["checks"] >= 1
    assert len(getattr(rt.sync_runtime, "persisted", [])) == 1


def test_async_runtime_resume_run_is_not_supported():
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )

    with pytest.raises(NotImplementedError, match="resume_run"):
        asyncio.run(
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


def test_async_runtime_run_with_resume_markers_is_not_supported():
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )

    with pytest.raises(NotImplementedError, match="resume-marker"):
        asyncio.run(
            rt.run(
                workflow_id="wf-1",
                conversation_id="c-1",
                turn_node_id="t-1",
                initial_state={},
                _resume_step_seq=7,
                _resume_last_exec_node=object(),
            )
        )


def test_async_runtime_native_scheduler_persists_cancelled_terminal(monkeypatch):
    """Sync mirror: cancelled-terminal persistence in `tests/runtime/test_workflow_cancel_event_sourced.py`."""
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
    """Sync mirror: worker bound / concurrency guard in sync `WorkflowRuntime`."""
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
    """Sync mirror: ordered step completion in sync workflow step dispatch."""
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


@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata(monkeypatch):
    """Sync mirror: `tests/runtime/test_trace_sink_parallel_nested_minimal.py` trace event shape; also `tests/workflow/test_tracing_e2e.py`."""
    emitted: list[tuple[str, dict[str, object]]] = []

    class _FakeEmitter:
        sink = None

        def step_started(self, ctx):
            emitted.append(("started", ctx.as_fields()))

        def step_completed(self, ctx, *, status, duration_ms, extra=None):
            payload = dict(ctx.as_fields())
            payload["status"] = status
            payload["duration_ms"] = duration_ms
            emitted.append(("completed", payload))

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.emitter = kwargs["events"]

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def _should_step_uow(self, *args, **kwargs):
            return False

        def _maybe_step_uow(self):
            from contextlib import nullcontext

            return nullcontext()

        def _persist_step_exec(self, **kwargs):
            return object()

        def _persist_checkpoint(self, **kwargs):
            return None

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"path": "trace"})])

    fake_emitter = _FakeEmitter()
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=True,
        events=fake_emitter,
        experimental_native_scheduler=True,
    )
    out = await rt.run(
        workflow_id="wf-native-trace",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={},
    )

    assert out.status == "succeeded"
    assert out.final_state["path"] == "trace"
    assert [kind for kind, _payload in emitted] == ["started", "completed"]
    started = emitted[0][1]
    completed = emitted[1][1]
    assert started["run_id"] == completed["run_id"]
    assert started["node_id"] == "start"
    assert started["trace_id"] == started["run_id"]
    assert completed["status"] == "ok"
    assert int(completed["duration_ms"]) >= 0


@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort(monkeypatch):
    class _FailingEmitter:
        sink = None

        def step_started(self, ctx):
            raise RuntimeError("trace boom")

        def step_completed(self, ctx, *, status, duration_ms, extra=None):
            raise RuntimeError("trace boom")

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.emitter = kwargs["events"]

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def _should_step_uow(self, *args, **kwargs):
            return False

        def _maybe_step_uow(self):
            from contextlib import nullcontext

            return nullcontext()

        def _persist_step_exec(self, **kwargs):
            return object()

        def _persist_checkpoint(self, **kwargs):
            return None

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ok": True})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=True,
        events=_FailingEmitter(),
        experimental_native_scheduler=True,
    )
    out = await rt.run(
        workflow_id="wf-native-trace-failure",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={},
    )

    assert out.status == "succeeded"
    assert out.final_state["ok"] is True


@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_nested_invocation_reuses_trace_emitter(monkeypatch):
    seen: list[tuple[str, bool]] = []

    class _FakeEmitter:
        sink = None

        def step_started(self, ctx):
            pass

        def step_completed(self, ctx, *, status, duration_ms, extra=None):
            pass

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.emitter = kwargs["events"]

        def run(self, **kwargs):
            return RunResult(
                run_id=str(kwargs.get("run_id") or "sync-run"),
                final_state=dict(kwargs.get("initial_state") or {}),
                mq=queue.Queue(),
                status="succeeded",
            )

        def _should_step_uow(self, *args, **kwargs):
            return False

        def _maybe_step_uow(self):
            from contextlib import nullcontext

            return nullcontext()

        def _persist_step_exec(self, **kwargs):
            return object()

        def _persist_checkpoint(self, **kwargs):
            return None

        def _child_workflow_initial_state(self, *, parent_state, invocation):
            return dict(parent_state)

        def _apply_workflow_invocation_result(self, *, state, invocation, child_result):
            state[str(getattr(invocation, "result_state_key", "child_result"))] = dict(
                getattr(child_result, "final_state", {}) or {}
            )

    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        if workflow_id == "wf-parent":
            start = _Node("start", "spawn", terminal=False, fanout=False)
            end = _Node("end", "end", terminal=True, fanout=False)
            edge = type("_Edge", (), {"target_ids": ["end"], "metadata": {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}})()
            return start, {"start": start, "end": end}, {"start": [edge], "end": []}
        child = _Node("child-start", "child", terminal=True, fanout=False)
        return child, {"child-start": child}, {"child-start": []}

    monkeypatch.setattr("kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime)
    monkeypatch.setattr("kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate)

    resolver = AsyncMappingStepResolver()
    fake_emitter = _FakeEmitter()
    from kogwistar.runtime.models import WorkflowInvocationRequest

    @resolver.register("spawn")
    async def _spawn(ctx):
        seen.append(("spawn", ctx.events is fake_emitter))
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            workflow_invocations=[
                WorkflowInvocationRequest(
                    workflow_id="wf-child",
                    result_state_key="child_result",
                    run_id="child-run",
                )
            ],
        )

    @resolver.register("child")
    async def _child(ctx):
        seen.append(("child", ctx.events is fake_emitter))
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"child": True})])

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"done": True})])

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=True,
        events=fake_emitter,
        experimental_native_scheduler=True,
    )
    out = await rt.run(
        workflow_id="wf-parent",
        conversation_id="c1",
        turn_node_id="t1",
        initial_state={},
    )

    assert out.status == "succeeded"
    assert out.final_state["done"] is True
    assert seen == [("spawn", True), ("child", True)]
