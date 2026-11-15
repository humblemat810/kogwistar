from __future__ import annotations

import asyncio
import copy
import queue
import time

import pytest

from kogwistar.runtime import (
    AsyncMappingStepResolver,
    AsyncWorkflowRuntime,
    MappingStepResolver,
)
from kogwistar.runtime.async_runtime import _SyncResolverAdapter
from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended
from kogwistar.runtime.runtime import (
    RunResult,
    apply_state_update_inplace,
)
from kogwistar.runtime.telemetry import TraceContext

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


def test_async_runtime_native_scheduler_sync_handlers_run_inline_without_to_thread():
    import tempfile
    import threading
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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
    start = _wf_node(
        workflow_id=workflow_id, node_id="wf|start", op="start", start=True
    )
    leaf = _wf_node(
        workflow_id=workflow_id, node_id="wf|leaf", op="leaf", terminal=True
    )
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

    handler_threads: list[int] = []

    def _step(op: str):
        def _run(_ctx):
            handler_threads.append(threading.get_ident())
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"last_op": op})]
            )

        return _run

    rt = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=_step,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        experimental_native_scheduler=True,
    )

    caller_thread = threading.get_ident()
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
    assert handler_threads
    assert set(handler_threads) == {caller_thread}


def test_async_runtime_fences_concurrent_duplicate_run_ids(tmp_path):
    """Sync mirror: `tests/runtime/test_run_ownership.py::test_only_one_live_owner_can_claim_a_run`."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import WorkflowNode
    from kogwistar.runtime.runtime_checkpoints import WorkflowRunOwnershipError
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    workflow_id = "wf-async-run-owner"
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"),
        kg_graph_type="workflow",
        embedding_function=FakeEmbeddingFunction(dim=3),
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"),
        kg_graph_type="conversation",
        embedding_function=FakeEmbeddingFunction(dim=3),
        backend_factory=build_fake_backend,
    )
    workflow_engine.write.add_node(
        WorkflowNode(
            id="start",
            label="start",
            type="entity",
            doc_id="start",
            summary="start",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": "blocked",
                "wf_start": True,
                "wf_terminal": True,
                "wf_version": "v1",
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    invocations = 0

    async def _blocked(_ctx):
        nonlocal invocations
        invocations += 1
        entered.set()
        await release.wait()
        return RunSuccess(conversation_node_id=None, state_update=[])

    def _runtime():
        resolver = AsyncMappingStepResolver()
        resolver.register("blocked")(_blocked)
        return AsyncWorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
            trace=False,
            experimental_native_scheduler=True,
        )

    async def _exercise():
        first_runtime = _runtime()
        second_runtime = _runtime()
        first = asyncio.create_task(
            first_runtime.run(
                workflow_id=workflow_id,
                conversation_id="conversation-async-run-owner",
                turn_node_id="turn-1",
                initial_state={},
                run_id="run-async-owner-shared",
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(WorkflowRunOwnershipError):
            await second_runtime.run(
                workflow_id=workflow_id,
                conversation_id="conversation-async-run-owner",
                turn_node_id="turn-1",
                initial_state={},
                run_id="run-async-owner-shared",
            )
        release.set()
        return await first

    result = asyncio.run(_exercise())
    assert result.status == "succeeded"
    assert invocations == 1


def test_async_runtime_rejects_disabled_native_scheduler():
    with pytest.raises(ValueError, match="native async scheduler"):
        AsyncWorkflowRuntime(
            workflow_engine=object(),
            conversation_engine=object(),
            step_resolver=lambda _op: (
                lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])
            ),
            predicate_registry={},
            trace=False,
            experimental_native_scheduler=False,
        )


def test_async_runtime_run_sync_is_not_supported():
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (
            lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])
        ),
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
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=True, fanout=False)
        return start, {"start": start}, {"start": []}

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"path": "linear"})]
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
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
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

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"branches": "a"})]
        )

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"branches": "b"})]
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
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"picked": "low"})]
        )

    @resolver.register("high")
    async def _high(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"picked": "high"})]
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
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"parts": "a"})]
        )

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"parts": "b"})]
        )

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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"done": True})]
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
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"parts": "a"})]
        )

    @resolver.register("b")
    async def _b(_ctx):
        await asyncio.sleep(0)
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"parts": "b"})]
        )

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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"done": True})]
        )

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


def test_async_runtime_native_scheduler_without_join_executes_once_per_token(
    monkeypatch,
):
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
            {
                "start": [_Edge("a"), _Edge("b")],
                "a": [_Edge("c")],
                "b": [_Edge("c")],
                "c": [],
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
            {
                "start": [
                    _Edge("a", multiplicity="many"),
                    _Edge("b", multiplicity="many"),
                ],
                "a": [],
                "b": [],
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

    @resolver.register("a")
    async def _a(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"targets": "a"})]
        )

    @resolver.register("b")
    async def _b(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"targets": "b"})]
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
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
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
    assert set(rt_join.keys()) >= {
        "join_node_ids",
        "join_outstanding",
        "join_waiters",
        "pending",
        "suspended",
    }


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


def test_async_runtime_native_scheduler_persists_pending_token_parent_links_on_cancel(
    monkeypatch,
):
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
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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


@pytest.mark.ci_full
@pytest.mark.parametrize("terminal_case", ["success", "failure"])
def test_async_runtime_side_by_side_node_edge_and_terminal_parity(terminal_case):
    """Sync mirror: node/edge parity in `tests/workflow/test_workflow_join.py` and `tests/workflow/test_workflow_native_update.py`."""
    import tempfile
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import (
        WorkflowCompletedNode,
        WorkflowEdge,
        WorkflowFailedNode,
        WorkflowNode,
    )
    from kogwistar.runtime.runtime import WorkflowRuntime
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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
            # This index fingerprint includes the trace span, whose excerpt
            # contains the intentionally variable measured step duration.
            md.pop("node_refs_fp", None)
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

    root = (
        Path(tempfile.gettempdir())
        / f"phase4_side_by_side_{terminal_case}_{uuid.uuid4().hex}"
    )
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
    start = _wf_node(
        workflow_id=workflow_id, node_id="wf|start", op="start", start=True
    )
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
    assert sync_out.errors == async_out.errors
    assert sync_out.errors == (["boom"] if terminal_case == "failure" else [])
    assert sync_out.final_state == async_out.final_state
    sync_nodes = dict((item[0], item[2]) for item in _norm_nodes(sync_conv, run_id))
    async_nodes = dict((item[0], item[2]) for item in _norm_nodes(async_conv, run_id))
    node_diffs = {
        node_id: {
            key: (
                sync_nodes.get(node_id, {}).get(key),
                async_nodes.get(node_id, {}).get(key),
            )
            for key in set(sync_nodes.get(node_id, {}))
            | set(async_nodes.get(node_id, {}))
            if sync_nodes.get(node_id, {}).get(key)
            != async_nodes.get(node_id, {}).get(key)
        }
        for node_id in set(sync_nodes) | set(async_nodes)
        if sync_nodes.get(node_id) != async_nodes.get(node_id)
    }
    assert not node_diffs, f"sync/async node metadata differs: {node_diffs!r}"
    assert _norm_edges(sync_conv, run_id) == _norm_edges(async_conv, run_id)

    terminal_kind = (
        "workflow_completed" if terminal_case == "success" else "workflow_failed"
    )
    sync_terminal = sync_conv.get_nodes(
        where={"$and": [{"entity_type": terminal_kind}, {"run_id": run_id}]},
        limit=10,
    )
    async_terminal = async_conv.get_nodes(
        where={"$and": [{"entity_type": terminal_kind}, {"run_id": run_id}]},
        limit=10,
    )
    assert len(sync_terminal) == len(async_terminal) == 1
    assert isinstance(
        sync_terminal[0],
        WorkflowCompletedNode if terminal_case == "success" else WorkflowFailedNode,
    )
    assert isinstance(
        async_terminal[0],
        WorkflowCompletedNode if terminal_case == "success" else WorkflowFailedNode,
    )


@pytest.mark.ci_full
@pytest.mark.parametrize("client_status", ["success", "failure"])
def test_async_runtime_suspend_and_resume_roundtrip(client_status):
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py` roundtrip cases."""
    import tempfile
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import (
        RunFailure,
        RunSuccess,
        RunSuspended,
        WorkflowEdge,
        WorkflowNode,
    )
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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

    root = (
        Path(tempfile.gettempdir())
        / f"phase4_suspend_resume_{client_status}_{uuid.uuid4().hex}"
    )
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
    workflow_engine.write.add_node(
        _wf_node(workflow_id=workflow_id, node_id="wf|start", op="start", start=True)
    )
    workflow_engine.write.add_node(
        _wf_node(workflow_id=workflow_id, node_id="wf|gate", op="gate")
    )
    workflow_engine.write.add_node(
        _wf_node(workflow_id=workflow_id, node_id="wf|recover", op="recover")
    )
    workflow_engine.write.add_node(
        _wf_node(workflow_id=workflow_id, node_id="wf|end", op="end", terminal=True)
    )
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id="wf|start->gate",
            src="wf|start",
            dst="wf|gate",
        )
    )
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id, edge_id="wf|gate->end", src="wf|gate", dst="wf|end"
        )
    )
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id="wf|gate->recover",
            src="wf|gate",
            dst="wf|recover",
        )
    )
    workflow_engine.write.add_edge(
        _wf_edge(
            workflow_id=workflow_id,
            edge_id="wf|recover->end",
            src="wf|recover",
            dst="wf|end",
        )
    )

    rt = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=lambda op: (
            lambda _ctx: (
                RunSuccess(
                    conversation_node_id=None, state_update=[("u", {"started": True})]
                )
                if op == "start"
                else RunSuspended(
                    conversation_node_id=None,
                    state_update=[],
                    wait_reason="await_client",
                    resume_payload={"kind": "need_client"},
                )
                if op == "gate"
                else RunSuccess(
                    conversation_node_id=None,
                    state_update=[
                        (
                            "u",
                            {
                                "ended": True,
                                **({"recovered": True} if op == "recover" else {}),
                            },
                        )
                    ],
                )
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
        RunFailure(
            conversation_node_id=None,
            state_update=[],
            errors=["client fail"],
            next_step_names=["wf|recover"],
        )
        if client_status == "failure"
        else RunSuccess(
            conversation_node_id=None, state_update=[("u", {"resumed": True})]
        )
    )
    out2 = asyncio.run(
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
    if client_status == "failure":
        assert out2.status == "succeeded"
        assert out2.final_state.get("recovered") is True
        assert out2.final_state.get("ended") is True
    else:
        assert out2.status == "succeeded"
        assert out2.final_state.get("resumed") is True
        assert out2.final_state.get("ended") is True


@pytest.mark.ci_full
def test_async_runtime_nested_workflow_invocation_matches_sync():
    """Sync mirror: `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_synthesized_design_is_persisted_and_used`."""
    import tempfile
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import (
        RunSuccess,
        WorkflowEdge,
        WorkflowInvocationRequest,
        WorkflowNode,
    )
    from kogwistar.runtime.runtime import WorkflowRuntime
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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
        eng.write.add_node(
            _wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True)
        )
        eng.write.add_node(
            _wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True)
        )
        eng.write.add_edge(
            _wf_edge(
                workflow_id=parent_wf,
                edge_id="p|start->end",
                src="p|start",
                dst="p|end",
            )
        )
        eng.write.add_node(
            _wf_node(
                workflow_id=child_wf,
                node_id="c|start",
                op="child",
                start=True,
                terminal=True,
            )
        )

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
            _trace_context=TraceContext.new_root(
                run_id=run_id,
                token_id=run_id,
                step_seq=0,
                node_id="p|start",
            ),
        )
    )
    assert sync_out.status == async_out.status == "succeeded"
    assert sync_out.final_state == async_out.final_state
    assert sync_out.final_state["parent_seen"] is True
    assert sync_out.final_state["child_result"]["child_value"] == 7
    parent_node = async_conv.read.get_nodes(ids=[f"wf_run|{run_id}"], limit=1)[0]
    child_node = async_conv.read.get_nodes(ids=["wf_run|child-run"], limit=1)[0]
    assert parent_node.metadata["trace_id"] == child_node.metadata["trace_id"]
    assert parent_node.metadata["trace_id"] != run_id


@pytest.mark.ci_full
def test_async_runtime_nested_workflow_child_failure_fails_parent():
    """Sync mirror: `tests/runtime/test_workflow_invocation_and_route_next.py` nested workflow failure semantics; also `tests/runtime/test_workflow_cancel_event_sourced.py`."""
    import tempfile
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import (
        RunFailure,
        RunSuccess,
        WorkflowEdge,
        WorkflowInvocationRequest,
        WorkflowNode,
    )
    from kogwistar.runtime.runtime import WorkflowRuntime
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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
        eng.write.add_node(
            _wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True)
        )
        eng.write.add_node(
            _wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True)
        )
        eng.write.add_edge(
            _wf_edge(
                workflow_id=parent_wf,
                edge_id="p|start->end",
                src="p|start",
                dst="p|end",
            )
        )
        eng.write.add_node(
            _wf_node(
                workflow_id=child_wf,
                node_id="c|start",
                op="child",
                start=True,
                terminal=True,
            )
        )

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


@pytest.mark.ci_full
def test_async_runtime_parent_cancellation_propagates_to_child():
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py` cancellation propagation; also `tests/runtime/test_workflow_cancel_event_sourced.py`."""
    import tempfile
    import uuid
    from pathlib import Path

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import (
        RunSuccess,
        WorkflowEdge,
        WorkflowInvocationRequest,
        WorkflowNode,
    )
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _wf_node(
        *,
        workflow_id: str,
        node_id: str,
        op: str,
        start: bool = False,
        terminal: bool = False,
    ) -> WorkflowNode:
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
    async_wf.write.add_node(
        _wf_node(workflow_id=parent_wf, node_id="p|start", op="spawn", start=True)
    )
    async_wf.write.add_node(
        _wf_node(workflow_id=parent_wf, node_id="p|end", op="end", terminal=True)
    )
    async_wf.write.add_edge(
        _wf_edge(
            workflow_id=parent_wf, edge_id="p|start->end", src="p|start", dst="p|end"
        )
    )
    async_wf.write.add_node(
        _wf_node(workflow_id=child_wf, node_id="c|start", op="child", start=True)
    )
    async_wf.write.add_node(
        _wf_node(workflow_id=child_wf, node_id="c|end", op="child-end", terminal=True)
    )
    async_wf.write.add_edge(
        _wf_edge(
            workflow_id=child_wf, edge_id="c|start->end", src="c|start", dst="c|end"
        )
    )

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
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"child_started": True})]
            )

        async def _child_end(_ctx):
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"child_ended": True})]
            )

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


def test_async_runtime_native_scheduler_cancel_idempotent_terminal_persistence(
    monkeypatch,
):
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
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        end = _Node("end", "end", terminal=True, fanout=False)
        return start, {"start": start, "end": end}, {"start": [_Edge("end")], "end": []}

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


def test_async_runtime_resume_run_resuspends_same_token_then_continues(tmp_path):
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py::test_resume_run_can_resuspend_same_token_with_updated_payload`."""
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction

    def _grounding():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _node(node_id: str, op: str, *, start: bool = False, terminal: bool = False):
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[_grounding()],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": "wf-async-resume",
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

    def _edge(edge_id: str, source: str, target: str):
        return WorkflowEdge(
            id=edge_id,
            source_ids=[source],
            target_ids=[target],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id="wf-async-resume",
            mentions=[_grounding()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": "wf-async-resume",
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

    embedding = FakeEmbeddingFunction(dim=3)
    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "wf"),
        kg_graph_type="workflow",
        embedding_function=embedding,
        backend_factory=build_fake_backend,
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"),
        kg_graph_type="conversation",
        embedding_function=embedding,
        backend_factory=build_fake_backend,
    )
    for node in (
        _node("start", "start", start=True),
        _node("gate", "gate"),
        _node("end", "end", terminal=True),
    ):
        workflow_engine.write.add_node(node)
    workflow_engine.write.add_edge(_edge("start-gate", "start", "gate"))
    workflow_engine.write.add_edge(_edge("gate-end", "gate", "end"))

    seen: list[tuple[str, int]] = []
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(ctx):
        seen.append((ctx.workflow_node_id, ctx.step_seq))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("gate")
    async def _gate(ctx):
        seen.append((ctx.workflow_node_id, ctx.step_seq))
        return RunSuspended(
            conversation_node_id=None,
            resume_payload={"message": "first pause"},
        )

    @resolver.register("end")
    async def _end(ctx):
        seen.append((ctx.workflow_node_id, ctx.step_seq))
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ended": True})],
        )

    runtime = AsyncWorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=1,
        trace=False,
        experimental_native_scheduler=True,
    )
    run_id = "run-async-resume-contract"
    first = asyncio.run(
        runtime.run(
            workflow_id="wf-async-resume",
            conversation_id="conversation-async-resume",
            turn_node_id="turn-1",
            initial_state={"started": True},
            run_id=run_id,
        )
    )
    checkpoint = runtime.sync_runtime._latest_checkpoint_for_run(
        conversation_id="conversation-async-resume", run_id=run_id
    )
    from kogwistar.runtime.checkpoint_state import decode_checkpoint_state

    checkpoint_state = decode_checkpoint_state(
        checkpoint.metadata["state_json"], run_id=run_id
    )
    token_id = checkpoint_state["_rt_join"]["suspended"][0][2]

    second = asyncio.run(
        runtime.resume_run(
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=token_id,
            client_result=RunSuspended(
                conversation_node_id=None,
                state_update=[("u", {"retry_count": 1})],
                resume_payload={"message": "second pause"},
            ),
            workflow_id="wf-async-resume",
            conversation_id="conversation-async-resume",
            turn_node_id="turn-1",
        )
    )
    assert second.status == "suspended"
    assert second.final_state["retry_count"] == 1
    second_snapshot = second.final_state["_rt_join"]
    assert second_snapshot["suspended"] == [("gate", 0, token_id, None)]

    third = asyncio.run(
        runtime.resume_run(
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"approved": True})],
            ),
            workflow_id="wf-async-resume",
            conversation_id="conversation-async-resume",
            turn_node_id="turn-1",
        )
    )
    assert first.status == "suspended"
    assert third.status == "succeeded"
    assert third.final_state["retry_count"] == 1
    assert third.final_state["approved"] is True
    assert third.final_state["ended"] is True
    assert seen == [("start", 0), ("gate", 1), ("end", 4)]

    with pytest.raises(ValueError, match="Expected one suspended token"):
        asyncio.run(
            runtime.resume_run(
                run_id=run_id,
                suspended_node_id="gate",
                suspended_token_id=token_id,
                client_result=RunSuccess(
                    conversation_node_id=None,
                    state_update=[("u", {"duplicate_resume": True})],
                ),
                workflow_id="wf-async-resume",
                conversation_id="conversation-async-resume",
                turn_node_id="turn-1",
            )
        )

    steps = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
        limit=20,
    )
    by_seq = {int(step.metadata["step_seq"]): step for step in steps}
    assert sorted(by_seq) == [0, 1, 2, 3, 4]
    assert by_seq[2].metadata["op"] == "client_sandbox_resume"
    assert by_seq[2].metadata["status"] == "suspended"
    assert by_seq[3].metadata["op"] == "client_sandbox_resume"
    assert by_seq[3].metadata["status"] == "ok"
    assert by_seq[4].metadata["workflow_node_id"] == "end"
    continuation_edges = conversation_engine.read.get_edges(
        where={"$and": [{"run_id": run_id}, {"relation": "wf_next_step_exec"}]},
        limit=20,
    )
    assert any(
        edge.source_ids == [by_seq[3].safe_get_id()]
        and edge.target_ids == [by_seq[4].safe_get_id()]
        for edge in continuation_edges
    )


@pytest.mark.ci_full
@pytest.mark.asyncio
async def test_async_runtime_client_resume_roundtrip_postgres(request, tmp_path):
    """Persist a client suspension and resume continuation through PostgreSQL."""
    import uuid

    from sqlalchemy import text

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
    from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
    from tests.conftest import FakeEmbeddingFunction

    sa_engine = request.getfixturevalue("sa_engine")
    if sa_engine is None:
        pytest.skip("PostgreSQL fixture is unavailable")

    schema_base = f"async_client_resume_{uuid.uuid4().hex}"
    workflow_schema = f"{schema_base}_wf"
    conversation_schema = f"{schema_base}_conv"

    def _node(node_id: str, op: str, *, start: bool = False, terminal: bool = False):
        grounding = Grounding(spans=[Span.from_dummy_for_conversation()])
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[grounding],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": "wf-pg-client-resume",
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

    def _edge(edge_id: str, source_id: str, target_id: str):
        grounding = Grounding(spans=[Span.from_dummy_for_conversation()])
        return WorkflowEdge(
            id=edge_id,
            source_ids=[source_id],
            target_ids=[target_id],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=edge_id,
            mentions=[grounding],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": "wf-pg-client-resume",
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

    try:
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "workflow-client-resume-pg-meta"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=3, schema=workflow_schema
            ),
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conversation-client-resume-pg-meta"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=3, schema=conversation_schema
            ),
        )
        for node in (
            _node("start", "start", start=True),
            _node("gate", "gate"),
            _node("end", "end", terminal=True),
        ):
            workflow_engine.write.add_node(node)
        workflow_engine.write.add_edge(_edge("start-gate", "start", "gate"))
        workflow_engine.write.add_edge(_edge("gate-end", "gate", "end"))

        resolver = AsyncMappingStepResolver()
        seen: list[tuple[str, int]] = []

        @resolver.register("start")
        async def _start(ctx):
            seen.append((ctx.workflow_node_id, ctx.step_seq))
            return RunSuccess(conversation_node_id=None, state_update=[])

        @resolver.register("gate")
        async def _gate(ctx):
            seen.append((ctx.workflow_node_id, ctx.step_seq))
            return RunSuspended(
                conversation_node_id=None,
                resume_payload={"prompt": "approve"},
            )

        @resolver.register("end")
        async def _end(ctx):
            seen.append((ctx.workflow_node_id, ctx.step_seq))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"completed_after_resume": True})],
            )

        runtime = AsyncWorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            max_workers=1,
            trace=False,
            experimental_native_scheduler=True,
        )
        run_id = f"run-pg-client-resume-{uuid.uuid4().hex}"
        conversation_id = f"conversation-pg-client-resume-{uuid.uuid4().hex}"
        turn_node_id = f"turn-{uuid.uuid4().hex}"
        suspended = await runtime.run(
            workflow_id="wf-pg-client-resume",
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state={"started": True},
            run_id=run_id,
        )
        checkpoint = runtime.sync_runtime._latest_checkpoint_for_run(
            conversation_id=conversation_id, run_id=run_id
        )
        from kogwistar.runtime.checkpoint_state import decode_checkpoint_state

        checkpoint_state = decode_checkpoint_state(
            checkpoint.metadata["state_json"], run_id=run_id
        )
        token_id = checkpoint_state["_rt_join"]["suspended"][0][2]

        resumed = await runtime.resume_run(
            run_id=run_id,
            suspended_node_id="gate",
            suspended_token_id=token_id,
            client_result=RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"approved": True})],
            ),
            workflow_id="wf-pg-client-resume",
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
        )

        assert suspended.status == "suspended"
        assert resumed.status == "succeeded"
        assert resumed.final_state["approved"] is True
        assert resumed.final_state["completed_after_resume"] is True
        assert seen == [("start", 0), ("gate", 1), ("end", 3)]
        step_nodes = conversation_engine.read.get_nodes(
            where={"$and": [{"entity_type": "workflow_step_exec"}, {"run_id": run_id}]},
            limit=20,
        )
        by_seq = {int(step.metadata["step_seq"]): step for step in step_nodes}
        assert sorted(by_seq) == [0, 1, 2, 3]
        assert by_seq[2].metadata["op"] == "client_sandbox_resume"
        assert by_seq[3].metadata["workflow_node_id"] == "end"
        next_edges = conversation_engine.read.get_edges(
            where={"$and": [{"run_id": run_id}, {"relation": "wf_next_step_exec"}]},
            limit=20,
        )
        assert any(
            edge.source_ids == [by_seq[2].safe_get_id()]
            and edge.target_ids == [by_seq[3].safe_get_id()]
            for edge in next_edges
        )
    finally:
        with sa_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{workflow_schema}" CASCADE'))
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{conversation_schema}" CASCADE'))


def test_async_runtime_resume_run_delegates_to_selected_rust_authority(monkeypatch):
    import kogwistar.runtime.rust_runtime_authority as rust_authority

    calls: list[tuple[str, object]] = []

    class _FakeAuthority:
        def __init__(self, *, runtime, base_url, cache_dir):
            calls.append(("init", base_url))

        async def resume(self, **kwargs):
            calls.append(("resume", kwargs))
            return RunResult(
                run_id=str(kwargs["run_id"]),
                final_state={"rust_resumed": True},
                mq=queue.Queue(),
                status="succeeded",
            )

        async def aclose(self):
            calls.append(("close", None))

    monkeypatch.setattr(rust_authority, "rust_runtime_authority_selected", lambda: True)
    monkeypatch.setattr(
        rust_authority, "rust_runtime_authority_url", lambda: "http://rust-runtime"
    )
    monkeypatch.setattr(rust_authority, "AsyncRustRuntimeAuthority", _FakeAuthority)

    runtime = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda op: (
            lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])
        ),
        predicate_registry={},
        experimental_native_scheduler=True,
    )

    def _python_checkpoint_must_not_run(**kwargs):
        raise AssertionError("Rust selection must not read the Python checkpoint")

    runtime.sync_runtime._latest_checkpoint_for_run = _python_checkpoint_must_not_run
    result = asyncio.run(
        runtime.resume_run(
            run_id="run-rust-resume",
            suspended_node_id="gate",
            suspended_token_id="token-rust-resume",
            client_result=RunSuccess(conversation_node_id=None, state_update=[]),
            workflow_id="wf-rust-resume",
            conversation_id="conv-rust-resume",
            turn_node_id="turn-rust-resume",
        )
    )

    assert result.status == "succeeded"
    assert result.final_state == {"rust_resumed": True}
    assert [name for name, _value in calls] == ["init", "resume", "close"]
    resume_payload = calls[1][1]
    assert isinstance(resume_payload, dict)
    assert resume_payload["suspended_token_id"] == "token-rust-resume"


def test_async_runtime_checkpoint_resume_runs_pending_frontier(monkeypatch):
    """Sync mirror: `tests/runtime/test_workflow_terminal_status.py::test_runtime_resume_from_latest_checkpoint_reuses_state_and_sequence`."""

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
        def __init__(self, node_id, op, *, terminal=False):
            self.id = node_id
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": False}

    class _Checkpoint:
        metadata = {
            "workflow_id": "wf-resume",
            "step_seq": 6,
            "state_json": {
                "before_resume": True,
                "_rt_join": {
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                    "pending": [("finish", 0, "token-finish", None)],
                    "suspended": [],
                },
            },
        }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start")
        finish = _Node("finish", "finish", terminal=True)
        return start, {"start": start, "finish": finish}, {"start": [], "finish": []}

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    seen: list[tuple[str, int]] = []
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(ctx):
        seen.append((ctx.workflow_node_id, ctx.step_seq))
        return RunSuccess(conversation_node_id=None, state_update=[])

    @resolver.register("finish")
    async def _finish(ctx):
        seen.append((ctx.workflow_node_id, ctx.step_seq))
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"resumed": True})],
        )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    rt.sync_runtime._latest_checkpoint_for_run = lambda **_kwargs: _Checkpoint()

    out = asyncio.run(
        rt.resume_from_latest_checkpoint(
            run_id="run-resume",
            workflow_id="wf-resume",
            conversation_id="c1",
            turn_node_id="t1",
        )
    )

    assert out.status == "succeeded"
    assert out.final_state["before_resume"] is True
    assert out.final_state["resumed"] is True
    assert seen == [("finish", 7)]

    with pytest.raises(ValueError, match="checkpoint has no pending work"):
        asyncio.run(
            rt.run(
                workflow_id="wf-resume",
                conversation_id="c1",
                turn_node_id="t1",
                run_id="run-empty-frontier",
                initial_state={
                    "_rt_join": {
                        "join_node_ids": [],
                        "join_outstanding": [],
                        "join_waiters": {},
                        "pending": [],
                        "suspended": [],
                    }
                },
                _resume_step_seq=8,
            )
        )
    assert seen == [("finish", 7)]


def test_async_runtime_checkpoint_restore_preserves_suspended_frontier(monkeypatch):
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
        def __init__(self, node_id, op, *, terminal=False):
            self.id = node_id
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": False}

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start")
        finish = _Node("finish", "finish", terminal=True)
        return start, {"start": start, "finish": finish}, {"start": [], "finish": []}

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    resolver = AsyncMappingStepResolver()
    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    suspended_frontier = [("finish", 0, "token-suspended", None)]
    out = asyncio.run(
        rt.run(
            workflow_id="wf-suspended-restore",
            conversation_id="c1",
            turn_node_id="t1",
            run_id="run-suspended-restore",
            initial_state={
                "_rt_join": {
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                    "pending": [],
                    "suspended": suspended_frontier,
                }
            },
        )
    )

    assert out.status == "suspended"
    assert out.final_state["_rt_join"]["suspended"] == suspended_frontier
    assert out.final_state["_rt_join"]["pending"] == []


@pytest.mark.ci_full
@pytest.mark.asyncio
async def test_async_runtime_checkpoint_resume_roundtrip_postgres(request, tmp_path):
    """Resume an async runtime from a checkpoint persisted in PostgreSQL."""
    import uuid

    from sqlalchemy import text

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
    from kogwistar.runtime.models import WorkflowNode
    from tests.conftest import FakeEmbeddingFunction

    sa_engine = request.getfixturevalue("sa_engine")
    if sa_engine is None:
        pytest.skip("PostgreSQL fixture is unavailable")

    schema_base = f"async_resume_{uuid.uuid4().hex}"
    workflow_schema = f"{schema_base}_wf"
    conversation_schema = f"{schema_base}_conv"

    def _workflow_node(
        node_id: str, op: str, *, start: bool = False, terminal: bool = False
    ):
        grounding = Grounding(spans=[Span.from_dummy_for_conversation()])
        return WorkflowNode(
            id=node_id,
            label=node_id,
            type="entity",
            doc_id=node_id,
            summary=op,
            mentions=[grounding],
            properties={},
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": "wf-pg-resume",
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

    try:
        workflow_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "workflow-pg-meta"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=3, schema=workflow_schema
            ),
        )
        conversation_engine = GraphKnowledgeEngine(
            persist_directory=str(tmp_path / "conversation-pg-meta"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend=PgVectorBackend(
                engine=sa_engine, embedding_dim=3, schema=conversation_schema
            ),
        )
        workflow_engine.write.add_node(_workflow_node("start", "start", start=True))
        workflow_engine.write.add_node(
            _workflow_node("finish", "finish", terminal=True)
        )

        seen: list[tuple[str, int]] = []
        resolver = AsyncMappingStepResolver()

        @resolver.register("start")
        async def _start(ctx):
            seen.append((ctx.workflow_node_id, ctx.step_seq))
            return RunSuccess(conversation_node_id=None, state_update=[])

        @resolver.register("finish")
        async def _finish(ctx):
            seen.append((ctx.workflow_node_id, ctx.step_seq))
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"resumed_from_pg": True})],
            )

        runtime = AsyncWorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            trace=False,
            checkpoint_every_n_steps=1,
            experimental_native_scheduler=True,
        )
        run_id = f"run-pg-resume-{uuid.uuid4().hex}"
        runtime.sync_runtime._persist_workflow_run(
            conversation_id="conversation-pg-resume",
            workflow_id="wf-pg-resume",
            run_id=run_id,
            turn_node_id="turn-pg-resume",
            status="planned",
        )
        runtime.sync_runtime._persist_checkpoint(
            conversation_id="conversation-pg-resume",
            workflow_id="wf-pg-resume",
            run_id=run_id,
            step_seq=6,
            state={
                "before_resume": True,
                "_rt_join": {
                    "join_node_ids": [],
                    "join_outstanding": [],
                    "join_waiters": {},
                    "pending": [("finish", 0, "token-finish", None)],
                    "suspended": [],
                },
            },
        )

        out = await runtime.resume_from_latest_checkpoint(
            run_id=run_id,
            workflow_id="wf-pg-resume",
            conversation_id="conversation-pg-resume",
            turn_node_id="turn-pg-resume",
        )

        assert out.status == "succeeded"
        assert out.final_state["before_resume"] is True
        assert out.final_state["resumed_from_pg"] is True
        assert seen == [("finish", 7)]
        step_nodes = conversation_engine.read.get_nodes(
            where={"$and": [{"run_id": run_id}, {"entity_type": "workflow_step_exec"}]},
            limit=10,
        )
        assert len(step_nodes) == 1
        assert step_nodes[0].metadata["workflow_node_id"] == "finish"
        assert step_nodes[0].metadata["step_seq"] == 7
    finally:
        with sa_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{workflow_schema}" CASCADE'))
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{conversation_schema}" CASCADE'))


def test_async_runtime_rust_authority_rejects_native_resume_marker(monkeypatch):
    import importlib

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=object(),
        step_resolver=lambda _op: (
            lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])
        ),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    authority = importlib.import_module("kogwistar.runtime.rust_runtime_authority")
    monkeypatch.setattr(authority, "rust_runtime_authority_selected", lambda: True)

    with pytest.raises(NotImplementedError, match="Rust async authority"):
        asyncio.run(
            rt.run(
                workflow_id="wf-1",
                conversation_id="c-1",
                turn_node_id="t-1",
                initial_state={},
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
            self.metadata = {
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
                "wf_predicate": None,
            }

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node("start", "start", terminal=False, fanout=False)
        end = _Node("end", "end", terminal=True, fanout=False)
        return start, {"start": start, "end": end}, {"start": [_Edge("end")], "end": []}

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
            return RunResult(
                run_id="sync",
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
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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
            return RunSuccess(
                conversation_node_id=None, state_update=[("a", {"seen": "a"})]
            )
        finally:
            active["n"] -= 1

    @resolver.register("b")
    async def _b(_ctx):
        active["n"] += 1
        max_seen["n"] = max(max_seen["n"], active["n"])
        try:
            await asyncio.sleep(0.08)
            return RunSuccess(
                conversation_node_id=None, state_update=[("a", {"seen": "b"})]
            )
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


def test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order(
    monkeypatch,
):
    """Sync mirror: ordered step completion in sync workflow step dispatch."""

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return RunResult(
                run_id="sync",
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
        a = _Node("a", "a", terminal=True, fanout=False)
        b = _Node("b", "b", terminal=True, fanout=False)
        return (
            start,
            {"start": start, "a": a, "b": b},
            {"start": [_Edge("a"), _Edge("b")], "a": [], "b": []},
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

    delay = {"a": 0.08, "b": 0.01}

    @resolver.register("a")
    async def _a(_ctx):
        await asyncio.sleep(delay["a"])
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"order": "a"})]
        )

    @resolver.register("b")
    async def _b(_ctx):
        await asyncio.sleep(delay["b"])
        return RunSuccess(
            conversation_node_id=None, state_update=[("a", {"order": "b"})]
        )

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
    # Each top-level run gets a fresh runtime run id; compare the workflow
    # result rather than that intentionally volatile plumbing key.
    stable_state_1 = {
        key: value
        for key, value in out1.final_state.items()
        if key != "_wf_current_run_id"
    }
    stable_state_2 = {
        key: value
        for key, value in out2.final_state.items()
        if key != "_wf_current_run_id"
    }
    assert stable_state_1 == stable_state_2


@pytest.mark.ci_full
@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata(
    monkeypatch,
):
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

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"path": "trace"})]
        )

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


@pytest.mark.ci_full
@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort(
    monkeypatch,
):
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

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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


@pytest.mark.ci_full
@pytest.mark.asyncio
async def test_async_runtime_native_scheduler_nested_invocation_reuses_trace_emitter(
    monkeypatch,
):
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
            edge = type(
                "_Edge",
                (),
                {
                    "target_ids": ["end"],
                    "metadata": {
                        "wf_priority": 100,
                        "wf_is_default": True,
                        "wf_multiplicity": "one",
                        "wf_predicate": None,
                    },
                },
            )()
            return start, {"start": start, "end": end}, {"start": [edge], "end": []}
        child = _Node("child-start", "child", terminal=True, fanout=False)
        return child, {"child-start": child}, {"child-start": []}

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.WorkflowRuntime", _FakeWorkflowRuntime
    )
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design", _fake_validate
    )

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
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"child": True})]
        )

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(
            conversation_node_id=None, state_update=[("u", {"done": True})]
        )

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


@pytest.mark.ci_full
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_kind",
    [
        pytest.param("fake", id="fake", marks=pytest.mark.ci_full),
        pytest.param("chroma", id="chroma", marks=pytest.mark.ci_full),
        pytest.param("pg", id="pg", marks=pytest.mark.ci_full),
    ],
)
async def test_async_runtime_suspend_with_join_waiter_returns_suspended_when_idle(
    tmp_path, request, backend_kind
):
    """Sync mirror: `tests/runtime/test_workflow_suspend_resume.py::test_workflow_suspend_with_join_waiter_returns_suspended_when_idle`."""
    import uuid
    from contextlib import contextmanager

    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.engine_core.models import Grounding, Span
    from kogwistar.engine_core.postgres_backend import PgVectorBackend
    from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
    from tests._helpers.fake_backend import build_fake_backend
    from tests.conftest import FakeEmbeddingFunction, _is_missing_pgvector_extension

    def _g():
        return Grounding(spans=[Span.from_dummy_for_conversation()])

    def _node(
        node_id: str, op: str, *, start=False, fanout=False, join=False, terminal=False
    ):
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
                "workflow_id": "join_waiter_suspend_idle_async",
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
                "wf_fanout": fanout,
                "wf_join": join,
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            embedding=None,
        )

    def _edge(src: str, dst: str):
        return WorkflowEdge(
            id=f"{src}->{dst}",
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=f"{src}->{dst}",
            mentions=[_g()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": "join_waiter_suspend_idle_async",
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

    @contextmanager
    def _engine_pair():
        if backend_kind == "fake":
            wf_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "wf_fake"),
                kg_graph_type="workflow",
                embedding_function=FakeEmbeddingFunction(dim=3),
                backend_factory=build_fake_backend,
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "conv_fake"),
                kg_graph_type="conversation",
                embedding_function=FakeEmbeddingFunction(dim=3),
                backend_factory=build_fake_backend,
            )
            yield wf_engine, conv_engine
            return

        if backend_kind == "chroma":
            pytest.importorskip("chromadb")
            wf_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "wf_chroma"),
                kg_graph_type="workflow",
                embedding_function=FakeEmbeddingFunction(dim=3),
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(tmp_path / "conv_chroma"),
                kg_graph_type="conversation",
                embedding_function=FakeEmbeddingFunction(dim=3),
            )
            yield wf_engine, conv_engine
            return

        if backend_kind == "pg":
            sa = pytest.importorskip("sqlalchemy")
            sa_engine = request.getfixturevalue("sa_engine")
            if sa_engine is None:
                pytest.skip("pg fixtures are unavailable")
            base_schema = f"join_suspend_async_{uuid.uuid4().hex}"
            wf_schema = f"{base_schema}_wf"
            conv_schema = f"{base_schema}_conv"
            try:
                try:
                    wf_backend = PgVectorBackend(
                        engine=sa_engine, embedding_dim=3, schema=wf_schema
                    )
                    conv_backend = PgVectorBackend(
                        engine=sa_engine, embedding_dim=3, schema=conv_schema
                    )
                except Exception as exc:
                    if _is_missing_pgvector_extension(exc):
                        pytest.skip(f"pg backend unavailable: {exc}")
                    raise
                wf_engine = GraphKnowledgeEngine(
                    persist_directory=str(tmp_path / "wf_pg_meta"),
                    kg_graph_type="workflow",
                    embedding_function=FakeEmbeddingFunction(dim=3),
                    backend=wf_backend,
                )
                conv_engine = GraphKnowledgeEngine(
                    persist_directory=str(tmp_path / "conv_pg_meta"),
                    kg_graph_type="conversation",
                    embedding_function=FakeEmbeddingFunction(dim=3),
                    backend=conv_backend,
                )
                yield wf_engine, conv_engine
            finally:
                with sa_engine.begin() as conn:
                    conn.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{wf_schema}" CASCADE')
                    )
                    conn.execute(
                        sa.text(f'DROP SCHEMA IF EXISTS "{conv_schema}" CASCADE')
                    )
            return

        raise ValueError(f"unsupported backend_kind: {backend_kind}")

    seen: list[str] = []

    def _resolver(op: str):
        async def _start(_ctx):
            seen.append("start")
            return RunSuccess(conversation_node_id=None, state_update=[])

        async def _light(_ctx):
            seen.append("light")
            return RunSuccess(conversation_node_id=None, state_update=[])

        async def _pause(_ctx):
            seen.append("pause")
            return RunSuspended(
                conversation_node_id=None,
                resume_payload={
                    "type": "recoverable_error",
                    "op": "pause_op",
                    "category": "budget_pinned",
                    "message": "sibling branch paused",
                    "errors": ["paused"],
                },
            )

        async def _join(_ctx):
            seen.append("join")
            raise AssertionError(
                "join must not execute while sibling token is suspended"
            )

        async def _end(_ctx):
            seen.append("end")
            return RunSuccess(conversation_node_id=None, state_update=[])

        return {
            "start_op": _start,
            "light_op": _light,
            "pause_op": _pause,
            "join_op": _join,
            "end_op": _end,
        }[op]

    wf_id = "join_waiter_suspend_idle_async"
    with _engine_pair() as (wf_engine, conv_engine):
        for node in (
            _node("start", "start_op", start=True, fanout=True),
            _node("a_light", "light_op"),
            _node("z_pause", "pause_op"),
            _node("join", "join_op", join=True),
            _node("end", "end_op", terminal=True),
        ):
            wf_engine.write.add_node(node)
        for edge in (
            _edge("start", "a_light"),
            _edge("start", "z_pause"),
            _edge("a_light", "join"),
            _edge("z_pause", "join"),
            _edge("join", "end"),
        ):
            wf_engine.write.add_edge(edge)

        out = await asyncio.wait_for(
            AsyncWorkflowRuntime(
                workflow_engine=wf_engine,
                conversation_engine=conv_engine,
                step_resolver=_resolver,
                predicate_registry={},
                checkpoint_every_n_steps=1,
                max_workers=1,
                experimental_native_scheduler=True,
            ).run(
                workflow_id=wf_id,
                conversation_id="conv_join_waiter_suspend_async",
                turn_node_id="turn_1",
                initial_state={
                    "conversation_id": "conv_join_waiter_suspend_async",
                    "user_id": "user_1",
                    "turn_node_id": "turn_1",
                    "turn_index": 0,
                    "role": "user",
                    "user_text": "",
                    "mem_id": "mem_1",
                },
                run_id=f"run_{uuid.uuid4().hex}",
            ),
            timeout=10,
        )

    rt_join = out.final_state.get("_rt_join", {})
    assert out.status == "suspended"
    assert seen == ["start", "light", "pause"]
    assert rt_join.get("suspended", [])[0][0] == "z_pause"
    assert rt_join.get("join_waiters", {}).get("join")
