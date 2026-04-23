from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass

import pytest

from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime import AsyncMappingStepResolver, AsyncWorkflowRuntime, MappingStepResolver
from kogwistar.runtime.async_runtime import _SyncResolverAdapter
from kogwistar.runtime.replay import _apply_state_update
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.runtime import RunResult, StepContext, WorkflowRuntime

pytestmark = [pytest.mark.ci]


def test_async_runtime_default_ops_equal():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_default_ops_equal`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_default_sync_ops_equal_default_async_ops`.
    """
    async_resolver = AsyncMappingStepResolver(
        handlers=dict(default_resolver.handlers),
        default=default_resolver.default,
    )
    assert set(default_resolver.ops) == set(async_resolver.ops)


def test_async_runtime_exported():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_exported`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_exported`.
    """
    from kogwistar.runtime import AsyncMappingStepResolver as ExportedAsyncResolver
    from kogwistar.runtime import AsyncWorkflowRuntime as Exported

    assert Exported is AsyncWorkflowRuntime
    assert ExportedAsyncResolver is AsyncMappingStepResolver


def test_async_runtime_adapter_forwards_close_sandbox_run():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_adapter_forwards_close_sandbox_run`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_adapter_forwards_close_sandbox_run`.
    """

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


def test_async_runtime_accept_same_workflow_graph_model():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_accept_same_workflow_graph_model`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_sync_and_async_runtime_accept_same_workflow_graph_model`.
    """
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


def test_async_runtime_state_merge_semantics():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_state_merge_semantics`.
    Refactored from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`.
    Source retained: native scheduler test owns end-to-end state merge; this test owns sync/async bijection.
    """
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = type("_Resolver", (), {"_state_schema": {}})()

    sync_state = {}
    replay_state = {}

    state_update = [
        ("u", {"answer": "ok"}),
        ("a", {"ops": "first"}),
        ("e", {"nums": [1, 2]}),
    ]

    runtime.apply_state_update(sync_state, state_update)
    _apply_state_update(replay_state, state_update)

    assert replay_state == sync_state


def test_async_runtime_step_context_and_result_contract():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_step_context_and_result_contract`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_step_context_and_result_contract`.
    """
    ctx = StepContext(
        run_id="run-async-bijection",
        workflow_id="wf-async-bijection",
        workflow_node_id="node-async-bijection",
        op="noop",
        token_id="tok-async-bijection",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        state={},
    )

    assert ctx.trace_ctx.run_id == "run-async-bijection"
    assert ctx.trace_ctx.node_id == "node-async-bijection"

    result = RunSuccess(
        conversation_node_id=None,
        state_update=[("u", {"ok": True})],
    )
    assert result.status == "success"
    assert result.state_update == [("u", {"ok": True})]


def test_async_runtime_state_schema_metadata_available():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_state_schema_metadata_available`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_state_schema_metadata_available`.
    """
    async_resolver = AsyncMappingStepResolver()
    async_resolver.set_state_schema({"events": "a", "answer": "u"})
    assert async_resolver.describe_state() == {"events": "a", "answer": "u"}


def test_async_runtime_deps_available_in_handler():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_deps_available_in_handler`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_deps_available_in_handler`.
    """
    async_resolver = AsyncMappingStepResolver()

    class _DepsCtx:
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


def test_async_runtime_deps_live_but_omitted_from_checkpoint_payload(monkeypatch):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_deps_live_but_omitted_from_checkpoint_payload`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_deps_live_but_omitted_from_checkpoint_payload`.
    """
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.step_resolver = kwargs["step_resolver"]

        def run(self, **kwargs):
            initial_state = dict(kwargs.get("initial_state") or {})
            checkpoint_state = {k: v for k, v in initial_state.items() if k != "_deps"}

            class _Ctx:
                state_view = initial_state

            live_result = self.step_resolver("use_deps")(_Ctx())
            live_out = asyncio.run(live_result) if asyncio.iscoroutine(live_result) else live_result
            return type(
                "_Out",
                (),
                {
                    "status": "succeeded",
                    "final_state": {
                        "live_x": live_out.state_update[0][1]["x"],
                        "checkpoint_state": checkpoint_state,
                    },
                },
            )()

    resolver = MappingStepResolver()

    @resolver.register("use_deps")
    def _use_deps(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"x": ctx.state_view["_deps"]["x"]})],
        )

    rt = _FakeWorkflowRuntime(step_resolver=resolver)
    out = rt.run(initial_state={"_deps": {"x": 9}, "keep": 1})
    assert out.status == "succeeded"
    assert out.final_state["live_x"] == 9
    assert "_deps" not in out.final_state["checkpoint_state"]
    assert out.final_state["checkpoint_state"]["keep"] == 1


def test_async_runtime_linear_terminal_status_equivalent(monkeypatch):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_linear_terminal_status_equivalent`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_linear_terminal_status_equivalent_to_sync`.
    """

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

    @dataclass
    class _Node:
        id: str
        op: str
        metadata: dict

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node(
            id="wf-linear:start",
            op="start",
            metadata={"wf_terminal": True},
        )
        return start, {start.id: start}, {start.id: []}

    class _Write:
        def add_node(self, _node):
            return None

        def add_edge(self, _edge):
            return None

    class _Engine:
        write = _Write()

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"path": "linear"})],
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design",
        _fake_validate,
    )
    rt = AsyncWorkflowRuntime(
        workflow_engine=_Engine(),
        conversation_engine=_Engine(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    sync_out = _FakeWorkflowRuntime().run(
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
    async_user_state = {
        k: v for k, v in async_out.final_state.items() if k != "_rt_join"
    }
    assert async_user_state == sync_out.final_state


def test_async_runtime_branch_join_status_and_state_equivalent(monkeypatch):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_branch_join_status_and_state_equivalent`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_branch_join_status_and_state_equivalent_to_sync`.
    """

    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            state = dict(kwargs.get("initial_state") or {})
            branch_values = list(state.get("branch_values") or [])
            for value in branch_values:
                state[f"b{value}"] = int(value)
            state["joined_total"] = sum(int(v) for v in branch_values)
            state["path"] = "branch_join"
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-branch-join"),
                final_state=state,
                mq=queue.Queue(),
                status="succeeded",
            )

    @dataclass
    class _Node:
        id: str
        op: str
        metadata: dict

    @dataclass
    class _Edge:
        id: str
        source_ids: list[str]
        target_ids: list[str]
        metadata: dict

    def _edge(edge_id: str, src: str, dst: str, *, many: bool = False) -> _Edge:
        return _Edge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            metadata={
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "many" if many else "one",
                "wf_predicate": None,
            },
        )

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        nodes = {
            "wf-branch-join:start": _Node(
                id="wf-branch-join:start",
                op="start",
                metadata={"wf_fanout": True},
            ),
            "wf-branch-join:b2": _Node(
                id="wf-branch-join:b2",
                op="b2",
                metadata={},
            ),
            "wf-branch-join:b3": _Node(
                id="wf-branch-join:b3",
                op="b3",
                metadata={},
            ),
            "wf-branch-join:b5": _Node(
                id="wf-branch-join:b5",
                op="b5",
                metadata={},
            ),
            "wf-branch-join:join": _Node(
                id="wf-branch-join:join",
                op="join",
                metadata={"wf_join": True, "wf_join_is_merge": True},
            ),
            "wf-branch-join:end": _Node(
                id="wf-branch-join:end",
                op="end",
                metadata={"wf_terminal": True},
            ),
        }
        adj = {
            "wf-branch-join:start": [
                _edge("e-start-b2", "wf-branch-join:start", "wf-branch-join:b2", many=True),
                _edge("e-start-b3", "wf-branch-join:start", "wf-branch-join:b3", many=True),
                _edge("e-start-b5", "wf-branch-join:start", "wf-branch-join:b5", many=True),
            ],
            "wf-branch-join:b2": [
                _edge("e-b2-join", "wf-branch-join:b2", "wf-branch-join:join"),
            ],
            "wf-branch-join:b3": [
                _edge("e-b3-join", "wf-branch-join:b3", "wf-branch-join:join"),
            ],
            "wf-branch-join:b5": [
                _edge("e-b5-join", "wf-branch-join:b5", "wf-branch-join:join"),
            ],
            "wf-branch-join:join": [
                _edge("e-join-end", "wf-branch-join:join", "wf-branch-join:end"),
            ],
            "wf-branch-join:end": [],
        }
        return nodes["wf-branch-join:start"], nodes, adj

    class _Write:
        def add_node(self, _node):
            return None

        def add_edge(self, _edge):
            return None

    class _Engine:
        write = _Write()

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    for op, value in (("b2", 2), ("b3", 3), ("b5", 5)):

        @resolver.register(op)
        async def _branch(_ctx, op=op, value=value):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {op: value})],
            )

    @resolver.register("join")
    async def _join(ctx):
        total = sum(int(ctx.state_view.get(op, 0)) for op in ("b2", "b3", "b5"))
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"joined_total": total})],
        )

    @resolver.register("end")
    async def _end(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"path": "branch_join"})],
        )

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design",
        _fake_validate,
    )
    rt = AsyncWorkflowRuntime(
        workflow_engine=_Engine(),
        conversation_engine=_Engine(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )
    initial = {"branch_values": [2, 3, 5]}
    sync_out = _FakeWorkflowRuntime().run(
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
    async_user_state = {
        k: v for k, v in async_out.final_state.items() if k != "_rt_join"
    }
    assert async_user_state == sync_out.final_state
    assert async_user_state["joined_total"] == 10


def test_async_runtime_route_next_alias_can_fan_out_multiple_branches():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_route_next_alias_can_fan_out_multiple_branches`.
    Moved from `tests/runtime/test_workflow_invocation_and_route_next.py::test_route_next_alias_can_fan_out_multiple_branches`.
    """
    @dataclass
    class _Node:
        id: str
        label: str
        op: str
        metadata: dict

    @dataclass
    class _Edge:
        id: str
        label: str
        source_ids: list[str]
        target_ids: list[str]
        metadata: dict

        def safe_get_id(self):
            return self.id

        @property
        def priority(self):
            return int(self.metadata.get("wf_priority", 100))

        @property
        def multiplicity(self):
            return str(self.metadata.get("wf_multiplicity", "one"))

        @property
        def is_default(self):
            return bool(self.metadata.get("wf_is_default", False))

        @property
        def predicate(self):
            return self.metadata.get("wf_predicate")

    node = _Node(id="start", label="start", op="start", metadata={"wf_fanout": True})
    edges = [
        _Edge(
            id="e-left",
            label="go_left",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={"wf_priority": 100, "wf_multiplicity": "many", "wf_is_default": False, "wf_predicate": None},
        ),
        _Edge(
            id="e-right",
            label="go_right",
            source_ids=["start"],
            target_ids=["n|right"],
            metadata={"wf_priority": 100, "wf_multiplicity": "many", "wf_is_default": False, "wf_predicate": None},
        ),
    ]
    picked = AsyncWorkflowRuntime._select_next_edges(
        node,
        edges,
        {},
        RunSuccess(conversation_node_id=None, state_update=[], _route_next=["go_left", "n|right"]),
        {},
    )
    assert [edge.id for edge in picked] == ["e-left", "e-right"]

    pred_node = _Node(id="start", label="start", op="start", metadata={"wf_fanout": False})
    pred_edges = [
        _Edge(
            id="e-pred",
            label="pred_path",
            source_ids=["start"],
            target_ids=["n|left"],
            metadata={"wf_priority": 100, "wf_multiplicity": "one", "wf_is_default": False, "wf_predicate": "if_true"},
        ),
        _Edge(
            id="e-default",
            label="default_path",
            source_ids=["start"],
            target_ids=["n|fallback"],
            metadata={"wf_priority": 100, "wf_multiplicity": "one", "wf_is_default": True, "wf_predicate": "if_false"},
        ),
    ]
    picked = AsyncWorkflowRuntime._select_next_edges(
        pred_node,
        pred_edges,
        {},
        RunSuccess(conversation_node_id=None, state_update=[]),
        {},
    )
    assert [edge.id for edge in picked] == ["e-default"]


def test_async_runtime_native_update_schema_applies_known_and_falls_back_unknown(monkeypatch):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_native_update_schema_applies_known_and_falls_back_unknown`.
    Moved from `tests/workflow/test_workflow_native_update.py::test_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown`.
    """
    @dataclass
    class _Node:
        id: str
        op: str
        metadata: dict

    def _fake_validate(*, workflow_engine, workflow_id, predicate_registry, resolver):
        start = _Node(
            id="wf_native_update:start",
            op="start",
            metadata={"wf_terminal": True},
        )
        return start, {start.id: start}, {start.id: []}

    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.validate_workflow_design",
        _fake_validate,
    )

    class _Write:
        def add_node(self, _node):
            return None

        def add_edge(self, _edge):
            return None

    class _Engine:
        write = _Write()

    resolver = AsyncMappingStepResolver()
    resolver.set_state_schema({"op_log": "a"})

    @resolver.register("start")
    async def _start(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            update={"op_log": ["x"], "dyn": 1},
        )

    rt = AsyncWorkflowRuntime(
        workflow_engine=_Engine(),
        conversation_engine=_Engine(),
        step_resolver=resolver,
        predicate_registry={},
        trace=False,
    )
    out = asyncio.run(
        rt.run(
            workflow_id="wf_native_update",
            conversation_id="c1",
            turn_node_id="t1",
            initial_state={},
            run_id="run-native",
        )
    )
    assert out.final_state["op_log"] == ["x"]
    assert out.final_state["dyn"] == 1


def test_async_runtime_step_context_send_lane_message_delegates_to_sender(tmp_path):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_step_context_send_lane_message_delegates_to_sender`.
    New async mirror for the StepContext lane-message delegation bijection pair.
    """
    calls = []

    def _sender(**kwargs):
        calls.append(kwargs)
        return {"message_id": "msg-1"}

    ctx = StepContext(
        run_id="run-1",
        workflow_id="wf-1",
        workflow_node_id="node-1",
        op="op-1",
        token_id="tok-1",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path,
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={"x": 1},
        lane_message_sender=_sender,
    )

    result = ctx.send_lane_message(
        conversation_id="conv-1",
        inbox_id="inbox:worker:demo",
        sender_id="lane:foreground",
        recipient_id="lane:worker:demo",
        msg_type="request.demo",
        payload={"hello": "world"},
    )

    assert result == {"message_id": "msg-1"}
    assert calls and calls[0]["msg_type"] == "request.demo"


def test_async_runtime_step_context_send_lane_message_requires_sender(tmp_path):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_step_context_send_lane_message_requires_sender`.
    New async mirror for the StepContext missing-sender bijection pair.
    """
    ctx = StepContext(
        run_id="run-1",
        workflow_id="wf-1",
        workflow_node_id="node-1",
        op="op-1",
        token_id="tok-1",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path,
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={"x": 1},
    )

    try:
        ctx.send_lane_message(
            conversation_id="conv-1",
            inbox_id="inbox:worker:demo",
            sender_id="lane:foreground",
            recipient_id="lane:worker:demo",
            msg_type="request.demo",
            payload={"hello": "world"},
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "lane message sender not configured" in str(exc)


def test_async_runtime_step_context_emit_lane_message_event_delegates_to_sink(tmp_path):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_step_context_emit_lane_message_event_delegates_to_sink`.
    New async mirror for the StepContext lane-event delegation bijection pair.
    """
    calls = []

    def _sink(event):
        calls.append(event)

    ctx = StepContext(
        run_id="run-1",
        workflow_id="wf-1",
        workflow_node_id="node-1",
        op="op-1",
        token_id="tok-1",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path,
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={"x": 1},
        lane_message_event_sink=_sink,
    )

    ctx.emit_lane_message_event({"event_type": "worker.requested", "run_id": "run-1"})

    assert calls == [{"event_type": "worker.requested", "run_id": "run-1"}]


def test_async_runtime_step_context_emit_lane_message_event_requires_sink(tmp_path):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_step_context_emit_lane_message_event_requires_sink`.
    New async mirror for the StepContext missing-sink bijection pair.
    """
    ctx = StepContext(
        run_id="run-1",
        workflow_id="wf-1",
        workflow_node_id="node-1",
        op="op-1",
        token_id="tok-1",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path,
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={"x": 1},
    )

    try:
        ctx.emit_lane_message_event({"event_type": "worker.requested", "run_id": "run-1"})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "event sink not configured" in str(exc)


def test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_preserves_nested_ops_and_state_schema_in_adapter`.
    Refactored from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter`.
    Source retained: adapter test owns sync-adapter metadata; this test owns async resolver parity.
    """
    resolver = AsyncMappingStepResolver()
    resolver.nested_ops.add("noop")
    resolver.set_state_schema({"events": "a"})

    @resolver.register("noop")
    def _sync_handler(ctx):
        assert "events" not in ctx.state_view
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ok": True})],
        )

    out = asyncio.run(
        resolver.resolve_async("noop")(
            type(
                "_Ctx",
                (),
                {
                    "conversation_id": "conv-async-bijection",
                    "workflow_id": "wf-async-bijection",
                    "workflow_node_id": "node-async-bijection",
                    "run_id": "run-async-bijection",
                    "token_id": "tok-async-bijection",
                    "attempt": 1,
                    "step_seq": 1,
                    "state_view": {},
                },
            )()
        )
    )
    assert "noop" in resolver.nested_ops
    assert resolver.describe_state() == {"events": "a"}
    assert out.status == "success"
    assert out.state_update == [("u", {"ok": True})]


def test_async_runtime_trace_fast_path_configuration():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_trace_fast_path_configuration`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_trace_fast_path_configuration_matches_sync_runtime`.
    """

    class _Engine:
        backend_kind = "memory"

    rt = AsyncWorkflowRuntime(
        workflow_engine=_Engine(),
        conversation_engine=_Engine(),
        step_resolver=AsyncMappingStepResolver(),
        predicate_registry={},
        trace=True,
        experimental_native_scheduler=True,
    )

    assert rt.sync_runtime.fast_trace_persistence is True


def test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend(
    monkeypatch,
):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend`.
    """

    class _ConversationEngine:
        backend = object()

    monkeypatch.setattr(
        "kogwistar.engine_core.postgres_backend.PgVectorBackend",
        type("_PgVectorBackend", (), {}),
    )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=_ConversationEngine(),
        step_resolver=AsyncMappingStepResolver(),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )

    assert rt.sync_runtime.transaction_mode == "none"


def test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend(monkeypatch):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_auto_transaction_mode_uses_step_for_pg_backend`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend`.
    """

    class _PgVectorBackend:
        pass

    class _ConversationEngine:
        backend = _PgVectorBackend()

    monkeypatch.setattr(
        "kogwistar.engine_core.postgres_backend.PgVectorBackend",
        _PgVectorBackend,
    )

    rt = AsyncWorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=_ConversationEngine(),
        step_resolver=AsyncMappingStepResolver(),
        predicate_registry={},
        trace=False,
        experimental_native_scheduler=True,
    )

    assert rt.sync_runtime.transaction_mode == "step"


def test_async_runtime_missing_op_behavior():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_missing_op_behavior`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_missing_op_behavior_matches_sync_resolver`.
    """
    sync_resolver = MappingStepResolver()
    async_resolver = AsyncMappingStepResolver()

    with pytest.raises(KeyError):
        sync_resolver.resolve("missing")
    with pytest.raises(KeyError):
        asyncio.run(async_resolver.resolve_async("missing")(_DummyCtx()))


def test_async_runtime_exception_to_runfailure():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_exception_to_runfailure`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_exception_to_runfailure_matches_sync_resolver`.
    """
    sync_resolver = MappingStepResolver()
    async_resolver = AsyncMappingStepResolver()

    @sync_resolver.register("boom")
    def _boom_sync(_ctx):
        raise RuntimeError("sync boom")

    @async_resolver.register("boom")
    async def _boom_async(_ctx):
        raise RuntimeError("async boom")

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "state_view": {},
        },
    )()

    sync_out = sync_resolver.resolve("boom")(ctx)
    async_out = asyncio.run(async_resolver.resolve_async("boom")(ctx))
    assert isinstance(sync_out, RunFailure)
    assert isinstance(async_out, RunFailure)
    assert any("sync boom" in e for e in sync_out.errors)
    assert any("async boom" in e for e in async_out.errors)


def test_async_runtime_async_handler_callability():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_async_handler_callability`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_adapter_accepts_async_handler`.
    """
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("async_like")
    async def _async(_ctx):
        await asyncio.sleep(0.001)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"async_key": "ok"})],
        )

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "state_view": {},
        },
    )()

    out = asyncio.run(async_resolver.resolve_async("async_like")(ctx))
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"async_key": "ok"})]


def test_async_runtime_sync_handler_callability():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_sync_handler_callability`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_adapter_accepts_sync_handler`.
    """
    async_resolver = AsyncMappingStepResolver()

    @async_resolver.register("sync_like")
    def _sync(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "v"})])

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "state_view": {},
        },
    )()
    out = asyncio.run(async_resolver.resolve_async("sync_like")(ctx))
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"k": "v"})]


def test_async_runtime_registered_op_appears_in_op_set():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_registered_op_appears_in_op_set`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_registered_async_op_appears_in_async_op_set`.
    """
    resolver = AsyncMappingStepResolver()

    @resolver.register("op_set")
    async def _op(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "ok"})])

    assert "op_set" in resolver.ops
    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "state_view": {},
        },
    )()
    out = asyncio.run(resolver.resolve_async("op_set")(ctx))
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"k": "ok"})]


def test_async_runtime_preserves_sandboxed_behavior():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_preserves_sandboxed_behavior`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_preserves_sandboxed_behavior`.
    """
    resolver = AsyncMappingStepResolver()

    class _FakeSandbox:
        def run(self, code, state, context):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sandbox_code": code, "sandbox_ctx_op": context.get("op")})],
            )

    resolver.set_sandbox(_FakeSandbox())
    resolver.sandboxed_ops.add("py")

    @resolver.register("py")
    async def _sandboxed(_ctx):
        return "result = {'state_update': []}"

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "turn_node_id": "turn-async-bijection",
            "state_view": {},
            "op": "py",
        },
    )()
    out = asyncio.run(resolver.resolve_async("py")(ctx))
    assert isinstance(out, RunSuccess)
    assert out.state_update[0][1]["sandbox_ctx_op"] == "py"


def test_async_runtime_non_sandboxed_op_does_not_prepare_sandbox(tmp_path):
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_non_sandboxed_op_does_not_prepare_sandbox`.
    Moved from `tests/runtime/test_sandbox.py::test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op`.
    """
    class _FailIfCalledSandbox:
        def run(self, code, state, context):
            raise AssertionError("sandbox should not run for non-sandboxed ops")

        def close_run(self, run_id: str) -> None:
            return None

    resolver = AsyncMappingStepResolver()
    resolver.set_sandbox(_FailIfCalledSandbox())

    @resolver.register("normal_op")
    def _normal(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ok": True})])

    out = asyncio.run(
        resolver.resolve_async("normal_op")(
            type(
                "_Ctx",
                (),
                {
                    "conversation_id": "conv-async-bijection",
                    "workflow_id": "wf-async-bijection",
                    "workflow_node_id": "node-async-bijection",
                    "run_id": "run-async-bijection",
                    "token_id": "tok-async-bijection",
                    "attempt": 1,
                    "step_seq": 1,
                    "state_view": {"value": 1},
                },
            )()
        )
    )
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"ok": True})]


def test_async_runtime_legacy_update_warning_preserved():
    """Sync mirror: `tests/runtime/test_sync_runtime_bijection_contract.py::test_sync_runtime_legacy_update_warning_preserved`.
    Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_legacy_update_warning_preserved`.
    """
    import kogwistar.runtime.resolvers as runtime_resolvers

    runtime_resolvers._LEGACY_UPDATE_WARNING_EMITTED = False
    resolver = AsyncMappingStepResolver()

    @resolver.register("legacy")
    async def _legacy(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], update={"k": 1})

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-async-bijection",
            "workflow_id": "wf-async-bijection",
            "workflow_node_id": "node-async-bijection",
            "run_id": "run-async-bijection",
            "token_id": "tok-async-bijection",
            "attempt": 1,
            "step_seq": 1,
            "turn_node_id": "turn-async-bijection",
            "state_view": {},
        },
    )()
    with pytest.warns(RuntimeWarning, match="legacy update detected"):
        out = asyncio.run(resolver.resolve_async("legacy")(ctx))
    assert isinstance(out, RunSuccess)
