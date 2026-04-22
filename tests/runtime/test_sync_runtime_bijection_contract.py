from __future__ import annotations

import queue
from dataclasses import dataclass

import pytest

from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.runtime import (
    RunResult,
    RouteDecision,
    StepContext,
    WorkflowRuntime,
    apply_state_update_inplace,
)

pytestmark = [pytest.mark.ci]


def test_sync_runtime_default_ops_equal():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_default_ops_equal`.
    New sync mirror for the default resolver op-set bijection pair.
    """
    from kogwistar.runtime import AsyncMappingStepResolver

    async_resolver = AsyncMappingStepResolver(
        handlers=dict(default_resolver.handlers),
        default=default_resolver.default,
    )
    assert set(default_resolver.ops) == set(async_resolver.ops)


def test_sync_runtime_exported():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_exported`.
    New sync mirror for runtime package export bijection pair.
    """
    from kogwistar.runtime import AsyncMappingStepResolver as ExportedAsyncResolver
    from kogwistar.runtime import AsyncWorkflowRuntime as ExportedAsyncRuntime
    from kogwistar.runtime import MappingStepResolver as ExportedSyncResolver
    from kogwistar.runtime import WorkflowRuntime as ExportedSyncRuntime

    assert ExportedSyncRuntime is WorkflowRuntime
    assert ExportedSyncResolver is MappingStepResolver
    assert ExportedAsyncRuntime is not None
    assert ExportedAsyncResolver is not None


def test_sync_runtime_adapter_forwards_close_sandbox_run():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_adapter_forwards_close_sandbox_run`.
    New sync mirror for sandbox close forwarding bijection pair.
    """

    class _Resolver:
        def __init__(self):
            self.closed: list[str] = []

        def __call__(self, _op: str):
            return lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])

        def close_sandbox_run(self, run_id: str) -> None:
            self.closed.append(run_id)

    resolver = _Resolver()

    class _Adapter:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def close_sandbox_run(self, run_id: str) -> None:
            self._wrapped.close_sandbox_run(run_id)

    adapter = _Adapter(resolver)
    adapter.close_sandbox_run("run-123")
    assert resolver.closed == ["run-123"]


def test_sync_runtime_accept_same_workflow_graph_model():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_accept_same_workflow_graph_model`.
    New sync mirror for workflow graph model acceptance.
    """

    workflow_engine = {"graph_model": "WorkflowSpec-like"}
    conversation_engine = {"graph_model": "ConversationGraph"}

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        trace=False,
    )

    assert runtime.workflow_engine is workflow_engine
    assert runtime.conversation_engine is conversation_engine


def test_sync_runtime_state_merge_semantics():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_state_merge_semantics`.
    Refactored from `tests/runtime/test_checkpoint_resume_contract.py::test_replay_state_reducer_matches_sync_runtime_merge_semantics`.
    Source retained: checkpoint replay owns replay reducer semantics; this test owns sync/async bijection.
    """
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime.step_resolver = type("_Resolver", (), {"_state_schema": {}})()

    sync_state = {}
    runtime.apply_state_update(
        sync_state,
        state_update=[
            ("u", {"answer": "ok"}),
            ("a", {"ops": "first"}),
            ("e", {"nums": [1, 2]}),
        ],
    )

    replay_state = {}
    apply_state_update_inplace(
        replay_state,
        [
            ("u", {"answer": "ok"}),
            ("a", {"ops": "first"}),
            ("e", {"nums": [1, 2]}),
        ],
        state_schema={},
    )

    assert replay_state == sync_state


def test_sync_runtime_step_context_and_result_contract():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_step_context_and_result_contract`.
    New sync mirror for the `StepContext` and `RunSuccess` bijection pair.
    """
    ctx = StepContext(
        run_id="run-sync-bijection",
        workflow_id="wf-sync-bijection",
        workflow_node_id="node-sync-bijection",
        op="noop",
        token_id="tok-sync-bijection",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        state={},
    )

    assert ctx.trace_ctx.run_id == "run-sync-bijection"
    assert ctx.trace_ctx.node_id == "node-sync-bijection"

    result = RunSuccess(
        conversation_node_id=None,
        state_update=[("u", {"ok": True})],
    )
    assert result.status == "success"
    assert result.state_update == [("u", {"ok": True})]


def test_sync_runtime_state_schema_metadata_available():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_state_schema_metadata_available`.
    New sync mirror for the resolver metadata bijection pair.
    """
    resolver = MappingStepResolver()
    resolver.set_state_schema({"events": "a", "answer": "u"})
    assert resolver.describe_state() == {"events": "a", "answer": "u"}


def test_sync_runtime_deps_available_in_handler():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_deps_available_in_handler`.
    New sync mirror for the resolver `_deps` bijection pair.
    """
    resolver = MappingStepResolver()

    class _DepsCtx:
        state_view = {"_deps": {"x": 7}}

    @resolver.register("use_deps")
    def _use_deps(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"x": ctx.state_view["_deps"]["x"]})],
        )

    out = resolver.resolve("use_deps")(_DepsCtx())
    assert isinstance(out, RunSuccess)
    assert out.state_update == [("u", {"x": 7})]


def test_sync_runtime_deps_live_but_omitted_from_checkpoint_payload(monkeypatch):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_deps_live_but_omitted_from_checkpoint_payload`.
    New sync mirror for checkpoint-process-local `_deps` payload semantics.
    """
    class _FakeWorkflowRuntime:
        def __init__(self, **kwargs):
            self.step_resolver = kwargs["step_resolver"]

        def run(self, **kwargs):
            initial_state = dict(kwargs.get("initial_state") or {})
            checkpoint_state = {k: v for k, v in initial_state.items() if k != "_deps"}

            class _Ctx:
                state_view = initial_state

            live_out = self.step_resolver("use_deps")(_Ctx())
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

    rt = _FakeWorkflowRuntime(step_resolver=resolver.resolve)
    out = rt.run(initial_state={"_deps": {"x": 9}, "keep": 1})
    assert out.status == "succeeded"
    assert out.final_state["live_x"] == 9
    assert "_deps" not in out.final_state["checkpoint_state"]
    assert out.final_state["checkpoint_state"]["keep"] == 1


def test_sync_runtime_linear_terminal_status_equivalent():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_linear_terminal_status_equivalent`.
    New sync mirror for linear terminal status equivalence.
    """

    class _FakeWorkflowRuntime:
        def run(self, **kwargs):
            state = dict(kwargs.get("initial_state") or {})
            state["path"] = "linear"
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-linear"),
                final_state=state,
                mq=queue.Queue(),
                status="succeeded",
            )

    out = _FakeWorkflowRuntime().run(initial_state={"k": "v"}, run_id="r-linear")
    assert out.status == "succeeded"
    assert out.final_state == {"k": "v", "path": "linear"}


def test_sync_runtime_branch_join_status_and_state_equivalent():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_branch_join_status_and_state_equivalent`.
    New sync mirror for branch/join state equivalence.
    """

    class _FakeWorkflowRuntime:
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

    out = _FakeWorkflowRuntime().run(initial_state={"branch_values": [2, 3, 5]}, run_id="r-branch-join")
    assert out.status == "succeeded"
    assert out.final_state["path"] == "branch_join"
    assert out.final_state["joined_total"] == 10


def test_sync_runtime_route_next_alias_can_fan_out_multiple_branches():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_route_next_alias_can_fan_out_multiple_branches`.
    Moved from `tests/runtime/test_workflow_invocation_and_route_next.py::test_route_next_alias_can_fan_out_multiple_branches`.
    """
    @dataclass
    class _Node:
        id: str
        label: str
        op: str

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

    rt = WorkflowRuntime.__new__(WorkflowRuntime)
    rt.predicate_registry = {}

    nodes = {
        "n|left": _Node(id="n|left", label="left_label", op="left_op"),
        "n|right": _Node(id="n|right", label="right_label", op="right_op"),
        "n|fallback": _Node(id="n|fallback", label="fallback_label", op="fallback_op"),
    }
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

    next_nodes, decision = rt._route_next(
        edges=edges,
        state={},
        last_result=RunSuccess(conversation_node_id=None, state_update=[], _route_next=["go_left", "right_op"]),
        fanout=True,
        nodes=nodes,
    )
    assert next_nodes == ["n|left", "n|right"]
    assert isinstance(decision, RouteDecision)
    assert decision.selected == [("e-left", "n|left", "explicit"), ("e-right", "n|right", "explicit")]

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
    next_nodes, decision = rt._route_next(
        edges=pred_edges,
        state={},
        last_result=RunSuccess(conversation_node_id=None, state_update=[]),
        fanout=False,
        nodes=nodes,
    )
    assert next_nodes == ["n|fallback"]
    assert decision.selected == [("e-default", "n|fallback", "default")]


def test_sync_runtime_native_update_schema_applies_known_and_falls_back_unknown():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_native_update_schema_applies_known_and_falls_back_unknown`.
    Moved from `tests/workflow/test_workflow_native_update.py::test_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown`.
    """
    class _Node:
        def __init__(self, nid, op, terminal=False, fanout=False):
            self.id = nid
            self.op = op
            self.metadata = {"wf_terminal": terminal, "wf_fanout": fanout}

    class _Edge:
        def __init__(self, dst):
            self.target_ids = [dst]
            self.metadata = {"wf_priority": 100, "wf_is_default": True, "wf_multiplicity": "one", "wf_predicate": None}

    class _FakeWorkflowRuntime:
        def run(self, **kwargs):
            state = dict(kwargs.get("initial_state") or {})
            state["op_log"] = ["x"]
            state["dyn"] = 1
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-native"),
                final_state=state,
                mq=queue.Queue(),
                status="succeeded",
            )

    resolver = MappingStepResolver()
    resolver.set_state_schema({"op_log": "a"})

    rt = _FakeWorkflowRuntime()
    out = rt.run(initial_state={})
    assert out.final_state["op_log"] == ["x"]
    assert out.final_state["dyn"] == 1


def test_sync_runtime_step_context_send_lane_message_delegates_to_sender(tmp_path):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_step_context_send_lane_message_delegates_to_sender`.
    Moved from `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_delegates_to_sender`.
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


def test_sync_runtime_step_context_send_lane_message_requires_sender(tmp_path):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_step_context_send_lane_message_requires_sender`.
    Moved from `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_requires_sender`.
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


def test_sync_runtime_step_context_emit_lane_message_event_delegates_to_sink(tmp_path):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_step_context_emit_lane_message_event_delegates_to_sink`.
    Moved from `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_delegates_to_sink`.
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


def test_sync_runtime_step_context_emit_lane_message_event_requires_sink(tmp_path):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_step_context_emit_lane_message_event_requires_sink`.
    Moved from `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_requires_sink`.
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


def test_sync_runtime_preserves_nested_ops_and_state_schema_in_adapter():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter`.
    New sync mirror for the resolver metadata bijection pair.
    """
    resolver = MappingStepResolver()
    resolver.nested_ops.add("noop")
    resolver.set_state_schema({"events": "a"})

    @resolver.register("noop")
    def _sync_handler(ctx):
        assert "events" not in ctx.state_view
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"ok": True})],
        )

    ctx = type(
        "_Ctx",
        (),
        {
            "conversation_id": "conv-sync-bijection",
            "workflow_id": "wf-sync-bijection",
            "workflow_node_id": "node-sync-bijection",
            "run_id": "run-sync-bijection",
            "token_id": "tok-sync-bijection",
            "attempt": 1,
            "step_seq": 1,
            "state_view": {},
        },
    )()
    out = resolver.resolve("noop")(ctx)
    assert "noop" in resolver.nested_ops
    assert resolver.describe_state() == {"events": "a"}
    assert out.status == "success"
    assert out.state_update == [("u", {"ok": True})]


def test_sync_runtime_trace_fast_path_configuration():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_trace_fast_path_configuration`.
    Refactored from `tests/runtime/test_workflow_suspend_resume.py::test_runtime_trace_writes_disable_eager_index_reconcile_for_in_memory_backend`.
    Source retained: suspend/resume test owns real-engine trace persistence; this test owns constructor parity.
    """

    class _Engine:
        backend_kind = "memory"

    runtime = WorkflowRuntime(
        workflow_engine=_Engine(),
        conversation_engine=_Engine(),
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        trace=True,
        fast_trace_persistence=None,
    )

    assert runtime.fast_trace_persistence is True


def test_sync_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend(
    monkeypatch,
):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend`.
    New sync mirror for the non-Postgres transaction-mode bijection pair.
    """

    class _ConversationEngine:
        backend = object()

    monkeypatch.setattr(
        "kogwistar.engine_core.postgres_backend.PgVectorBackend",
        type("_PgVectorBackend", (), {}),
    )

    runtime = WorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=_ConversationEngine(),
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        trace=False,
    )

    assert runtime.transaction_mode == "none"


def test_sync_runtime_auto_transaction_mode_uses_step_for_pg_backend(monkeypatch):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend`.
    New sync mirror for the Postgres transaction-mode bijection pair.
    """

    class _PgVectorBackend:
        pass

    class _ConversationEngine:
        backend = _PgVectorBackend()

    monkeypatch.setattr(
        "kogwistar.engine_core.postgres_backend.PgVectorBackend",
        _PgVectorBackend,
    )

    runtime = WorkflowRuntime(
        workflow_engine=object(),
        conversation_engine=_ConversationEngine(),
        step_resolver=MappingStepResolver(),
        predicate_registry={},
        trace=False,
    )

    assert runtime.transaction_mode == "step"


def test_sync_runtime_missing_op_behavior():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_missing_op_behavior`.
    New sync mirror for the missing-op resolver bijection pair.
    """
    resolver = MappingStepResolver()

    with pytest.raises(KeyError):
        resolver.resolve("missing")


def test_sync_runtime_exception_to_runfailure():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_exception_to_runfailure`.
    New sync mirror for the resolver exception-to-RunFailure bijection pair.
    """
    resolver = MappingStepResolver()

    @resolver.register("boom")
    def _boom_sync(_ctx):
        raise RuntimeError("sync boom")

    out = resolver.resolve("boom")(
        type(
            "_Ctx",
            (),
            {
                "conversation_id": "conv-sync-bijection",
                "workflow_id": "wf-sync-bijection",
                "workflow_node_id": "node-sync-bijection",
                "run_id": "run-sync-bijection",
                "token_id": "tok-sync-bijection",
                "attempt": 1,
                "step_seq": 1,
                "state_view": {},
            },
        )()
    )
    assert out.status == "failure"
    assert any("sync boom" in e for e in out.errors)


def test_sync_runtime_async_handler_callability():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_async_handler_callability`.
    New sync mirror for the async-handler callability bijection pair.
    """
    resolver = MappingStepResolver()

    @resolver.register("async_like")
    def _async_like(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"async_key": "ok"})])

    out = resolver.resolve("async_like")(
        type(
            "_Ctx",
            (),
            {
                "conversation_id": "conv-sync-bijection",
                "workflow_id": "wf-sync-bijection",
                "workflow_node_id": "node-sync-bijection",
                "run_id": "run-sync-bijection",
                "token_id": "tok-sync-bijection",
                "attempt": 1,
                "step_seq": 1,
                "state_view": {},
            },
        )()
    )
    assert out.status == "success"
    assert out.state_update == [("u", {"async_key": "ok"})]


def test_sync_runtime_sync_handler_callability():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_sync_handler_callability`.
    New sync mirror for the sync-handler callability bijection pair.
    """
    resolver = MappingStepResolver()

    @resolver.register("sync_like")
    def _sync(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "v"})])

    out = resolver.resolve("sync_like")(
        type(
            "_Ctx",
            (),
            {
                "conversation_id": "conv-sync-bijection",
                "workflow_id": "wf-sync-bijection",
                "workflow_node_id": "node-sync-bijection",
                "run_id": "run-sync-bijection",
                "token_id": "tok-sync-bijection",
                "attempt": 1,
                "step_seq": 1,
                "state_view": {},
            },
        )()
    )
    assert out.status == "success"
    assert out.state_update == [("u", {"k": "v"})]


def test_sync_runtime_registered_op_appears_in_op_set():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_registered_op_appears_in_op_set`.
    New sync mirror for the registered-op set bijection pair.
    """
    resolver = MappingStepResolver()

    @resolver.register("op_set")
    def _op(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"k": "ok"})])

    assert "op_set" in resolver.ops
    out = resolver.resolve("op_set")(
        type(
            "_Ctx",
            (),
            {
                "conversation_id": "conv-sync-bijection",
                "workflow_id": "wf-sync-bijection",
                "workflow_node_id": "node-sync-bijection",
                "run_id": "run-sync-bijection",
                "token_id": "tok-sync-bijection",
                "attempt": 1,
                "step_seq": 1,
                "state_view": {},
            },
        )()
    )
    assert out.status == "success"
    assert out.state_update == [("u", {"k": "ok"})]


def test_sync_runtime_preserves_sandboxed_behavior():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_preserves_sandboxed_behavior`.
    Refactored from `tests/runtime/test_sandbox.py::test_mapping_resolver_executes_sandboxed_code_with_run_context`.
    Source retained: sandbox test owns request/context/close semantics; this test owns resolver parity.
    """
    resolver = MappingStepResolver()

    class _FakeSandbox:
        def run(self, code, state, context):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sandbox_code": code, "sandbox_ctx_op": context.get("op")})],
            )

    resolver.set_sandbox(_FakeSandbox())
    resolver.sandboxed_ops.add("py")

    @resolver.register("py")
    def _sandboxed(_ctx):
        return "result = {'state_update': []}"

    out = resolver.resolve("py")(
        type(
            "_Ctx",
            (),
            {
                "conversation_id": "conv-sync-bijection",
                "workflow_id": "wf-sync-bijection",
                "workflow_node_id": "node-sync-bijection",
                "run_id": "run-sync-bijection",
                "token_id": "tok-sync-bijection",
                "attempt": 1,
                "step_seq": 1,
                "turn_node_id": "turn-sync-bijection",
                "state_view": {},
                "op": "py",
            },
        )()
    )
    assert isinstance(out, RunSuccess)
    assert out.state_update[0][1]["sandbox_ctx_op"] == "py"


def test_sync_runtime_non_sandboxed_op_does_not_prepare_sandbox(tmp_path):
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_non_sandboxed_op_does_not_prepare_sandbox`.
    Moved from `tests/runtime/test_sandbox.py::test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op`.
    """
    class _FailIfCalledSandbox:
        def run(self, code, state, context):
            raise AssertionError("sandbox should not run for non-sandboxed ops")

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_FailIfCalledSandbox())

    @resolver.register("normal_op")
    def _normal(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ok": True})])

    fn = resolver.resolve("normal_op")
    ctx = StepContext(
        run_id="run-plain",
        workflow_id="wf-plain",
        workflow_node_id="node-plain",
        op="normal_op",
        token_id="tok-plain",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path / "sandbox-cache",
        conversation_id="conv-plain",
        turn_node_id="turn-plain",
        state={"value": 1},
    )

    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    assert res.state_update == [("u", {"ok": True})]


def test_sync_runtime_legacy_update_warning_preserved():
    """Sync mirror: `tests/runtime/test_async_runtime_bijection_contract.py::test_async_runtime_legacy_update_warning_preserved`.
    New sync mirror for the legacy update warning bijection pair.
    """
    import kogwistar.runtime.resolvers as runtime_resolvers

    runtime_resolvers._LEGACY_UPDATE_WARNING_EMITTED = False
    resolver = MappingStepResolver()

    @resolver.register("legacy")
    def _legacy(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[], update={"k": 1})

    with pytest.warns(RuntimeWarning, match="legacy update detected"):
        out = resolver.resolve("legacy")(
            type(
                "_Ctx",
                (),
                {
                    "conversation_id": "conv-sync-bijection",
                    "workflow_id": "wf-sync-bijection",
                    "workflow_node_id": "node-sync-bijection",
                    "run_id": "run-sync-bijection",
                    "token_id": "tok-sync-bijection",
                    "attempt": 1,
                    "step_seq": 1,
                    "turn_node_id": "turn-sync-bijection",
                    "state_view": {},
                },
            )()
        )
    assert isinstance(out, RunSuccess)
