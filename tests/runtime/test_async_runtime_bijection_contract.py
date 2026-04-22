from __future__ import annotations

import asyncio

import pytest

from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime import AsyncMappingStepResolver, AsyncWorkflowRuntime, MappingStepResolver
from kogwistar.runtime.replay import _apply_state_update
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime

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
