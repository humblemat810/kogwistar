from __future__ import annotations

import pytest

from kogwistar.conversation.resolvers import default_resolver
from kogwistar.runtime import MappingStepResolver
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime, apply_state_update_inplace

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
