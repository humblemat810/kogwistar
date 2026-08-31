from __future__ import annotations

import queue
from types import SimpleNamespace

from kogwistar.runtime.models import WorkflowInvocationRequest
from kogwistar.runtime.runtime import RunResult, WorkflowRuntime
from kogwistar.runtime.telemetry import TraceContext


def test_nested_invocation_key_is_part_of_deterministic_child_run_identity() -> None:
    common = {
        "conversation_id": "conversation-1",
        "turn_node_id": "turn-1",
        "parent_run_id": "parent-run-1",
    }
    first = WorkflowRuntime._workflow_invocation_plan(
        invocation=WorkflowInvocationRequest(
            workflow_id="child-workflow", result_state_key="child", invocation_key="act-1"
        ),
        **common,
    )
    repeated = WorkflowRuntime._workflow_invocation_plan(
        invocation=WorkflowInvocationRequest(
            workflow_id="child-workflow", result_state_key="child", invocation_key="act-1"
        ),
        **common,
    )
    next_action = WorkflowRuntime._workflow_invocation_plan(
        invocation=WorkflowInvocationRequest(
            workflow_id="child-workflow", result_state_key="child", invocation_key="act-2"
        ),
        **common,
    )

    assert first["child_run_id"] == repeated["child_run_id"]
    assert first["child_run_id"] != next_action["child_run_id"]


def test_nested_invocation_without_key_preserves_existing_identity() -> None:
    invocation = WorkflowInvocationRequest(
        workflow_id="child-workflow", result_state_key="child"
    )
    plan = WorkflowRuntime._workflow_invocation_plan(
        invocation=invocation,
        conversation_id="conversation-1",
        turn_node_id="turn-1",
        parent_run_id="parent-run-1",
    )
    assert plan["child_run_id"]


def test_resume_loads_persisted_w3c_trace_as_continuation_parent(monkeypatch) -> None:
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    latest = SimpleNamespace(
        metadata={
            "workflow_id": "workflow-1",
            "state_json": {"value": 1},
            "step_seq": 3,
            "trace_id": "0123456789abcdef0123456789abcdef",
            "run_execution_span_id": "0123456789abcdef",
        }
    )
    monkeypatch.setattr(runtime, "_latest_checkpoint_for_run", lambda **_: latest)
    captured = {}

    def _run(**kwargs):
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(runtime, "run", _run)
    assert runtime.resume_from_latest_checkpoint(
        run_id="run-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
        turn_node_id="turn-1",
    ) == "result"

    context = captured["_trace_context"]
    assert isinstance(context, TraceContext)
    assert context.has_valid_w3c_ids is True
    assert context.trace_id == latest.metadata["trace_id"]
    assert context.span_id == latest.metadata["run_execution_span_id"]


def test_supplied_trace_context_is_persisted_on_workflow_run(monkeypatch) -> None:
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    captured = {}

    class _Read:
        def get_nodes(self, **kwargs):
            return []

    class _Write:
        def add_node(self, node):
            captured.update(node.metadata)

    runtime.conversation_engine = SimpleNamespace(
        read=_Read(), write=_Write()
    )
    runtime._trace_write_mode = lambda: __import__("contextlib").nullcontext()
    context = TraceContext.new_root(
        run_id="parent", token_id="token", step_seq=0, node_id="start"
    ).child_span(node_id="workflow")

    runtime._persist_workflow_run(
        conversation_id="conversation-1",
        workflow_id="workflow-1",
        run_id="run-1",
        turn_node_id="turn-1",
        status="running",
        trace_context=context,
    )

    assert captured["trace_id"] == context.trace_id
    assert captured["run_execution_span_id"] == context.span_id
    assert captured["parent_span_id"] == context.parent_span_id


def test_nested_invocation_resumes_existing_child_checkpoint(monkeypatch) -> None:
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    checkpoint = SimpleNamespace(metadata={"step_seq": 4})
    calls: list[str] = []
    runtime._terminal_run_result = lambda **_: None
    runtime._latest_checkpoint_for_run = lambda **_: checkpoint
    runtime._persist_workflow_invocation_lineage = lambda **_: True
    runtime.resume_from_latest_checkpoint = lambda **kwargs: calls.append(
        f"resume:{kwargs['run_id']}"
    ) or RunResult(
        final_state={}, run_id="child-run", mq=queue.Queue(), status="succeeded"
    )
    runtime._workflow_invocation_plan = staticmethod(
        lambda **_: {
            "child_run_id": "child-run",
            "conversation_id": "conversation-1",
            "turn_node_id": "turn-1",
            "result_state_key": "child",
        }
    )
    invocation = WorkflowInvocationRequest(
        workflow_id="child-workflow", result_state_key="child", invocation_key="act-1"
    )

    result = runtime._run_workflow_invocation(
        invocation=invocation,
        parent_state={},
        conversation_id="conversation-1",
        turn_node_id="turn-1",
        parent_run_id="parent-run",
    )

    assert result.run_id == "child-run"
    assert calls == ["resume:child-run"]


def test_nested_invocation_passes_parent_trace_context_to_new_child() -> None:
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime._child_workflow_initial_state = lambda **_: {}
    runtime._terminal_run_result = lambda **_: None
    runtime._latest_checkpoint_for_run = lambda **_: None
    runtime._persist_workflow_run = lambda **_: None
    runtime._persist_workflow_invocation_lineage = lambda **_: True
    captured = {}

    def _run(**kwargs):
        captured.update(kwargs)
        return RunResult(
            final_state={}, run_id=kwargs["run_id"], mq=queue.Queue(), status="succeeded"
        )

    runtime.run = _run
    invocation = WorkflowInvocationRequest(
        workflow_id="child-workflow", result_state_key="child", invocation_key="act-1"
    )
    parent = TraceContext.new_root(
        run_id="parent", token_id="token", step_seq=1, node_id="act"
    )

    runtime._run_workflow_invocation(
        invocation=invocation,
        parent_state={},
        conversation_id="conversation-1",
        turn_node_id="turn-1",
        parent_run_id="parent",
        parent_trace_context=parent,
    )

    assert captured["_trace_context"] is parent
