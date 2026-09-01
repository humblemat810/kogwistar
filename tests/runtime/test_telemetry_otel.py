from __future__ import annotations

import time
import threading

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.runtime.models import RunSuccess, WorkflowInvocationRequest
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import StepContext
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.runtime.telemetry import EventEmitter, TraceContext
from kogwistar.runtime.telemetry_otel import (
    OpenTelemetrySink,
    opentelemetry_available,
    try_create_opentelemetry_sink,
)

pytestmark = [pytest.mark.unit, pytest.mark.runtime]


class _Span:
    def __init__(self, name, context, attributes):
        self.name = name
        self.context = context
        self.attributes = dict(attributes or {})
        self.events = []
        self.ended = False

    def add_event(self, name, attributes=None):
        self.events.append((name, dict(attributes or {})))

    def end(self):
        self.ended = True


class _Tracer:
    def __init__(self):
        self.spans = []
        self.fail = False

    def start_span(self, name, *, context=None, attributes=None):
        if self.fail:
            raise RuntimeError("export failed")
        span = _Span(name, context, attributes)
        self.spans.append(span)
        return span


class _BlockingTracer(_Tracer):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def start_span(self, name, *, context=None, attributes=None):
        self.started.set()
        self.release.wait(1.0)
        return super().start_span(name, context=context, attributes=attributes)


def _ctx(*, step_seq=1, node_id="node", attempt=1):
    return TraceContext(
        run_id="run-1",
        token_id="token-1",
        step_seq=step_seq,
        node_id=node_id,
        attempt=attempt,
        conversation_id="conversation-1",
        turn_node_id="turn-1",
    )


def _wait_until(predicate):
    deadline = time.monotonic() + 1.0
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_trace_context_can_create_and_validate_w3c_root_and_child():
    root = TraceContext.new_root(
        run_id="run-1", token_id="token-1", step_seq=0, node_id="start"
    )
    child = root.child_span(step_seq=1, node_id="step")

    assert root.has_valid_w3c_ids is True
    assert child.has_valid_w3c_ids is True
    assert child.trace_id == root.trace_id
    assert child.span_id != root.span_id
    assert child.parent_span_id == root.span_id
    assert root.require_valid_w3c_ids() is root


def test_trace_context_rejects_legacy_pseudo_ids_as_w3c_ids():
    legacy = _ctx()

    assert legacy.has_valid_w3c_ids is False
    try:
        legacy.require_valid_w3c_ids()
    except ValueError as exc:
        assert "valid W3C" in str(exc)
    else:
        raise AssertionError("legacy trace context must not be accepted as W3C")


def test_step_context_preserves_one_supplied_span_context():
    root = TraceContext.new_root(
        run_id="run-1", token_id="root", step_seq=0, node_id="start"
    )
    step_trace = root.child_span(token_id="token-1", step_seq=1, node_id="step")
    ctx = StepContext(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_node_id="step",
        op="noop",
        token_id="token-1",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        trace_context=step_trace,
    )

    assert ctx.trace_ctx is step_trace
    assert ctx.trace_ctx.trace_id == root.trace_id


def test_otel_optional_dependency_probe_is_safe():
    assert isinstance(opentelemetry_available(), bool)


def test_otel_factory_disables_cleanly_without_optional_dependency(monkeypatch):
    def _missing(cls, **kwargs):
        raise ImportError("opentelemetry is not installed")

    monkeypatch.setattr(OpenTelemetrySink, "from_opentelemetry", classmethod(_missing))
    assert try_create_opentelemetry_sink() is None


def test_otel_projects_workflow_and_child_step_spans():
    tracer = _Tracer()
    sink = OpenTelemetrySink(tracer, context_factory=lambda parent: ("child-of", parent))
    emitter = EventEmitter(sink=sink)

    emitter.emit(
        type="workflow_run_started",
        ctx=_ctx(node_id="start"),
        payload={"workflow_id": "workflow-1"},
    )
    emitter.step_started(_ctx())
    emitter.step_completed(_ctx(), status="ok", duration_ms=7)
    emitter.emit(type="checkpoint_saved", ctx=_ctx())
    emitter.emit(type="workflow_run_completed", ctx=_ctx(node_id="run", step_seq=2))
    assert sink.flush(1.0)
    sink.close()

    run, step = tracer.spans
    assert run.name == "kogwistar.workflow.run"
    assert step.name == "kogwistar.workflow.step_attempt"
    assert step.context == ("child-of", run)
    assert run.attributes["kogwistar.run_id"] == "run-1"
    assert run.attributes["kogwistar.workflow_id"] == "workflow-1"
    assert step.attributes["kogwistar.node_id"] == "node"
    assert any(event[0] == "checkpoint_saved" for event in run.events)
    assert run.ended and step.ended
    assert not sink.is_alive


def test_otel_keeps_kogwistar_w3c_ids_as_attributes():
    tracer = _Tracer()
    sink = OpenTelemetrySink(tracer)
    root = TraceContext.new_root(
        run_id="run-1", token_id="root", step_seq=0, node_id="start"
    )
    event = {"type": "workflow_run_started", **root.as_fields()}
    event["payload_json"] = '{"workflow_id":"workflow-1"}'
    sink.emit(event)
    assert sink.flush(1.0)
    sink.close()

    assert tracer.spans[0].attributes["kogwistar.trace_id"] == root.trace_id
    assert tracer.spans[0].attributes["kogwistar.span_id"] == root.span_id


def test_otel_queue_drops_and_export_failure_isolated():
    tracer = _BlockingTracer()
    sink = OpenTelemetrySink(tracer, queue_max=1)
    sink.emit({"type": "workflow_run_started", "run_id": "r"})
    assert tracer.started.wait(1.0)
    sink.emit({"type": "workflow_run_started", "run_id": "r2"})
    sink.emit({"type": "workflow_run_started", "run_id": "r3"})
    assert sink.dropped_events == 1
    tracer.release.set()
    assert sink.flush(1.0)
    sink.close()
    assert not sink.is_alive


def test_otel_closes_run_and_open_steps_on_cancel_or_suspend():
    for terminal_event in ("workflow_run_cancelled", "workflow_run_suspended"):
        tracer = _Tracer()
        sink = OpenTelemetrySink(tracer)
        emitter = EventEmitter(sink=sink)
        emitter.emit(
            type="workflow_run_started",
            ctx=_ctx(node_id="start"),
            payload={"workflow_id": "workflow-1"},
        )
        emitter.step_started(_ctx(node_id="running"))
        emitter.emit(type=terminal_event, ctx=_ctx(node_id="run", step_seq=2))
        assert sink.flush(1.0)
        sink.close()

        assert len(tracer.spans) == 2
        assert all(span.ended for span in tracer.spans)
        assert tracer.spans[0].events[-1][0] == terminal_event

    tracer = _Tracer()
    tracer.fail = True
    sink = OpenTelemetrySink(tracer)
    EventEmitter(sink=sink).emit(type="workflow_run_started", ctx=_ctx())
    _wait_until(lambda: sink.export_errors == 1)
    sink.close()
    assert not sink.is_alive


@pytest.mark.e2e
def test_real_workflow_runtime_projects_lifecycle_to_otel_sink(tmp_path):
    """The adapter is exercised through WorkflowRuntime, not only EventEmitter."""
    from tests.runtime.test_trace_sink_smoke import _add_edge, _add_node

    workflow_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "workflow"), kg_graph_type="workflow"
    )
    conversation_engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"), kg_graph_type="conversation"
    )
    tracer = _Tracer()
    sink = OpenTelemetrySink(tracer, context_factory=lambda parent: ("child-of", parent))
    resolver = MappingStepResolver()
    workflow_id = "wf_otel_runtime"

    try:
        _add_node(
            workflow_engine,
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|start",
            op="start",
            start=True,
        )
        _add_node(
            workflow_engine,
            workflow_id=workflow_id,
            node_id=f"wf|{workflow_id}|done",
            op="done",
            terminal=True,
        )
        _add_edge(
            workflow_engine,
            workflow_id=workflow_id,
            edge_id=f"wf|{workflow_id}|start->done",
            src=f"wf|{workflow_id}|start",
            dst=f"wf|{workflow_id}|done",
        )

        @resolver.register("start")
        def _start(ctx):
            return RunSuccess(state_update=[("u", {"started": True})])

        @resolver.register("done")
        def _done(ctx):
            return RunSuccess(state_update=[("u", {"done": True})])

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            sink=sink,
        )
        result = runtime.run(
            workflow_id=workflow_id,
            conversation_id="conv-otel-runtime",
            turn_node_id="turn-otel-runtime",
            run_id="run-otel-runtime",
            initial_state={},
        )
        assert result.status == "succeeded"
        assert sink.flush(1.0)
        assert [span.name for span in tracer.spans] == [
            "kogwistar.workflow.run",
            "kogwistar.workflow.step_attempt",
            "kogwistar.workflow.step_attempt",
        ]
        assert tracer.spans[0].ended
        assert all(span.ended for span in tracer.spans[1:])
        assert any(event[0] == "checkpoint_saved" for event in tracer.spans[0].events)
    finally:
        sink.close()
        workflow_engine.close()
        conversation_engine.close()


@pytest.mark.e2e
def test_otel_resume_uses_same_trace_and_new_continuation_span():
    """Restarted execution keeps W3C trace identity without reusing a span."""
    tracer = _Tracer()
    sink = OpenTelemetrySink(tracer)
    root = TraceContext.new_root(
        run_id="run-resume", token_id="token", step_seq=0, node_id="start"
    )
    first = {"type": "workflow_run_started", **root.as_fields()}
    continuation = root.child_span(step_seq=4, node_id="resume")
    second = {"type": "workflow_run_started", **continuation.as_fields()}
    sink.emit(first)
    sink.emit({"type": "workflow_run_completed", **root.as_fields()})
    assert sink.flush(1.0)
    sink.emit(second)
    sink.emit({"type": "workflow_run_completed", **continuation.as_fields()})
    assert sink.flush(1.0)
    sink.close()

    assert len(tracer.spans) == 2
    assert tracer.spans[0].attributes["kogwistar.trace_id"] == root.trace_id
    assert tracer.spans[1].attributes["kogwistar.trace_id"] == root.trace_id
    assert tracer.spans[0].attributes["kogwistar.span_id"] != tracer.spans[1].attributes[
        "kogwistar.span_id"
    ]


@pytest.mark.e2e
def test_nested_workflow_runtime_projects_child_run_under_parent_span(tmp_path):
    """Nested invoke-and-await keeps trace identity and OTel parentage."""
    from tests.runtime.test_workflow_invocation_and_route_next import (
        _build_engine_pair,
        _edge,
        _node,
    )

    workflow_engine, conversation_engine = _build_engine_pair(tmp_path)
    tracer = _Tracer()
    sink = OpenTelemetrySink(
        tracer, context_factory=lambda parent: ("child-of", parent)
    )
    resolver = MappingStepResolver()
    parent_id = "wf_otel_parent"
    child_id = "wf_otel_child"
    try:
        for node in (
            _node(workflow_id=parent_id, node_id="parent-start", op="start", start=True),
            _node(workflow_id=parent_id, node_id="parent-act", op="act"),
            _node(workflow_id=parent_id, node_id="parent-done", op="done", terminal=True),
            _node(workflow_id=child_id, node_id="child-start", op="child", start=True),
            _node(workflow_id=child_id, node_id="child-done", op="child-done", terminal=True),
        ):
            workflow_engine.write.add_node(node)
        for edge in (
            _edge(workflow_id=parent_id, edge_id="parent-1", src="parent-start", dst="parent-act"),
            _edge(workflow_id=parent_id, edge_id="parent-2", src="parent-act", dst="parent-done"),
            _edge(workflow_id=child_id, edge_id="child-1", src="child-start", dst="child-done"),
        ):
            workflow_engine.write.add_edge(edge)

        @resolver.register("start")
        def _start(ctx):
            return RunSuccess(state_update=[])

        @resolver.register("act")
        def _act(ctx):
            return RunSuccess(
                state_update=[],
                workflow_invocations=[
                    WorkflowInvocationRequest(
                        workflow_id=child_id,
                        invocation_key="otel-child-action",
                        result_state_key="child_result",
                    )
                ],
            )

        @resolver.register("done")
        def _done(ctx):
            return RunSuccess(state_update=[])

        @resolver.register("child")
        def _child(ctx):
            return RunSuccess(state_update=[])

        @resolver.register("child-done")
        def _child_done(ctx):
            return RunSuccess(state_update=[])

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=resolver,
            predicate_registry={},
            checkpoint_every_n_steps=1,
            sink=sink,
        )
        result = runtime.run(
            workflow_id=parent_id,
            conversation_id="conv-otel-nested",
            turn_node_id="turn-otel-nested",
            run_id="run-otel-parent",
            initial_state={},
            _trace_context=TraceContext.new_root(
                run_id="run-otel-parent",
                token_id="run-otel-parent",
                step_seq=0,
                node_id="start",
            ),
        )
        assert result.status == "succeeded"
        assert sink.flush(1.0)

        run_spans = [span for span in tracer.spans if span.name == "kogwistar.workflow.run"]
        assert len(run_spans) == 2
        parent_span, child_span = run_spans
        parent_step = next(
            span
            for span in tracer.spans
            if span.name == "kogwistar.workflow.step_attempt"
            and span.attributes.get("kogwistar.node_id") == "parent-act"
        )
        assert child_span.attributes["kogwistar.trace_id"] == parent_span.attributes[
            "kogwistar.trace_id"
        ]
        assert child_span.context == ("child-of", parent_step)
    finally:
        sink.close()
        workflow_engine.close()
        conversation_engine.close()
