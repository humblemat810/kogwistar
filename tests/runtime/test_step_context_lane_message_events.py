from __future__ import annotations

from kogwistar.runtime.runtime import StepContext


def test_step_context_emit_lane_message_event_delegates_to_sink(tmp_path):
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


def test_step_context_emit_lane_message_event_requires_sink(tmp_path):
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
