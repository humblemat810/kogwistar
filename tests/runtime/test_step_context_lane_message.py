from __future__ import annotations

from kogwistar.runtime.runtime import StepContext


def test_step_context_send_lane_message_delegates_to_sender(tmp_path):
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


def test_step_context_send_lane_message_requires_sender(tmp_path):
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
