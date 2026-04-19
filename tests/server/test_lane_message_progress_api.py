from __future__ import annotations

import pytest


pytestmark = pytest.mark.server


def test_lane_message_progress_api_exposes_run_and_conversation_events(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    from kogwistar.server.chat_service import ChatRunService
    from kogwistar.server.run_registry import RunRegistry

    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=RunRegistry(workflow_engine.meta_sqlite),
    )
    conversation_engine.send_lane_message(
        conversation_id="conv-progress",
        inbox_id="inbox:worker:demo",
        sender_id="lane:foreground",
        recipient_id="lane:worker:demo",
        msg_type="request.progress",
        payload={"demo": True},
    )
    service.run_registry.append_event(
        "run-progress-1", "worker.requested", {"message_id": "msg-1"}
    )

    snap = service.lane_message_progress(
        run_id="run-progress-1", conversation_id="conv-progress"
    )
    assert snap["total"] >= 2
    assert any(item.get("event_type") == "worker.requested" for item in snap["items"])
    assert any(
        item.get("event_type") == "worker.pending" for item in snap["items"]
    )
