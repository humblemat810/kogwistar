from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.server_fixtures import build_engine_triplet
from kogwistar.server.auth_middleware import claims_ctx


pytestmark = pytest.mark.server


@pytest.fixture()
def engine_triplet():
    root = Path(".tmp_lane_message_progress_api_tests") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield build_engine_triplet(
            root=root,
            embedding_function=FakeEmbeddingFunction(),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


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

    token = claims_ctx.set({"storage_ns": conversation_engine.namespace})
    try:
        snap = service.lane_message_progress(
            run_id="run-progress-1", conversation_id="conv-progress"
        )
    finally:
        claims_ctx.reset(token)
    assert snap["total"] >= 2
    assert any(item.get("event_type") == "worker.requested" for item in snap["items"])
    pending = [
        item for item in snap["items"] if item.get("event_type") == "worker.pending"
    ]
    assert pending
    assert pending[0]["status"] == "pending"
    assert pending[0]["inbox_id"] == "inbox:worker:demo"
    assert pending[0]["msg_type"] == "request.progress"
    assert pending[0]["seq"] == 1
    assert pending[0]["conversation_seq"] == 1
