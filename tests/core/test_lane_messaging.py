from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_lane_messaging" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    return engine, test_db_dir


def test_send_lane_message_creates_graph_objects_and_projection():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            result = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
            )

            messages = engine.read.get_nodes(where={"artifact_kind": "lane_message"})
            assert len(messages) == 1
            assert messages[0].id == result.message_id
            assert messages[0].metadata["status"] == "pending"
            assert messages[0].metadata["conversation_id"] == "conv-demo"

            anchors = engine.read.get_nodes(where={"artifact_kind": "lane_inbox"})
            assert len(anchors) == 1
            edges = engine.read.get_edges(where={"relation": "in_inbox"})
            assert len(edges) == 1

            projected = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance"
            )
            assert len(projected) == 1
            assert projected[0].message_id == result.message_id
            assert projected[0].status == "pending"
            assert projected[0].seq == 1
            assert projected[0].conversation_seq == 1
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_projected_lane_message_claim_ack_and_requeue():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:index",
                sender_id="lane:foreground",
                recipient_id="lane:worker:index",
                msg_type="request.index",
                payload={"entity_id": "n-1"},
            )
            second = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:index",
                sender_id="lane:foreground",
                recipient_id="lane:worker:index",
                msg_type="request.index",
                payload={"entity_id": "n-2"},
            )

            claimed = engine.claim_projected_lane_messages(
                inbox_id="inbox:worker:index",
                claimed_by="worker-1",
                limit=1,
                lease_seconds=30,
            )
            assert [row.message_id for row in claimed] == [first.message_id]
            assert claimed[0].status == "claimed"

            engine.ack_projected_lane_message(
                message_id=first.message_id,
                claimed_by="worker-1",
            )
            engine.update_lane_message_status(
                message_id=first.message_id,
                status="completed",
                completed=True,
            )

            remaining = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:index",
            )
            statuses = {row.message_id: row.status for row in remaining}
            assert statuses[first.message_id] == "completed"
            assert statuses[second.message_id] == "pending"

            claimed_two = engine.claim_projected_lane_messages(
                inbox_id="inbox:worker:index",
                claimed_by="worker-1",
                limit=1,
                lease_seconds=30,
            )
            assert [row.message_id for row in claimed_two] == [second.message_id]
            engine.requeue_projected_lane_message(
                message_id=second.message_id,
                claimed_by="worker-1",
                error={"reason": "retry"},
                delay_seconds=0,
            )
            engine.update_lane_message_status(
                message_id=second.message_id,
                status="pending",
                error={"reason": "retry"},
            )

            listed = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:index",
            )
            retry_row = [row for row in listed if row.message_id == second.message_id][0]
            assert retry_row.status == "pending"
            assert retry_row.retry_count == 1
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
