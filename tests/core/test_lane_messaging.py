from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.messaging.service import LaneMessagingService
from kogwistar.server.auth_middleware import claims_ctx
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
            assert messages[0].metadata["purpose"] == "maintenance"
            acl_context = messages[0].metadata["acl_context"]
            assert acl_context["purpose"] == "lane_message"
            assert acl_context["source_graph"] == "conversation"
            assert acl_context["source_entity_id"] == result.message_id
            assert acl_context["visibility"] == "private"

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


def test_lane_message_cross_scope_send_denied_without_explicit_sharing():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-a"})
    try:
        with scoped_namespace(engine, namespace):
            with pytest.raises(Exception):
                engine.send_lane_message(
                    conversation_id="conv-demo",
                    inbox_id="inbox:worker:maintenance",
                    sender_id="lane:foreground",
                    recipient_id="lane:worker:maintenance",
                    msg_type="request.maintenance",
                    payload={"request_node_id": "req-1"},
                    security_scope="tenant-b",
                )
    finally:
        claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_cross_scope_read_denied_unless_explicit_shared():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    sender_token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-a"})
    try:
        with scoped_namespace(engine, namespace):
            private_msg = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.private",
                payload={"request_node_id": "req-private"},
                security_scope="tenant-a",
            )
            shared_msg = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.shared",
                payload={"request_node_id": "req-shared"},
                security_scope="tenant-a",
                shared_scope=True,
            )
    finally:
        claims_ctx.reset(sender_token)
    reader_token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-b"})
    try:
        with scoped_namespace(engine, namespace):
            visible = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance"
            )
            ids = [row.message_id for row in visible]
            assert private_msg.message_id not in ids
            assert shared_msg.message_id in ids
    finally:
        claims_ctx.reset(reader_token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_request_reply_round_trip_preserves_contract():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            request = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
                correlation_id="corr-1",
            )
            reply = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:foreground",
                sender_id="lane:worker:maintenance",
                recipient_id="lane:foreground",
                msg_type="reply.maintenance",
                payload={"result": "ok"},
                correlation_id="corr-1",
                reply_to=request.message_id,
            )

            request_raw = engine.backend.node_get(ids=[request.message_id], include=["documents", "metadatas"])
            reply_raw = engine.backend.node_get(ids=[reply.message_id], include=["documents", "metadatas"])
            assert request_raw["ids"] == [request.message_id]
            assert reply_raw["ids"] == [reply.message_id]
            request_meta = request_raw["metadatas"][0]
            reply_meta = reply_raw["metadatas"][0]
            assert request_meta["status"] == "pending"
            assert reply_meta["reply_to_message_id"] == request.message_id
            assert reply_meta["correlation_id"] == "corr-1"

            worker_rows = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance"
            )
            foreground_rows = engine.list_projected_lane_messages(
                inbox_id="inbox:foreground"
            )
            assert [row.message_id for row in worker_rows] == [request.message_id]
            assert [row.message_id for row in foreground_rows] == [reply.message_id]
            assert foreground_rows[0].correlation_id == "corr-1"
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_sample_integration_pins_stable_contract():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    service = LaneMessagingService(engine)

    try:
        with scoped_namespace(engine, namespace):
            sent = service.send_message(
                conversation_id="conv-integration",
                inbox_id="inbox:worker:integration",
                sender_id="lane:foreground",
                recipient_id="lane:worker:integration",
                msg_type="request.integration",
                payload={"step": "one"},
                correlation_id="corr-integration",
            )
            listed = service.list_projected(inbox_id="inbox:worker:integration")
            assert [row.message_id for row in listed] == [sent.message_id]

            claimed = service.claim_pending(
                inbox_id="inbox:worker:integration",
                claimed_by="worker-integration",
                limit=1,
                lease_seconds=30,
            )
            assert claimed[0].message_id == sent.message_id

            service.ack(
                message_id=sent.message_id,
                claimed_by="worker-integration",
            )
            after_ack = service.list_projected(inbox_id="inbox:worker:integration")
            assert after_ack[0].status == "completed"
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
