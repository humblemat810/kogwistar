from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import contextmanager
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
            acl_context = json.loads(messages[0].metadata["acl_context_json"])
            assert acl_context["purpose"] == "lane_message"
            assert acl_context["source_graph"] == "conversation"
            assert acl_context["source_entity_id"] == result.message_id
            assert acl_context["visibility"] == "private"
            assert json.loads(messages[0].metadata["payload_json"]) == {
                "request_node_id": "req-1"
            }

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


def test_send_lane_message_preserves_trace_fields_without_requiring_trace_nodes():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            result = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:runtime",
                sender_id="lane:foreground",
                recipient_id="lane:worker:runtime",
                msg_type="request.runtime",
                payload={"request_node_id": "req-1"},
                run_id="run-missing-trace-node",
                step_id="0",
            )

            messages = engine.read.get_nodes(where={"artifact_kind": "lane_message"})
            projected = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:runtime"
            )
            assert [str(node.id) for node in messages] == [result.message_id]
            assert messages[0].metadata["run_id"] == "run-missing-trace-node"
            assert messages[0].metadata["step_id"] == "0"
            assert len(projected) == 1
            assert projected[0].run_id == "run-missing-trace-node"
            assert projected[0].step_id == "0"
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_send_lane_message_projects_inside_engine_unit_of_work(monkeypatch):
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    state = {"depth": 0, "entered": 0}
    original_project = engine.meta_sqlite.project_lane_message

    @contextmanager
    def _unit_of_work():
        state["entered"] += 1
        state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    def _project_lane_message(**kwargs):
        assert state["depth"] > 0
        return original_project(**kwargs)

    monkeypatch.setattr(engine, "uow", _unit_of_work)
    monkeypatch.setattr(engine.meta_sqlite, "project_lane_message", _project_lane_message)

    try:
        with scoped_namespace(engine, namespace):
            result = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:uow",
                sender_id="lane:foreground",
                recipient_id="lane:worker:uow",
                msg_type="request.uow",
                payload={"request_node_id": "req-1"},
            )

            projected = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:uow"
            )
            assert state["entered"] >= 1
            assert [row.message_id for row in projected] == [result.message_id]
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_projection_can_split_maintenance_and_user_visible_flows():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            maintenance = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
            )
            user_visible = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.answer",
                payload={"request_node_id": "req-2"},
            )

            maintenance_rows = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance",
                purpose="maintenance",
            )
            user_visible_rows = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance",
                purpose="user_visible",
            )

            assert [row.message_id for row in maintenance_rows] == [maintenance.message_id]
            assert [row.message_id for row in user_visible_rows] == [user_visible.message_id]
            assert maintenance_rows[0].purpose == "maintenance"
            assert user_visible_rows[0].purpose == "user_visible"
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


def test_send_lane_message_idempotency_key_reuses_existing_message_and_projection():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    token = claims_ctx.set({"storage_ns": namespace})

    try:
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
                idempotency_key="idem:maintenance:req-1",
            )
            engine.meta_sqlite.clear_projected_lane_messages(namespace)
            assert engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance"
            ) == []

            second = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
                idempotency_key="idem:maintenance:req-1",
            )

            assert second.message_id == first.message_id
            messages = engine.read.get_nodes(where={"artifact_kind": "lane_message"})
            assert len(messages) == 1
            projected = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:maintenance"
            )
            assert len(projected) == 1
            assert projected[0].message_id == first.message_id
    finally:
        claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_send_lane_message_idempotency_key_rejects_shape_conflict():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    token = claims_ctx.set({"storage_ns": namespace})

    try:
        with scoped_namespace(engine, namespace):
            engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:worker:maintenance",
                sender_id="lane:foreground",
                recipient_id="lane:worker:maintenance",
                msg_type="request.maintenance",
                payload={"request_node_id": "req-1"},
                idempotency_key="idem:maintenance:req-1",
            )
            with pytest.raises(ValueError, match="lane message idempotency conflict"):
                engine.send_lane_message(
                    conversation_id="conv-demo",
                    inbox_id="inbox:worker:maintenance",
                    sender_id="lane:foreground",
                    recipient_id="lane:worker:maintenance",
                    msg_type="request.maintenance",
                    payload={"request_node_id": "req-2"},
                    idempotency_key="idem:maintenance:req-1",
                )
    finally:
        claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_lookup_supports_newest_first_and_time_filters():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"
    token = claims_ctx.set({"storage_ns": namespace})

    try:
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:foreground",
                sender_id="lane:worker",
                recipient_id="lane:foreground",
                msg_type="reply.maintenance.completed",
                payload={"result": "first"},
                correlation_id="corr-1",
            )
            time.sleep(1)
            second = engine.send_lane_message(
                conversation_id="conv-demo",
                inbox_id="inbox:foreground",
                sender_id="lane:worker",
                recipient_id="lane:foreground",
                msg_type="reply.maintenance.completed",
                payload={"result": "second"},
                correlation_id="corr-2",
                reply_to=first.message_id,
            )

            newest = engine.list_projected_lane_messages(
                inbox_id="inbox:foreground",
                newest_first=True,
                limit=1,
            )
            assert [row.message_id for row in newest] == [second.message_id]

            later = engine.list_projected_lane_messages(
                inbox_id="inbox:foreground",
                created_at_gte=newest[0].created_at,
            )
            assert [row.message_id for row in later] == [second.message_id]

            reply_rows = engine.list_projected_lane_messages(
                inbox_id="inbox:foreground",
                reply_to_message_id=first.message_id,
            )
            assert [row.message_id for row in reply_rows] == [second.message_id]

            found = engine.find_lane_messages(
                namespace=namespace,
                correlation_id="corr-2",
                newest_first=True,
                limit=1,
            )
            assert [str(node.id) for node in found] == [second.message_id]
    finally:
        claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)
