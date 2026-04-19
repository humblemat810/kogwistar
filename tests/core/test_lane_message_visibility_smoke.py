from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_lane_visibility" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    return engine, test_db_dir


def test_lane_message_visibility_smoke_send_claim_requeue():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            sent = engine.send_lane_message(
                conversation_id="conv-visibility",
                inbox_id="inbox:worker:visibility",
                sender_id="lane:foreground",
                recipient_id="lane:worker:visibility",
                msg_type="request.visibility",
                payload={"kind": "smoke"},
            )
            visible = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:visibility"
            )
            assert [row.message_id for row in visible] == [sent.message_id]
            assert visible[0].status == "pending"

            claimed = engine.claim_projected_lane_messages(
                inbox_id="inbox:worker:visibility",
                claimed_by="worker-smoke",
                limit=1,
                lease_seconds=30,
            )
            assert claimed[0].message_id == sent.message_id
            assert claimed[0].status == "claimed"

            engine.requeue_projected_lane_message(
                message_id=sent.message_id,
                claimed_by="worker-smoke",
                error={"reason": "smoke-retry"},
                delay_seconds=0,
            )
            after = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:visibility"
            )
            row = after[0]
            assert row.message_id == sent.message_id
            assert row.status == "pending"
            assert row.retry_count == 1
            assert row.error_json is not None
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_lane_message_visibility_smoke_dead_letter():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        with scoped_namespace(engine, namespace):
            sent = engine.send_lane_message(
                conversation_id="conv-dead",
                inbox_id="inbox:worker:dead",
                sender_id="lane:foreground",
                recipient_id="lane:worker:dead",
                msg_type="request.dead",
                payload={"kind": "dead-letter"},
            )
            claimed = engine.claim_projected_lane_messages(
                inbox_id="inbox:worker:dead",
                claimed_by="worker-dead",
                limit=1,
                lease_seconds=30,
            )
            assert claimed[0].message_id == sent.message_id
            engine.dead_letter_projected_lane_message(
                message_id=sent.message_id,
                claimed_by="worker-dead",
                error={"reason": "terminal"},
            )
            rows = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:dead"
            )
            row = rows[0]
            assert row.status == "dead-letter"
            assert row.claimed_by is None
            assert row.lease_until is None
            assert row.error_json is not None

            engine.ack_projected_lane_message(
                message_id=sent.message_id,
                claimed_by="worker-dead",
            )
            engine.requeue_projected_lane_message(
                message_id=sent.message_id,
                claimed_by="worker-dead",
                error={"reason": "should-not-change"},
                delay_seconds=0,
            )
            after = engine.list_projected_lane_messages(
                inbox_id="inbox:worker:dead"
            )
            row2 = after[0]
            assert row2.status == "dead-letter"
            assert row2.retry_count == 0
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
