from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_lane_rebuild" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    return engine, test_db_dir


def test_safe_repair_restores_missing_rows_without_touching_existing_claim():
    engine, test_db_dir = _make_engine()
    namespace = "ws:demo:conv:bg"

    try:
        token = claims_ctx.set({"storage_ns": namespace})
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-repair",
                inbox_id="inbox:worker:repair",
                sender_id="lane:foreground",
                recipient_id="lane:worker:repair",
                msg_type="request.repair",
                payload={"kind": "repair-1"},
            )
            second = engine.send_lane_message(
                conversation_id="conv-repair",
                inbox_id="inbox:worker:repair",
                sender_id="lane:foreground",
                recipient_id="lane:worker:repair",
                msg_type="request.repair",
                payload={"kind": "repair-2"},
            )

            original = engine.meta_sqlite.list_projected_lane_messages(
                namespace=namespace,
                inbox_id="inbox:worker:repair",
            )
            first_original = [row for row in original if row.message_id == first.message_id][0]
            assert engine.meta_sqlite.clear_projected_lane_messages(namespace) == 2
            engine.meta_sqlite.project_lane_message(
                message_id=first_original.message_id,
                namespace=first_original.namespace,
                purpose=first_original.purpose,
                inbox_id=first_original.inbox_id,
                conversation_id=first_original.conversation_id,
                recipient_id=first_original.recipient_id,
                sender_id=first_original.sender_id,
                msg_type=first_original.msg_type,
                status=first_original.status,
                created_at=first_original.created_at,
                available_at=first_original.available_at,
                run_id=first_original.run_id,
                step_id=first_original.step_id,
                correlation_id=first_original.correlation_id,
                payload_json=first_original.payload_json,
                error_json=first_original.error_json,
            )
            claimed = engine.meta_sqlite.claim_projected_lane_messages(
                namespace=namespace,
                inbox_id="inbox:worker:repair",
                claimed_by="worker-1",
                limit=1,
                lease_seconds=30,
            )
            assert [row.message_id for row in claimed] == [first.message_id]
            claims_ctx.reset(token)
            token = None

            result = engine.repair_lane_message_projection(namespace=namespace)

            rows = engine.meta_sqlite.list_projected_lane_messages(
                namespace=namespace,
                inbox_id="inbox:worker:repair",
            )
            by_id = {row.message_id: row for row in rows}
            assert result.namespace == namespace
            assert result.scanned_count == 2
            assert result.repaired_count == 1
            assert result.skipped_count == 1
            assert result.rebuilt is False
            assert by_id[first.message_id].status == "claimed"
            assert by_id[first.message_id].claimed_by == "worker-1"
            assert by_id[second.message_id].status == "pending"
    finally:
        if "token" in locals() and token is not None:
            claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_rebuild_clears_stale_rows_and_rematerializes_from_entity_events():
    engine, test_db_dir = _make_engine()
    namespace = "default"

    try:
        token = claims_ctx.set({"storage_ns": namespace})
        with scoped_namespace(engine, namespace):
            first = engine.send_lane_message(
                conversation_id="conv-rebuild",
                inbox_id="inbox:worker:rebuild",
                sender_id="lane:foreground",
                recipient_id="lane:worker:rebuild",
                msg_type="request.rebuild",
                payload={"kind": "rebuild"},
                correlation_id="corr-rebuild-1",
            )
            second = engine.send_lane_message(
                conversation_id="conv-rebuild",
                inbox_id="inbox:worker:rebuild",
                sender_id="lane:foreground",
                recipient_id="lane:worker:rebuild",
                msg_type="request.rebuild",
                payload={"kind": "rebuild-2"},
                correlation_id="corr-rebuild-2",
            )
            engine.update_lane_message_status(
                message_id=second.message_id,
                status="failed",
                error={"reason": "boom"},
            )
            engine.meta_sqlite.project_lane_message(
                message_id="msg:stale",
                namespace=namespace,
                purpose="maintenance",
                inbox_id="inbox:worker:rebuild",
                conversation_id="conv-rebuild",
                recipient_id="lane:worker:rebuild",
                sender_id="lane:foreground",
                msg_type="request.stale",
                status="pending",
                created_at=999,
                available_at=999,
                run_id=None,
                step_id=None,
                correlation_id="corr-stale",
                payload_json='{"kind":"stale"}',
                error_json=None,
            )

            result = engine.repair_lane_message_projection(namespace=namespace, rebuild=True)
            claims_ctx.reset(token)
            token = None

            rows = engine.meta_sqlite.list_projected_lane_messages(
                namespace=namespace,
                inbox_id="inbox:worker:rebuild",
            )
            assert result.rebuilt is True
            assert result.scanned_count == 2
            assert result.repaired_count == 2
            assert [row.message_id for row in rows] == [first.message_id, second.message_id]
            assert [row.seq for row in rows] == [1, 2]
            assert [row.retry_count for row in rows] == [0, 0]
            assert all(row.claimed_by is None for row in rows)

            by_id = {row.message_id: row for row in rows}
            assert by_id[first.message_id].purpose == "maintenance"
            assert by_id[first.message_id].payload_json == '{"kind":"rebuild"}'
            assert by_id[first.message_id].correlation_id == "corr-rebuild-1"
            assert by_id[second.message_id].status == "failed"
            assert json.loads(by_id[second.message_id].error_json or "{}") == {"reason": "boom"}
    finally:
        if "token" in locals() and token is not None:
            claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_repair_is_namespace_isolated():
    engine, test_db_dir = _make_engine()
    ns_a = "ws:a:conv:bg"
    ns_b = "ws:b:conv:bg"

    try:
        token = claims_ctx.set({"storage_ns": ns_a})
        with scoped_namespace(engine, ns_a):
            msg_a = engine.send_lane_message(
                conversation_id="conv-a",
                inbox_id="inbox:worker:repair",
                sender_id="lane:foreground",
                recipient_id="lane:worker:repair",
                msg_type="request.repair",
                payload={"workspace": "a"},
            )
        claims_ctx.reset(token)
        token = claims_ctx.set({"storage_ns": ns_b})
        with scoped_namespace(engine, ns_b):
            msg_b = engine.send_lane_message(
                conversation_id="conv-b",
                inbox_id="inbox:worker:repair",
                sender_id="lane:foreground",
                recipient_id="lane:worker:repair",
                msg_type="request.repair",
                payload={"workspace": "b"},
            )
        claims_ctx.reset(token)
        token = None

        assert engine.meta_sqlite.clear_projected_lane_messages(ns_a) == 1
        result = engine.repair_lane_message_projection(namespace=ns_a)

        rows_a = engine.meta_sqlite.list_projected_lane_messages(
            namespace=ns_a,
            inbox_id="inbox:worker:repair",
        )
        rows_b = engine.meta_sqlite.list_projected_lane_messages(
            namespace=ns_b,
            inbox_id="inbox:worker:repair",
        )
        assert result.repaired_count == 1
        assert [row.message_id for row in rows_a] == [msg_a.message_id]
        assert [row.message_id for row in rows_b] == [msg_b.message_id]
    finally:
        if "token" in locals() and token is not None:
            claims_ctx.reset(token)
        shutil.rmtree(test_db_dir, ignore_errors=True)
