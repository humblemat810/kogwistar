from __future__ import annotations

from pathlib import Path

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.meta_lane_messages import LaneMessageMetaStoreMixin
from kogwistar.messaging.models import ProjectedLaneMessageRow


pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "meta_factory",
    [
        lambda tmp_path: InMemoryMetaStore(),
        lambda tmp_path: _sqlite_meta(tmp_path),
    ],
    ids=["in-memory", "sqlite"],
)
def test_lane_message_projection_contract(meta_factory, tmp_path):
    meta = meta_factory(tmp_path)
    meta.project_lane_message(
        message_id="msg-1",
        namespace="ns-a",
        inbox_id="inbox:worker:index",
        conversation_id="conv-1",
        recipient_id="lane:worker:index",
        sender_id="lane:foreground",
        msg_type="request.index",
        status="pending",
        created_at=1,
        available_at=1,
        run_id=None,
        step_id=None,
        correlation_id="corr-1",
        payload_json='{"entity_id":"n-1"}',
    )
    meta.project_lane_message(
        message_id="msg-2",
        namespace="ns-a",
        inbox_id="inbox:worker:index",
        conversation_id="conv-1",
        recipient_id="lane:worker:index",
        sender_id="lane:foreground",
        msg_type="request.index",
        status="pending",
        created_at=2,
        available_at=2,
        run_id=None,
        step_id=None,
        correlation_id="corr-2",
        payload_json='{"entity_id":"n-2"}',
    )

    listed = meta.list_projected_lane_messages(namespace="ns-a", inbox_id="inbox:worker:index")
    assert [row.seq for row in listed] == [1, 2]
    assert [row.conversation_seq for row in listed] == [1, 2]
    assert all(row.purpose == "user_visible" for row in listed)

    claimed = meta.claim_projected_lane_messages(
        namespace="ns-a",
        inbox_id="inbox:worker:index",
        claimed_by="worker-1",
        limit=1,
        lease_seconds=30,
    )
    assert [row.message_id for row in claimed] == ["msg-1"]

    meta.ack_projected_lane_message(message_id="msg-1", claimed_by="worker-1")
    meta.requeue_projected_lane_message(
        message_id="msg-2",
        claimed_by="worker-1",
        error_json='{"reason":"retry"}',
        delay_seconds=0,
    )

    listed_after = meta.list_projected_lane_messages(namespace="ns-a", inbox_id="inbox:worker:index")
    status_by_id = {row.message_id: row.status for row in listed_after}
    retry_by_id = {row.message_id: row.retry_count for row in listed_after}
    assert status_by_id["msg-1"] == "completed"
    assert status_by_id["msg-2"] == "pending"
    assert retry_by_id["msg-2"] == 1


def test_lane_message_projection_rebuild_is_backend_parity(tmp_path):
    in_memory = InMemoryMetaStore()
    sqlite = _sqlite_meta(tmp_path)
    for meta in (in_memory, sqlite):
        meta.project_lane_message(
            message_id="msg-1",
            namespace="ns-a",
            inbox_id="inbox:worker:index",
            conversation_id="conv-1",
            recipient_id="lane:worker:index",
            sender_id="lane:foreground",
            msg_type="request.index",
            status="pending",
            created_at=1,
            available_at=1,
            run_id=None,
            step_id=None,
            correlation_id="corr-1",
            payload_json='{"entity_id":"n-1"}',
        )
        meta.project_lane_message(
            message_id="msg-2",
            namespace="ns-a",
            inbox_id="inbox:worker:index",
            conversation_id="conv-1",
            recipient_id="lane:worker:index",
            sender_id="lane:foreground",
            msg_type="request.index",
            status="pending",
            created_at=2,
            available_at=2,
            run_id=None,
            step_id=None,
            correlation_id="corr-2",
            payload_json='{"entity_id":"n-2"}',
        )

        rows = meta.list_projected_lane_messages(namespace="ns-a", inbox_id="inbox:worker:index")
        assert [row.message_id for row in rows] == ["msg-1", "msg-2"]
        assert all(isinstance(row, ProjectedLaneMessageRow) for row in rows)


def test_lane_message_metastore_classes_share_common_mixin():
    assert issubclass(InMemoryMetaStore, LaneMessageMetaStoreMixin)
    assert issubclass(EngineSQLite, LaneMessageMetaStoreMixin)
    assert issubclass(EnginePostgresMetaStore, LaneMessageMetaStoreMixin)


def test_lane_message_contract_sample_fixture_is_compliant(
    lane_message_contract_sample,
):
    send_result = lane_message_contract_sample["send_result"]
    projected_row = lane_message_contract_sample["projected_row"]
    assert send_result.message_id == projected_row.message_id
    assert projected_row.status == "pending"
    assert projected_row.seq == 1
    assert projected_row.conversation_seq == 1
    assert projected_row.purpose == "user_visible"
    assert projected_row.correlation_id == "corr-sample"
    assert projected_row.payload_json == '{"sample":true}'


def _sqlite_meta(tmp_path: Path) -> EngineSQLite:
    meta = EngineSQLite(tmp_path / "sqlite_meta")
    meta.ensure_initialized()
    return meta
