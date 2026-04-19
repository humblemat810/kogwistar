from __future__ import annotations

from pathlib import Path

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore


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


def _sqlite_meta(tmp_path: Path) -> EngineSQLite:
    meta = EngineSQLite(tmp_path / "sqlite_meta")
    meta.ensure_initialized()
    return meta
