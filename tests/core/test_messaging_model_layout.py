from dataclasses import replace

from kogwistar.engine_core.engine_sqlite import IndexJobRow, ProjectedLaneMessageSqlRow
from kogwistar.messaging.models import ProjectedLaneMessageRow


def test_projected_lane_message_row_uses_slots_without_changing_copy_contract() -> None:
    row = ProjectedLaneMessageRow(
        message_id="message-1",
        namespace="demo",
        purpose="user_visible",
        inbox_id="inbox-1",
        conversation_id="conversation-1",
        recipient_id="recipient-1",
        sender_id="sender-1",
        msg_type="request",
        status="pending",
        seq=1,
        conversation_seq=1,
        claimed_by=None,
        lease_until=None,
        retry_count=0,
        created_at=1,
        available_at=1,
        run_id=None,
        step_id=None,
        correlation_id=None,
    )

    assert not hasattr(row, "__dict__")
    updated = replace(row, status="completed", error_json='{"error":true}')
    assert row.status == "pending"
    assert updated.status == "completed"
    assert updated.error_json == '{"error":true}'


def test_sqlite_row_dtos_use_slots_without_changing_copy_contract() -> None:
    index_job = IndexJobRow(
        job_id="job-1",
        namespace="demo",
        entity_kind="node",
        entity_id="node-1",
        index_kind="embedding",
        coalesce_key="node-1",
        op="UPSERT",
        status="PENDING",
        lease_until=None,
        next_run_at=None,
        max_retries=3,
        retry_count=0,
        last_error=None,
        payload_json=None,
        created_at=1,
        updated_at=1,
    )
    lane_row = ProjectedLaneMessageSqlRow(
        message_id="message-1",
        namespace="demo",
        purpose="user_visible",
        inbox_id="inbox-1",
        conversation_id="conversation-1",
        recipient_id="recipient-1",
        sender_id="sender-1",
        msg_type="request",
        status="pending",
        seq=1,
        conversation_seq=1,
        claimed_by=None,
        lease_until=None,
        retry_count=0,
        created_at=1,
        available_at=1,
        run_id=None,
        step_id=None,
        correlation_id=None,
        payload_json=None,
        error_json=None,
    )

    assert not hasattr(index_job, "__dict__")
    assert not hasattr(lane_row, "__dict__")
    assert replace(index_job, status="DONE").status == "DONE"
    assert replace(lane_row, status="completed").status == "completed"
