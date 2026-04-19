from __future__ import annotations

import pytest

from kogwistar.messaging.models import (
    LaneMessageSendResult,
    ProjectedLaneMessageRow,
)


@pytest.fixture
def lane_message_contract_sample() -> dict[str, object]:
    return {
        "send_result": LaneMessageSendResult(
            message_id="msg-sample",
            conversation_anchor_id="anchor-conv",
            inbox_anchor_id="anchor-inbox",
            sender_anchor_id="anchor-sender",
            recipient_anchor_id="anchor-recipient",
        ),
        "projected_row": ProjectedLaneMessageRow(
            message_id="msg-sample",
            namespace="ns-sample",
            inbox_id="inbox:worker:sample",
            conversation_id="conv-sample",
            recipient_id="lane:worker:sample",
            sender_id="lane:foreground",
            msg_type="request.sample",
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
            correlation_id="corr-sample",
            payload_json='{"sample":true}',
            error_json=None,
        ),
    }
