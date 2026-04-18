from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectedLaneMessageRow:
    message_id: str
    namespace: str
    inbox_id: str
    conversation_id: str
    recipient_id: str
    sender_id: str
    msg_type: str
    status: str
    seq: int
    conversation_seq: int
    claimed_by: str | None
    lease_until: int | None
    retry_count: int
    created_at: int
    available_at: int
    run_id: str | None
    step_id: str | None
    correlation_id: str | None
    payload_json: str | None = None
    error_json: str | None = None


@dataclass(frozen=True)
class LaneMessageSendResult:
    message_id: str
    conversation_anchor_id: str
    inbox_anchor_id: str
    sender_anchor_id: str
    recipient_anchor_id: str


__all__ = [
    "ProjectedLaneMessageRow",
    "LaneMessageSendResult",
]
