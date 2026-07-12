from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ProjectedLaneMessageRow:
    message_id: str
    namespace: str
    purpose: str
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
    prev_message_id: str | None = None
    next_message_id: str | None = None
    inbox_tail_message_id: str | None = None
    conversation_tail_message_id: str | None = None


@dataclass(frozen=True)
class LaneMessageSendResult:
    message_id: str
    conversation_anchor_id: str
    inbox_anchor_id: str
    sender_anchor_id: str
    recipient_anchor_id: str


@dataclass(frozen=True)
class LaneMessageProjectionRepairResult:
    namespace: str
    scanned_count: int
    repaired_count: int
    skipped_count: int
    rebuilt: bool


@dataclass(frozen=True)
class LaneMessageLookup:
    namespace: str | None = None
    inbox_id: str | None = None
    conversation_id: str | None = None
    status: str | None = None
    purpose: str | None = None
    msg_type: str | None = None
    sender_id: str | None = None
    recipient_id: str | None = None
    correlation_id: str | None = None
    reply_to_message_id: str | None = None
    idempotency_key: str | None = None
    created_at_gte: datetime | int | float | str | None = None
    created_at_lte: datetime | int | float | str | None = None
    available_at_gte: int | None = None
    available_at_lte: int | None = None
    limit: int = 100
    newest_first: bool = False


__all__ = [
    "LaneMessageLookup",
    "ProjectedLaneMessageRow",
    "LaneMessageSendResult",
    "LaneMessageProjectionRepairResult",
]
