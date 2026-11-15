"""Run-scoped durable queue for safe-boundary event-agent control input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict

from kogwistar.messaging.models import LaneMessageSendResult

from .event_journal import LaneMessageJournalService

EventControlMode = Literal["steer", "queue", "cancel_and_continue"]
EventControlStatus = Literal["pending", "accepted", "committed", "failed"]


class LaneControlReceiptReader(Protocol):
    def list_projected(
        self,
        *,
        inbox_id: str,
        run_id: str,
        sender_id: str,
        correlation_id: str,
        limit: int,
        newest_first: bool,
    ) -> list[object]: ...


class EventControlSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    control_id: str
    message_id: str
    status: EventControlStatus
    effective_mode: EventControlMode


class EventAgentControlJournal:
    """Publish user input to the event workflow's run-specific control inbox.

    The caller must authenticate the user and authorize access to ``run_id``
    before calling this storage adapter. Provider capability negotiation must
    happen first; the accepted mode is recorded but never grants provider or
    tool authority.
    """

    def __init__(
        self,
        messaging: LaneMessageJournalService,
        *,
        inbox_id: str = "agent-event-controls",
    ) -> None:
        self._messaging = messaging
        self.inbox_id = str(inbox_id).strip()
        if not self.inbox_id:
            raise ValueError("control inbox ID must be non-empty")

    @staticmethod
    def _idempotency_key(run_id: str, user_id: str, control_id: str) -> str:
        identity = "\0".join((run_id, user_id, control_id)).encode("utf-8")
        return f"agent-control:v1:{hashlib.sha256(identity).hexdigest()}"

    def submit(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_id: str,
        recipient_id: str,
        control_id: str,
        text: str,
        effective_mode: EventControlMode,
        security_scope: str,
    ) -> EventControlSubmission:
        values = {
            "run_id": str(run_id).strip(),
            "conversation_id": str(conversation_id).strip(),
            "user_id": str(user_id).strip(),
            "recipient_id": str(recipient_id).strip(),
            "control_id": str(control_id).strip(),
            "text": str(text).strip(),
            "security_scope": str(security_scope).strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("run, conversation, user, recipient, control, text, and scope are required")
        inbox_id = f"{self.inbox_id}:run:{values['run_id']}"
        idempotency_key = self._idempotency_key(
            values["run_id"], values["user_id"], values["control_id"]
        )
        result: LaneMessageSendResult = self._messaging.send_message(
            conversation_id=values["conversation_id"],
            inbox_id=inbox_id,
            sender_id=f"user:{values['user_id']}",
            recipient_id=values["recipient_id"],
            msg_type="agent.steer",
            payload={
                "control_id": values["control_id"],
                "text": values["text"],
                "effective_mode": effective_mode,
                "user_id": values["user_id"],
            },
            run_id=values["run_id"],
            correlation_id=values["control_id"],
            purpose="user_visible",
            security_scope=values["security_scope"],
            idempotency_key=idempotency_key,
        )
        return EventControlSubmission(
            control_id=values["control_id"],
            message_id=result.message_id,
            status="pending",
            effective_mode=effective_mode,
        )

    def get_submission(
        self,
        *,
        run_id: str,
        user_id: str,
        control_id: str,
    ) -> EventControlSubmission | None:
        """Read the durable control lifecycle from its lane-message projection.

        A claimed message is accepted by the workflow; completed means the
        control payload has crossed the control-point checkpoint and was acked.
        The host must authorize run ownership before calling this method.
        """

        run_id = str(run_id).strip()
        user_id = str(user_id).strip()
        control_id = str(control_id).strip()
        if not run_id or not user_id or not control_id:
            raise ValueError("run, user, and control IDs are required")
        rows = cast(LaneControlReceiptReader, self._messaging).list_projected(
            inbox_id=f"{self.inbox_id}:run:{run_id}",
            run_id=run_id,
            sender_id=f"user:{user_id}",
            correlation_id=control_id,
            limit=1000,
            newest_first=False,
        )
        row = next(
            (
                candidate
                for candidate in rows
                if str(getattr(candidate, "msg_type", "")) == "agent.steer"
                and str(getattr(candidate, "run_id", "") or "") == run_id
                and str(getattr(candidate, "sender_id", "")) == f"user:{user_id}"
            ),
            None,
        )
        if row is None:
            return None
        lane_status = str(getattr(row, "status", ""))
        status: EventControlStatus
        if lane_status == "claimed":
            status = "accepted"
        elif lane_status == "completed":
            status = "committed"
        elif lane_status in {"failed", "cancelled", "dead-letter"}:
            status = "failed"
        else:
            status = "pending"
        payload = getattr(row, "payload_json", None)
        effective_mode: EventControlMode = "queue"
        if isinstance(payload, Mapping):
            raw_mode = payload.get("effective_mode")
            if raw_mode in {"steer", "queue", "cancel_and_continue"}:
                effective_mode = raw_mode
        else:
            try:
                parsed = json.loads(str(payload or "{}"))
                raw_mode = parsed.get("effective_mode") if isinstance(parsed, dict) else None
                if raw_mode in {"steer", "queue", "cancel_and_continue"}:
                    effective_mode = raw_mode
            except (TypeError, ValueError):
                pass
        return EventControlSubmission(
            control_id=control_id,
            message_id=str(getattr(row, "message_id", "")),
            status=status,
            effective_mode=effective_mode,
        )


__all__ = [
    "EventAgentControlJournal",
    "EventControlMode",
    "EventControlStatus",
    "EventControlSubmission",
]
