"""Explicit workflow control-point bindings over durable lane messages.

Control points are ordinary resolver operations.  They claim only visible,
targeted messages and leave acknowledgement to a later ordinary step, after
the control-point state has been checkpointed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from kogwistar.messaging.service import LaneMessagingService
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver


ControlPolicy = Literal["steer", "queue", "cancel_and_replace"]
_TERMINAL_MESSAGE_STATUSES = {"completed", "failed", "cancelled", "dead-letter"}


@dataclass(frozen=True, slots=True)
class ControlPointSpec:
    """Policy metadata for one ordinary workflow control-point node."""

    inbox_id: str
    policy: ControlPolicy = "steer"
    accepted_message_types: tuple[str, ...] = ("agent.steer",)
    recipient_id: str | None = None
    max_claims: int = 1
    lease_seconds: int = 60
    run_specific_inbox: bool = False
    claimed_by_key: str = "agent_claimed_by"
    output_key: str = "agent_control_input"
    # Ordinary workflow state: runtime-reserved underscore keys cannot be
    # rehydrated through the public checkpoint validator.
    pending_claims_key: str = "agent_control_pending_claims"
    high_water_key: str = "agent_control_high_water"

    def __post_init__(self) -> None:
        if not str(self.inbox_id).strip():
            raise ValueError("inbox_id must be non-empty")
        if self.policy not in {"steer", "queue", "cancel_and_replace"}:
            raise ValueError(f"unsupported control policy: {self.policy}")
        if not self.accepted_message_types or any(
            not str(item).strip() for item in self.accepted_message_types
        ):
            raise ValueError("accepted_message_types must be non-empty")
        if int(self.max_claims) != 1:
            raise ValueError("v1 control points support one FIFO claim")
        if int(self.lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive")

    def resolved_inbox_id(self, run_id: str) -> str:
        if self.run_specific_inbox:
            return f"{self.inbox_id}:run:{run_id}"
        return self.inbox_id


def _failure(message: str) -> RunFailure:
    return RunFailure(conversation_node_id=None, state_update=[], errors=[message])


def _payload(row: Any) -> dict[str, Any]:
    raw = getattr(row, "payload_json", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _claims(state: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = state.get(key, [])
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _ack_claims(
    messaging: LaneMessagingService,
    claims: list[dict[str, Any]],
) -> None:
    for claim in claims:
        message_id = str(claim.get("message_id") or "").strip()
        claimed_by = str(claim.get("claimed_by") or "").strip()
        if message_id and claimed_by:
            messaging.ack(message_id=message_id, claimed_by=claimed_by)


def register_control_point(
    resolver: MappingStepResolver,
    messaging: LaneMessagingService,
    spec: ControlPointSpec,
    *,
    op: str = "agent.control_point",
) -> None:
    """Register one visible, FIFO, durable control-point operation.

    The operation acknowledges claims recorded by the preceding checkpoint,
    then claims at most one new message.  The new claim remains pending in
    checkpointed state until an explicit ``register_control_ack_step`` runs.
    """

    @resolver.register(op)
    def _control(ctx: Any) -> RunSuccess | RunFailure:
        state = ctx._state
        claimed_by = str(state.get(spec.claimed_by_key) or ctx.run_id).strip()
        if not claimed_by:
            return _failure("control point requires a claimed-by identity")
        prior_claims = _claims(state, spec.pending_claims_key)
        try:
            _ack_claims(messaging, prior_claims)
        except Exception as exc:
            return _failure(f"control acknowledgement failed: {exc}")

        recipient_id = spec.recipient_id or state.get("agent_recipient_id")
        if not str(recipient_id or "").strip():
            return _failure("control point requires recipient identity")
        inbox_id = spec.resolved_inbox_id(str(ctx.run_id))
        rows = messaging.list_projected(
            inbox_id=inbox_id,
            run_id=str(ctx.run_id),
            recipient_id=str(recipient_id),
            status=None,
            limit=1000,
            newest_first=False,
        )
        accepted = [
            row
            for row in rows
            if str(getattr(row, "msg_type", "")) in spec.accepted_message_types
            and str(getattr(row, "status", "")) not in _TERMINAL_MESSAGE_STATUSES
        ]
        if spec.policy == "queue" and accepted:
            first = accepted[0]
            if str(getattr(first, "status", "")) == "claimed" and str(
                getattr(first, "claimed_by", "") or ""
            ) != claimed_by:
                accepted = []
            else:
                accepted = [first]
        elif accepted:
            accepted = [accepted[0]]

        high_water = int(state.get(spec.high_water_key) or 0)
        candidate = accepted[0] if accepted else None
        if candidate is None:
            empty_update: dict[str, Any] = {
                spec.pending_claims_key: [],
                "agent_control_received": False,
                "agent_control_policy": spec.policy,
            }
            # Resume may replay this control node after its state checkpoint
            # but before ack. Preserve the already-applied input while the
            # old claim is being acknowledged; do not erase durable work.
            if not prior_claims:
                empty_update[spec.output_key] = None
            return RunSuccess(
                state_update=[("u", empty_update)]
            )

        message_id = str(getattr(candidate, "message_id", ""))
        claimed = messaging.claim_pending(
            inbox_id=inbox_id,
            claimed_by=claimed_by,
            limit=1,
            lease_seconds=spec.lease_seconds,
            message_ids=[message_id],
            run_id=str(ctx.run_id),
            msg_type=str(getattr(candidate, "msg_type", "")),
            recipient_id=str(recipient_id),
        )
        if not claimed:
            empty_update = {
                spec.pending_claims_key: [],
                "agent_control_received": False,
                "agent_control_policy": spec.policy,
            }
            if not prior_claims:
                empty_update[spec.output_key] = None
            return RunSuccess(
                state_update=[("u", empty_update)]
            )
        row = claimed[0]
        seq = int(getattr(row, "seq", 0) or 0)
        duplicate = seq <= high_water
        next_high_water = max(high_water, seq)
        claim_record = {
            "message_id": str(row.message_id),
            "claimed_by": claimed_by,
            "seq": seq,
            # Message ID is the immutable conversation-graph input anchor;
            # retain delivery coordinates beside it for replay/audit.
            "conversation_id": str(getattr(row, "conversation_id", "") or ""),
            "correlation_id": getattr(row, "correlation_id", None),
            "step_id": getattr(row, "step_id", None),
            "run_id": str(getattr(row, "run_id", "") or ""),
        }
        update: dict[str, Any] = {
            spec.output_key: None if duplicate else _payload(row),
            spec.pending_claims_key: [claim_record],
            spec.high_water_key: next_high_water,
            "agent_control_received": not duplicate,
            "agent_control_duplicate": duplicate,
            "agent_control_policy": spec.policy,
        }
        if spec.policy == "cancel_and_replace" and not duplicate:
            update["agent_control_action"] = "cancel_and_replace"
        else:
            update["agent_control_action"] = "steer" if not duplicate else "duplicate"
        return RunSuccess(state_update=[("u", update)])


def register_control_ack_step(
    resolver: MappingStepResolver,
    messaging: LaneMessagingService,
    *,
    op: str = "agent.control_ack",
    pending_claims_key: str = "agent_control_pending_claims",
) -> None:
    """Acknowledge claims after their control-point state was checkpointed."""

    @resolver.register(op)
    def _ack(ctx: Any) -> RunSuccess | RunFailure:
        claims = _claims(ctx._state, pending_claims_key)
        try:
            _ack_claims(messaging, claims)
        except Exception as exc:
            return _failure(f"control acknowledgement failed: {exc}")
        return RunSuccess(state_update=[("u", {pending_claims_key: []})])


def terminalize_control_messages(
    messaging: LaneMessagingService,
    *,
    run_id: str,
    reason: str = "target_terminal",
) -> int:
    """Cancel unconsumed control inputs after their target run is terminal."""

    rows = messaging.list_projected(run_id=str(run_id), status=None, limit=10000)
    count = 0
    for row in rows:
        if str(getattr(row, "status", "")) in _TERMINAL_MESSAGE_STATUSES:
            continue
        messaging.update_message_status(
            message_id=str(row.message_id),
            status="cancelled",
            error={"reason": reason, "run_id": str(run_id)},
            completed=True,
        )
        count += 1
    return count


__all__ = [
    "ControlPointSpec",
    "ControlPolicy",
    "register_control_point",
    "register_control_ack_step",
    "terminalize_control_messages",
]
