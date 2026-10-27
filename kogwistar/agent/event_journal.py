"""Durable semantic provider-event journal over lane-message primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict

from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.messaging.models import LaneMessageSendResult, ProjectedLaneMessageRow
from .streaming import ProviderEvent, ProviderTextDelta


class EventTraceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    turn_id: str
    workflow_node_id: str
    step_id: str
    attempt: int = 1
    producer_step_seq: int | None = None
    producer_attempt: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    provider_id: str
    provider_response_id: str | None = None
    provider_sequence: int | None = None
    client_sse_sequence: int | None = None


class DurableProviderEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    trace: EventTraceMetadata
    event: ProviderEvent


class ClaimedProviderEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    claimed_by: str
    durable_event: DurableProviderEvent


class LaneMessageJournalService(Protocol):
    """Minimal messaging operations required by the event journal."""

    def send_message(
        self,
        *,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        payload: dict[str, object],
        run_id: str | None = None,
        step_id: str | None = None,
        correlation_id: str | None = None,
        purpose: str | None = None,
        security_scope: str | None = None,
        idempotency_key: str | None = None,
    ) -> LaneMessageSendResult: ...

    def claim_pending_attempt(
        self,
        *,
        inbox_id: str,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        run_id: str | None = None,
        msg_type: str | None = None,
        recipient_id: str | None = None,
    ) -> list[ProjectedLaneMessageRow]: ...

    def ack(self, *, message_id: str, claimed_by: str) -> None: ...

    def requeue(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict[str, object],
    ) -> None: ...

    def dead_letter(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict[str, object],
    ) -> None: ...

    def link_message_to_step(
        self,
        *,
        message_id: str,
        step_id: str,
        conversation_id: str,
        run_id: str,
        relation: Literal["handled_by_step", "emitted_by_step"] = "handled_by_step",
    ) -> None: ...


EventHandler = Callable[[DurableProviderEvent], Awaitable[None]]


def _canonical_event(event: ProviderEvent) -> JsonObject:
    return cast(JsonObject, event.model_dump(mode="json"))


def _event_identity(trace: EventTraceMetadata, event: ProviderEvent) -> str:
    """Build a replay-stable ID from provider coordinates, not workflow attempts.

    Providers that omit both an event ID and sequence number only get
    attempt-scoped deduplication; without a provider event coordinate, treating
    equal payloads as the same event could incorrectly collapse legitimate
    repeated events.
    """

    provider_coordinate: str | int | None = (
        event.provider_event_id
        if event.provider_event_id is not None
        else event.provider_sequence
    )
    identity: dict[str, JsonValue] = {
        "provider_id": trace.provider_id,
        "response_id": event.response_id or trace.provider_response_id,
        "workflow_node_id": trace.workflow_node_id,
        "event": _canonical_event(event),
    }
    if provider_coordinate is None:
        identity["fallback_attempt"] = {
            "run_id": trace.run_id,
            "turn_id": trace.turn_id,
            "workflow_node_id": trace.workflow_node_id,
            "step_id": trace.step_id,
            "attempt": trace.attempt,
        }
    else:
        identity["provider_coordinate"] = str(provider_coordinate)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class LaneProviderEventJournal:
    """Persist, claim, handle, and acknowledge one semantic event at a time.

    The handler must return only after its effects/checkpoint are durable. A
    failure requeues the claim, so the handler must be idempotent by event_id.
    """

    def __init__(
        self,
        messaging: LaneMessageJournalService,
        *,
        inbox_id: str = "agent-provider-events",
        recipient_id: str = "agent-event-consumer",
        claimed_by: str = "agent-event-worker",
        lease_seconds: int = 60,
    ) -> None:
        self._messaging = messaging
        self.inbox_id = inbox_id
        self.recipient_id = recipient_id
        self.claimed_by = claimed_by
        self.lease_seconds = int(lease_seconds)
        if not all((inbox_id.strip(), recipient_id.strip(), claimed_by.strip())):
            raise ValueError("event inbox and worker identities must be non-empty")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")

    def publish(
        self,
        event: ProviderEvent,
        *,
        trace: EventTraceMetadata,
        conversation_id: str,
        sender_id: str,
        security_scope: str,
    ) -> LaneMessageSendResult:
        if isinstance(event, ProviderTextDelta):
            raise ValueError("token deltas are transient and must not enter the graph")
        digest = _event_identity(trace, event)
        response_id = event.response_id or trace.provider_response_id
        stable_trace = trace.model_copy(
            update={
                # The runtime step sequence/attempt may change when a workflow
                # replays the same provider event. The provider response is
                # the stable producer operation; exact execution coordinates
                # remain in the runtime execution graph and telemetry.
                "step_id": f"provider-response:{response_id or 'unknown'}",
                "attempt": 1,
            }
        )
        durable = DurableProviderEvent(
            event_id=f"agent-event:{digest}",
            trace=stable_trace,
            event=event,
        )
        payload = cast(JsonObject, durable.model_dump(mode="json"))
        return self._messaging.send_message(
            conversation_id=conversation_id,
            inbox_id=self.inbox_id,
            sender_id=sender_id,
            recipient_id=self.recipient_id,
            msg_type="agent.provider_event",
            payload=cast(dict[str, object], payload),
            run_id=trace.run_id,
            step_id=stable_trace.step_id,
            correlation_id=durable.event_id,
            purpose="agent_event",
            security_scope=security_scope,
            idempotency_key=durable.event_id,
        )

    async def handle_one(
        self,
        *,
        run_id: str,
        handler: EventHandler,
    ) -> DurableProviderEvent | None:
        claim = self.claim_one(run_id=run_id)
        if claim is None:
            return None
        try:
            await handler(claim.durable_event)
        except Exception as exc:
            # Provider/tool failures can contain prompts, credentials, or other
            # untrusted content. Keep durable operational errors content-free.
            self.requeue(
                claim,
                error={
                    "code": "event_handler_failed",
                    "exception_type": type(exc).__name__,
                },
            )
            raise
        self.acknowledge(claim)
        return claim.durable_event

    def claim_one(self, *, run_id: str) -> ClaimedProviderEvent | None:
        claimed = self._messaging.claim_pending_attempt(
            inbox_id=self.inbox_id,
            worker_id=self.claimed_by,
            limit=1,
            lease_seconds=self.lease_seconds,
            run_id=run_id,
            msg_type="agent.provider_event",
            recipient_id=self.recipient_id,
        )
        if not claimed:
            return None
        row = claimed[0]
        if not row.claimed_by or row.claimed_by == self.claimed_by:
            raise RuntimeError("event claim did not preserve its attempt owner")
        durable = self._decode(row)
        return ClaimedProviderEvent(
            message_id=row.message_id,
            claimed_by=row.claimed_by,
            durable_event=durable,
        )

    def acknowledge(self, claim: ClaimedProviderEvent) -> None:
        self._messaging.ack(
            message_id=claim.message_id,
            claimed_by=claim.claimed_by,
        )

    def link_handling_step(
        self,
        claim: ClaimedProviderEvent,
        *,
        conversation_id: str,
        run_id: str,
        step_seq: int,
    ) -> None:
        if step_seq < 0:
            raise ValueError("step_seq must be non-negative")
        self._messaging.link_message_to_step(
            message_id=claim.message_id,
            step_id=f"wf_step|{run_id}|{step_seq}",
            conversation_id=conversation_id,
            run_id=run_id,
            relation="handled_by_step",
        )

    def link_producer_step(
        self,
        claim: ClaimedProviderEvent,
        *,
        conversation_id: str,
        run_id: str,
    ) -> None:
        trace = claim.durable_event.trace
        if trace.producer_step_seq is None:
            return
        self._messaging.link_message_to_step(
            message_id=claim.message_id,
            step_id=f"wf_step|{run_id}|{trace.producer_step_seq}",
            conversation_id=conversation_id,
            run_id=run_id,
            relation="emitted_by_step",
        )

    def requeue(self, claim: ClaimedProviderEvent, *, error: JsonObject) -> None:
        self._messaging.requeue(
            message_id=claim.message_id,
            claimed_by=claim.claimed_by,
            error=cast(dict[str, object], error),
        )

    def dead_letter(self, claim: ClaimedProviderEvent, *, error: JsonObject) -> None:
        self._messaging.dead_letter(
            message_id=claim.message_id,
            claimed_by=claim.claimed_by,
            error=cast(dict[str, object], error),
        )

    @staticmethod
    def _decode(row: ProjectedLaneMessageRow) -> DurableProviderEvent:
        raw = row.payload_json
        if not isinstance(raw, str):
            raise ValueError("durable provider event payload is missing")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("durable provider event payload must be an object")
        return DurableProviderEvent.model_validate(payload)


__all__ = [
    "ClaimedProviderEvent",
    "DurableProviderEvent",
    "EventTraceMetadata",
    "LaneMessageJournalService",
    "LaneProviderEventJournal",
]
