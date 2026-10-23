from __future__ import annotations

import asyncio
import json

import pytest

from kogwistar.agent.event_journal import EventTraceMetadata, LaneProviderEventJournal
from kogwistar.agent.event_tool_dispatch import NamedProjectionToolCallIdempotency
from kogwistar.agent.streaming import (
    ProviderResponseStarted,
    ProviderTextDelta,
    ProviderToolCallCompleted,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.messaging.models import LaneMessageSendResult, ProjectedLaneMessageRow
from kogwistar.messaging.service import LaneMessagingService
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class FakeMessaging:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.row: ProjectedLaneMessageRow | None = None

    def send_message(self, **kwargs: object) -> LaneMessageSendResult:
        self.calls.append(("publish", kwargs))
        payload = kwargs["payload"]
        assert isinstance(payload, dict)
        self.row = ProjectedLaneMessageRow(
            message_id="message-1",
            namespace="test",
            purpose="agent_event",
            inbox_id=str(kwargs["inbox_id"]),
            conversation_id=str(kwargs["conversation_id"]),
            recipient_id=str(kwargs["recipient_id"]),
            sender_id=str(kwargs["sender_id"]),
            msg_type=str(kwargs["msg_type"]),
            status="claimed",
            seq=1,
            conversation_seq=1,
            claimed_by="event-worker",
            lease_until=None,
            retry_count=0,
            created_at=0,
            available_at=0,
            run_id=str(kwargs["run_id"]),
            step_id=str(kwargs["step_id"]),
            correlation_id=str(kwargs["correlation_id"]),
            payload_json=json.dumps(payload),
        )
        return LaneMessageSendResult("message-1", "conversation", "inbox", "sender", "recipient")

    def claim_pending(self, **kwargs: object) -> list[ProjectedLaneMessageRow]:
        self.calls.append(("claim", kwargs))
        return [self.row] if self.row else []

    def ack(self, **kwargs: object) -> None:
        self.calls.append(("ack", kwargs))

    def requeue(self, **kwargs: object) -> None:
        self.calls.append(("requeue", kwargs))

    def dead_letter(self, **kwargs: object) -> None:
        self.calls.append(("dead_letter", kwargs))

    def link_message_to_step(self, **kwargs: object) -> None:
        self.calls.append(("link", kwargs))


def _trace() -> EventTraceMetadata:
    return EventTraceMetadata(
        run_id="run-1",
        turn_id="turn-1",
        workflow_node_id="wf:event:receive",
        step_id="step-2",
        attempt=1,
        provider_id="test-provider",
        provider_response_id="resp-1",
        provider_sequence=3,
        client_sse_sequence=8,
    )


def test_event_journal_persists_semantic_event_then_acknowledges_after_handler() -> None:
    messaging = FakeMessaging()
    journal = LaneProviderEventJournal(messaging, claimed_by="event-worker")
    journal.publish(
        ProviderResponseStarted(response_id="resp-1", provider_sequence=3),
        trace=_trace(),
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
    )
    handled: list[str] = []

    async def handler(event: object) -> None:
        handled.append(str(getattr(event, "event_id")))

    result = asyncio.run(journal.handle_one(run_id="run-1", handler=handler))
    assert result is not None
    assert handled == [result.event_id]
    assert [name for name, _ in messaging.calls] == ["publish", "claim", "ack"]
    payload = messaging.calls[0][1]
    assert isinstance(payload, dict)
    assert payload["purpose"] == "agent_event"
    assert payload["idempotency_key"] == result.event_id


def test_event_journal_identity_is_stable_across_workflow_replay() -> None:
    messaging = FakeMessaging()
    journal = LaneProviderEventJournal(messaging, claimed_by="event-worker")
    event = ProviderResponseStarted(
        provider_event_id="evt-1", response_id="resp-1", provider_sequence=3
    )
    first_trace = _trace()
    replay_trace = first_trace.model_copy(
        update={"step_id": "step-9", "attempt": 2}
    )

    journal.publish(
        event,
        trace=first_trace,
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
    )
    first_payload = messaging.calls[-1][1]
    journal.publish(
        event,
        trace=replay_trace,
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
    )
    replay_payload = messaging.calls[-1][1]

    assert isinstance(first_payload, dict) and isinstance(replay_payload, dict)
    assert first_payload["idempotency_key"] == replay_payload["idempotency_key"]
    assert first_payload["correlation_id"] == replay_payload["correlation_id"]


def test_event_journal_requeues_failed_handler_and_never_persists_token_deltas() -> None:
    messaging = FakeMessaging()
    journal = LaneProviderEventJournal(messaging, claimed_by="event-worker")

    with pytest.raises(ValueError, match="token deltas are transient"):
        journal.publish(
            ProviderTextDelta(delta="secret token"),
            trace=_trace(),
            conversation_id="conv-1",
            sender_id="agent-1",
            security_scope="private",
        )
    journal.publish(
        ProviderResponseStarted(response_id="resp-1"),
        trace=_trace(),
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
    )

    async def handler(_: object) -> None:
        raise ValueError("durable handler failed")

    with pytest.raises(ValueError, match="durable handler failed"):
        asyncio.run(journal.handle_one(run_id="run-1", handler=handler))
    assert [name for name, _ in messaging.calls] == ["publish", "claim", "requeue"]
    requeue_kwargs = messaging.calls[-1][1]
    assert isinstance(requeue_kwargs, dict)
    assert requeue_kwargs["error"] == {
        "code": "event_handler_failed",
        "exception_type": "ValueError",
    }
    assert "durable handler failed" not in str(requeue_kwargs["error"])


def test_event_effect_is_not_repeated_when_process_fails_before_ack(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"),
        kg_graph_type="conversation",
        embedding_function=ConstantEmbeddingFunction(dim=4),
        backend_factory=build_fake_backend,
    )
    claims_token = claims_ctx.set({"ns": "conversation", "security_scope": "private"})
    try:
        with scoped_namespace(engine, "conversation"):
            journal = LaneProviderEventJournal(
                LaneMessagingService(engine), claimed_by="event-worker"
            )
            event = ProviderToolCallCompleted(
                provider_event_id="tool-call-event-1",
                response_id="response-1",
                provider_sequence=7,
                call_id="call-1",
                name="lookup",
                arguments={"query": "alpha"},
            )
            first_publish = journal.publish(
                event,
                trace=_trace(),
                conversation_id="conversation-1",
                sender_id="agent-1",
                security_scope="private",
            )
            duplicate_publish = journal.publish(
                event,
                trace=_trace().model_copy(
                    update={"step_id": "step-9", "attempt": 2}
                ),
                conversation_id="conversation-1",
                sender_id="agent-1",
                security_scope="private",
            )
            assert duplicate_publish.message_id == first_publish.message_id
            idempotency = NamedProjectionToolCallIdempotency(engine.meta_sqlite)
            effects: list[str] = []

            async def crash_after_durable_effect(durable_event: object) -> None:
                event_id = getattr(durable_event, "event_id")
                idempotency.execute_once(
                    str(event_id),
                    lambda: effects.append("invoked") or {"matches": ["alpha"]},
                )
                raise RuntimeError("simulated process loss before event acknowledgement")

            with pytest.raises(RuntimeError, match="before event acknowledgement"):
                asyncio.run(
                    journal.handle_one(
                        run_id="run-1", handler=crash_after_durable_effect
                    )
                )

            outcomes: list[object] = []

            async def replay_after_restart(durable_event: object) -> None:
                event_id = getattr(durable_event, "event_id")
                outcomes.append(
                    idempotency.execute_once(
                        str(event_id),
                        lambda: (_ for _ in ()).throw(
                            AssertionError("replayed tool effect must not run")
                        ),
                    )
                )

            replayed = asyncio.run(
                journal.handle_one(run_id="run-1", handler=replay_after_restart)
            )

            assert replayed is not None
            assert replayed.event_id.startswith("agent-event:")
            assert outcomes == [{"matches": ["alpha"]}]
            assert effects == ["invoked"]
            rows = engine.list_projected_lane_messages(
                inbox_id="agent-provider-events", run_id="run-1"
            )
            assert len(rows) == 1
            assert rows[0].status == "completed"
    finally:
        claims_ctx.reset(claims_token)
