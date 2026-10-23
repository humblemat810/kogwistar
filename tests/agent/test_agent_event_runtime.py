from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import cast

import pytest

from kogwistar.agent.event_journal import (
    ClaimedProviderEvent,
    DurableProviderEvent,
    EventTraceMetadata,
    LaneProviderEventJournal,
)
from kogwistar.agent.event_runtime import (
    EventWorkflowCallbacks,
    _bounded_annotations,
    _emit_event_handled,
    _event_effect_idempotency_key,
    _event_handler_trace,
    _tool_idempotency_key,
    register_event_workflow_steps,
)
from kogwistar.agent.streaming import (
    ProviderCapabilities,
    ProviderEvent,
    ProviderFieldCompleted,
    ProviderReasoningSummary,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderResponseStarted,
    ProviderTextDelta,
    ProviderToolCallCompleted,
)
from kogwistar.messaging.models import LaneMessageSendResult
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import AsyncMappingStepResolver
from kogwistar.runtime.runtime import StepContext
from kogwistar.runtime.telemetry import EventEmitter, TraceContext

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


class FakeProvider:
    provider_id = "fake-stream"
    capabilities = ProviderCapabilities(structured_output=True)

    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        del request
        async def events() -> AsyncIterator[ProviderEvent]:
            yield ProviderResponseStarted(response_id="r1", provider_sequence=1)
            yield ProviderTextDelta(response_id="r1", provider_sequence=2, delta='{"answer":"ok"}')
            yield ProviderFieldCompleted(response_id="r1", provider_sequence=2, field_name="answer", value="ok")
            yield ProviderResponseCompleted(response_id="r1", provider_sequence=3, text='{"answer":"ok"}', structured_output={"answer": "ok"})

        return events()


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[ProviderEvent, EventTraceMetadata]] = []

    def publish(self, event: ProviderEvent, *, trace: EventTraceMetadata, **_: object) -> LaneMessageSendResult:
        self.events.append((event, trace))
        return LaneMessageSendResult(f"msg-{len(self.events)}", "c", "i", "s", "r")

    def claim_one(self, *, run_id: str) -> ClaimedProviderEvent | None:
        if not self.events:
            return None
        event, trace = self.events.pop(0)
        durable = DurableProviderEvent(event_id="event-1", trace=trace, event=event)
        return ClaimedProviderEvent(message_id="message-1", claimed_by="worker", durable_event=durable)

    def link_handling_step(self, *_: object, **__: object) -> None:
        return None

    def acknowledge(self, _: ClaimedProviderEvent) -> None:
        return None

    def dead_letter(self, *_: object, **__: object) -> None:
        raise AssertionError("unexpected dead letter")


class Callbacks(EventWorkflowCallbacks):
    def __init__(self) -> None:
        self.deltas: list[tuple[str, int]] = []

    def build_request(self, state: Mapping[str, object]) -> ProviderRequest:
        del state
        return ProviderRequest(model="fake", input="hello")

    async def before_model(self, context: StepContext) -> object:
        del context
        return {"hint": "safe"}

    async def publish_delta(self, context: StepContext, delta: str, sequence: int) -> object:
        del context
        self.deltas.append((delta, sequence))

    async def on_advisory_event(
        self,
        context: StepContext,
        event: ProviderFieldCompleted | ProviderReasoningSummary,
        *,
        idempotency_key: str,
    ) -> object:
        del context
        return {"reviewed": True, "event": event.kind, "idempotency_key": idempotency_key}

    async def before_tool_dispatch(
        self, context: StepContext, call: ProviderToolCallCompleted
    ) -> object:
        del context, call
        return {}

    async def authorize_and_dispatch_tool(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        *,
        idempotency_key: str,
    ) -> object:
        del context, call
        return {"idempotency_key": idempotency_key}

    async def after_tool_outcome(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        outcome: Mapping[str, object],
    ) -> object:
        del context, call, outcome
        return {}

    async def append_assistant_turn(
        self,
        context: StepContext,
        response: ProviderResponseCompleted,
        *,
        idempotency_key: str,
    ) -> object:
        del context, response, idempotency_key
        return None

    async def record_failure(
        self,
        context: StepContext,
        failure: ProviderResponseFailed,
        *,
        idempotency_key: str,
    ) -> object:
        del context, failure, idempotency_key
        return None


def test_persisted_hook_and_tool_annotations_are_bounded_json() -> None:
    assert _bounded_annotations({"safe": [1, "ok"]}) == {"safe": [1, "ok"]}
    assert _bounded_annotations({"large": "x" * 6_001}) == {
        "_status": "hook_output_over_budget"
    }


def test_tool_idempotency_key_is_stable_for_replay_and_scoped_to_response() -> None:
    first = ProviderToolCallCompleted(
        response_id="response-1", call_id="call-1", name="read", arguments={}
    )
    replay = first.model_copy()
    later_response = first.model_copy(update={"response_id": "response-2"})

    key = _tool_idempotency_key("run-1", "provider-1", first)
    assert _tool_idempotency_key("run-1", "provider-1", replay) == key
    assert _tool_idempotency_key("run-1", "provider-1", later_response) != key
    assert _tool_idempotency_key("run-1", "provider-2", first) != key


def test_graph_effect_idempotency_keys_are_stable_and_effect_scoped() -> None:
    assistant_key = _event_effect_idempotency_key(
        "run-1", "event-1", effect="assistant_turn"
    )
    assert _event_effect_idempotency_key(
        "run-1", "event-1", effect="assistant_turn"
    ) == assistant_key
    assert _event_effect_idempotency_key(
        "run-1", "event-1", effect="failure_record"
    ) != assistant_key
    assert _event_effect_idempotency_key(
        "run-2", "event-1", effect="assistant_turn"
    ) != assistant_key


def test_event_handler_trace_is_child_of_provider_request_span() -> None:
    source = TraceContext.new_root(
        run_id="run-1", token_id="token-1", step_seq=2, node_id="provider-request"
    )
    claim = ClaimedProviderEvent(
        message_id="message-1",
        claimed_by="worker",
        durable_event=DurableProviderEvent(
            event_id="event-1",
            trace=EventTraceMetadata(
                run_id="run-1",
                turn_id="turn-1",
                workflow_node_id="provider-request",
                step_id="2",
                provider_id="fake-stream",
                trace_id=source.trace_id,
                span_id=source.span_id,
            ),
            event=ProviderResponseStarted(response_id="response-1"),
        ),
    )

    handler = _event_handler_trace(_context(), claim)

    assert handler.trace_id == source.trace_id
    assert handler.span_id != source.span_id
    assert handler.parent_span_id == source.span_id


def test_event_handling_telemetry_keeps_provider_parentage_without_content() -> None:
    source = TraceContext.new_root(
        run_id="run-1", token_id="token-1", step_seq=2, node_id="provider-request"
    )
    claim = ClaimedProviderEvent(
        message_id="message-1",
        claimed_by="worker",
        durable_event=DurableProviderEvent(
            event_id="event-1",
            trace=EventTraceMetadata(
                run_id="run-1",
                turn_id="turn-1",
                workflow_node_id="provider-request",
                step_id="2",
                provider_id="fake-stream",
                trace_id=source.trace_id,
                span_id=source.span_id,
            ),
            event=ProviderReasoningSummary(response_id="response-1", text="private"),
        ),
    )

    class CapturingEmitter:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def emit(self, **kwargs: object) -> str:
            self.kwargs = kwargs
            return "telemetry-1"

    emitter = CapturingEmitter()
    context = _context()
    context.events = cast(EventEmitter, emitter)
    _emit_event_handled(context, claim)

    trace = emitter.kwargs["ctx"]
    payload = emitter.kwargs["payload"]
    assert isinstance(trace, TraceContext)
    assert trace.parent_span_id == source.span_id
    assert isinstance(payload, Mapping)
    assert payload["event_id"] == "event-1"
    assert "text" not in payload
    assert _bounded_annotations({"not_json": object()}) == {
        "_status": "invalid_hook_output"
    }


def _context() -> StepContext:
    return StepContext(
        run_id="run-1",
        workflow_id="agent.event.v1",
        workflow_node_id="wf:agent.event.v1:request",
        op="agent.event.request",
        token_id="token-1",
        attempt=1,
        step_seq=3,
        cache_dir=None,
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={},
    )


def test_request_step_persists_semantic_events_but_keeps_tokens_transient() -> None:
    resolver = AsyncMappingStepResolver()
    callbacks = Callbacks()
    journal = FakeJournal()
    trace_events: list[tuple[str, Mapping[str, object]]] = []

    def publish_trace_event(
        event_type: str, payload: Mapping[str, object]
    ) -> None:
        trace_events.append((event_type, payload))

    register_event_workflow_steps(
        resolver,
        provider=FakeProvider(),
        journal=cast(LaneProviderEventJournal, journal),
        callbacks=callbacks,
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
        advisory_field_names=frozenset({"answer"}),
        publish_trace_event=publish_trace_event,
    )
    request_step = resolver.handlers["agent.event.request"]
    request_callable = cast(Callable[[StepContext], Awaitable[object]], request_step)
    async def invoke_request() -> object:
        return await request_callable(_context())

    result = asyncio.run(invoke_request())

    assert isinstance(result, RunSuccess)
    assert callbacks.deltas == [('{"answer":"ok"}', 1)]
    assert [event.kind for event, _ in journal.events] == [
        "response_started",
        "field_completed",
        "response_completed",
    ]
    assert all(trace.run_id == "run-1" and trace.turn_id == "turn-1" for _, trace in journal.events)
    assert journal.events[1][1].client_sse_sequence == 1
    assert [event_type for event_type, _ in trace_events] == [
        "provider.request.started",
        "provider.event.persisted",
        "provider.event.persisted",
        "provider.event.persisted",
        "provider.request.completed",
    ]
    assert trace_events[0][1]["workflow_step_seq"] == 3
    assert trace_events[1][1]["provider_sequence"] == 1
    assert trace_events[-1][1]["provider_response_id"] == "r1"


@pytest.mark.parametrize(
    "event,allowed_fields,allow_summaries,hook_limit,initial_count,expected_route",
    [
        (ProviderFieldCompleted(field_name="answer", value="yes"), frozenset(), False, 8, 0, "no_action"),
        (ProviderFieldCompleted(field_name="reasoning", value="private"), frozenset(), True, 8, 0, "no_action"),
        (ProviderFieldCompleted(field_name="answer", value="yes"), frozenset({"answer"}), False, 8, 0, "advisory"),
        (ProviderReasoningSummary(text="summary"), frozenset(), False, 8, 0, "no_action"),
        (ProviderReasoningSummary(text="summary"), frozenset(), True, 8, 0, "advisory"),
        (ProviderFieldCompleted(field_name="answer", value="yes"), frozenset({"answer"}), False, 1, 1, "no_action"),
        (ProviderFieldCompleted(field_name="answer", value="yes"), frozenset({"answer"}), False, 1, -10, "advisory"),
    ],
)
def test_advisory_hook_routing_requires_opt_in_and_respects_run_budget(
    event: ProviderEvent,
    allowed_fields: frozenset[str],
    allow_summaries: bool,
    hook_limit: int,
    initial_count: int,
    expected_route: str,
) -> None:
    resolver = AsyncMappingStepResolver()
    journal = FakeJournal()
    journal.events.append(
        (
            event,
            EventTraceMetadata(
                run_id="run-1",
                turn_id="turn-1",
                workflow_node_id="request",
                step_id="1",
                provider_id="fake-stream",
            ),
        )
    )
    register_event_workflow_steps(
        resolver,
        provider=FakeProvider(),
        journal=cast(LaneProviderEventJournal, journal),
        callbacks=Callbacks(),
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
        advisory_field_names=allowed_fields,
        allow_reasoning_summaries=allow_summaries,
        max_advisory_hooks_per_run=hook_limit,
    )
    context = _context()
    context._state["agent_event_advisory_hook_count"] = initial_count
    receive = cast(
        Callable[[StepContext], Awaitable[object]],
        resolver.handlers["agent.event.receive"],
    )
    result = asyncio.run(receive(context))

    assert isinstance(result, RunSuccess)
    assert result.state_update[0][1]["agent_event_route"] == expected_route


@pytest.mark.parametrize(
    "prior_continuations,expected_route,expected_count",
    [(0, "resume", 1), (1, "final", 1)],
)
def test_advisory_followup_model_call_is_opt_in_and_bounded(
    prior_continuations: int, expected_route: str, expected_count: int
) -> None:
    resolver = AsyncMappingStepResolver()
    journal = FakeJournal()
    callbacks = Callbacks()
    register_event_workflow_steps(
        resolver,
        provider=FakeProvider(),
        journal=cast(LaneProviderEventJournal, journal),
        callbacks=callbacks,
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
        max_advisory_continuations_per_run=1,
    )
    event = ProviderResponseCompleted(response_id="response-1", text="answer")
    durable = DurableProviderEvent(
        event_id="event-final",
        trace=EventTraceMetadata(
            run_id="run-1",
            turn_id="turn-1",
            workflow_node_id="request",
            step_id="1",
            provider_id="fake-stream",
        ),
        event=event,
    )
    claim = ClaimedProviderEvent(
        message_id="message-final", claimed_by="worker", durable_event=durable
    )
    context = _context()
    context._state.update(
        {
            "agent_event_current": durable.model_dump(mode="json"),
            "agent_event_claim": claim.model_dump(mode="json"),
            "agent_event_advisory_response_hook_count": 1,
            "agent_event_advisory_continuation_count": prior_continuations,
        }
    )
    finalize = cast(
        Callable[[StepContext], Awaitable[object]],
        resolver.handlers["agent.event.finalize"],
    )

    result = asyncio.run(finalize(context))

    assert isinstance(result, RunSuccess)
    assert result.state_update[0][1]["agent_event_route"] == expected_route
    assert result.state_update[0][1].get(
        "agent_event_advisory_continuation_count", expected_count
    ) == expected_count


def test_advisory_continuation_limit_has_a_hard_ceiling() -> None:
    with pytest.raises(ValueError, match="between 0 and"):
        register_event_workflow_steps(
            AsyncMappingStepResolver(),
            provider=FakeProvider(),
            journal=cast(LaneProviderEventJournal, FakeJournal()),
            callbacks=Callbacks(),
            conversation_id="conv-1",
            sender_id="agent-1",
            security_scope="private",
            max_advisory_continuations_per_run=5,
        )


def test_advisory_hook_clamps_negative_recovered_counters_before_increment() -> None:
    resolver = AsyncMappingStepResolver()
    journal = FakeJournal()
    journal.events.append(
        (
            ProviderFieldCompleted(field_name="answer", value="ok"),
            EventTraceMetadata(
                run_id="run-1",
                turn_id="turn-1",
                workflow_node_id="request",
                step_id="1",
                provider_id="fake-stream",
            ),
        )
    )
    claim = journal.claim_one(run_id="run-1")
    assert claim is not None
    register_event_workflow_steps(
        resolver,
        provider=FakeProvider(),
        journal=cast(LaneProviderEventJournal, journal),
        callbacks=Callbacks(),
        conversation_id="conv-1",
        sender_id="agent-1",
        security_scope="private",
        advisory_field_names=frozenset({"answer"}),
    )
    context = _context()
    context._state.update(
        {
            "agent_event_current": claim.durable_event.model_dump(mode="json"),
            "agent_event_claim": claim.model_dump(mode="json"),
            "agent_event_advisory_hook_count": -10,
            "agent_event_advisory_response_hook_count": -10,
        }
    )
    hook = cast(
        Callable[[StepContext], Awaitable[object]], resolver.handlers["agent.event.hook"]
    )

    result = asyncio.run(hook(context))

    assert isinstance(result, RunSuccess)
    updates = result.state_update[0][1]
    assert updates["agent_event_advisory_hook_count"] == 1
    assert updates["agent_event_advisory_response_hook_count"] == 1
