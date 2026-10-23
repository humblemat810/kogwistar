"""Ordinary workflow-step bindings for the opt-in streaming event agent."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol, cast

from pydantic import TypeAdapter

from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.messaging.service import LaneMessagingService
from kogwistar.runtime.models import RunFailure, RunSuccess
from kogwistar.runtime.resolvers import AsyncMappingStepResolver
from kogwistar.runtime.runtime import StepContext
from kogwistar.runtime.telemetry import TraceContext

from .event_journal import (
    ClaimedProviderEvent,
    DurableProviderEvent,
    EventTraceMetadata,
    LaneProviderEventJournal,
)
from .control import ControlPointSpec, register_control_ack_step, register_control_point
from .streaming import (
    ProviderFieldCompleted,
    ProviderReasoningSummary,
    ProviderRequest,
    ProviderResponseCompleted,
    ProviderResponseFailed,
    ProviderTextDelta,
    ProviderToolCallCompleted,
    StreamingModelProvider,
)

EventCallback = Callable[..., object]
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_MAX_HOOK_ANNOTATION_CHARS = 6_000
_MAX_ADVISORY_CONTINUATIONS_PER_RUN = 4


class EventWorkflowCallbacks(Protocol):
    """Host-owned callbacks; tool dispatch must reuse host authorization."""

    def build_request(self, state: Mapping[str, object]) -> ProviderRequest: ...

    def before_model(self, context: StepContext) -> object: ...

    def publish_delta(self, context: StepContext, delta: str, sequence: int) -> object: ...

    def on_advisory_event(
        self,
        context: StepContext,
        event: ProviderFieldCompleted | ProviderReasoningSummary,
        *,
        idempotency_key: str,
    ) -> object: ...

    def authorize_and_dispatch_tool(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        *,
        idempotency_key: str,
    ) -> object: ...

    def before_tool_dispatch(
        self, context: StepContext, call: ProviderToolCallCompleted
    ) -> object: ...

    def after_tool_outcome(
        self,
        context: StepContext,
        call: ProviderToolCallCompleted,
        outcome: Mapping[str, object],
    ) -> object: ...

    def append_assistant_turn(
        self,
        context: StepContext,
        response: ProviderResponseCompleted,
        *,
        idempotency_key: str,
    ) -> object: ...

    def record_failure(
        self,
        context: StepContext,
        failure: ProviderResponseFailed,
        *,
        idempotency_key: str,
    ) -> object: ...


async def _call(callback: EventCallback, *args: object, **kwargs: object) -> object:
    result = callback(*args, **kwargs)
    return await cast(Awaitable[object], result) if inspect.isawaitable(result) else result


def _event_from_state(state: Mapping[str, object]) -> DurableProviderEvent:
    raw = state.get("agent_event_current")
    if not isinstance(raw, Mapping):
        raise ValueError("workflow state has no claimed semantic event")
    return DurableProviderEvent.model_validate(raw)


def _claim_from_state(state: Mapping[str, object]) -> ClaimedProviderEvent:
    raw = state.get("agent_event_claim")
    if not isinstance(raw, Mapping):
        raise ValueError("workflow state has no durable event claim")
    return ClaimedProviderEvent.model_validate(raw)


def _state_update(values: Mapping[str, JsonValue]) -> RunSuccess:
    return RunSuccess(state_update=[("u", dict(values))])


def _state_integer(state: Mapping[str, object], key: str, *, default: int = 0) -> int:
    value = state.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _tool_idempotency_key(
    run_id: str, provider_id: str, event: ProviderToolCallCompleted
) -> str:
    """Scope a provider call ID to its run, provider, and response."""

    identity = "\0".join(
        (run_id, provider_id, event.response_id or "", event.call_id)
    ).encode("utf-8")
    return f"agent-tool-call:v1:{hashlib.sha256(identity).hexdigest()}"


def _event_effect_idempotency_key(
    run_id: str, event_id: str, *, effect: str
) -> str:
    """Give replayable host-side effects a stable idempotency key."""

    identity = "\0".join((effect, run_id, event_id)).encode("utf-8")
    return f"agent-event-effect:v1:{hashlib.sha256(identity).hexdigest()}"


def _advisory_idempotency_key(run_id: str, event_id: str) -> str:
    identity = "\0".join(("advisory-hook", run_id, event_id)).encode("utf-8")
    return f"agent-advisory-hook:v1:{hashlib.sha256(identity).hexdigest()}"


def _event_handler_trace(
    ctx: StepContext, claim: ClaimedProviderEvent
) -> TraceContext:
    """Create a handler span causally parented by the originating provider step."""

    source = claim.durable_event.trace
    if source.trace_id and source.span_id:
        parent = TraceContext(
            run_id=source.run_id,
            token_id=ctx.token_id,
            step_seq=ctx.step_seq,
            node_id=source.workflow_node_id,
            attempt=source.attempt,
            conversation_id=ctx.conversation_id,
            turn_node_id=ctx.turn_node_id,
            trace_id=source.trace_id,
            span_id=source.span_id,
            parent_span_id=source.parent_span_id,
        )
    else:
        parent = ctx.trace_ctx
    return parent.child_span(
        token_id=ctx.token_id,
        step_seq=ctx.step_seq,
        node_id=ctx.workflow_node_id,
        attempt=ctx.attempt,
    )


def _emit_event_handled(ctx: StepContext, claim: ClaimedProviderEvent) -> None:
    emitter = ctx.events
    if emitter is None:
        return
    try:
        emitter.emit(
            type="agent_provider_event_handled",
            ctx=_event_handler_trace(ctx, claim),
            payload={
                "provider_id": claim.durable_event.trace.provider_id,
                "event_id": claim.durable_event.event_id,
                "event_kind": claim.durable_event.event.kind,
                "message_id": claim.message_id,
                "workflow_node_id": ctx.workflow_node_id,
                "step_seq": ctx.step_seq,
            },
        )
    except Exception:
        # Workflow state and graph links are durable; telemetry is observational.
        pass


def _bounded_annotations(raw: object) -> JsonObject:
    if not isinstance(raw, Mapping):
        return {}
    try:
        value = _JSON_OBJECT_ADAPTER.validate_python(dict(raw))
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return {"_status": "invalid_hook_output"}
    if len(serialized) > _MAX_HOOK_ANNOTATION_CHARS:
        return {"_status": "hook_output_over_budget"}
    return cast(JsonObject, value)


def register_event_workflow_steps(
    resolver: AsyncMappingStepResolver,
    *,
    provider: StreamingModelProvider,
    journal: LaneProviderEventJournal,
    callbacks: EventWorkflowCallbacks,
    conversation_id: str,
    sender_id: str,
    security_scope: str,
    control_messaging: LaneMessagingService | None = None,
    control_spec: ControlPointSpec | None = None,
    advisory_field_names: frozenset[str] = frozenset(),
    allow_reasoning_summaries: bool = False,
    max_advisory_hooks_per_run: int = 8,
    max_advisory_continuations_per_run: int = 0,
) -> dict[str, Callable[..., bool]]:
    """Register the graph event-loop operations and their pure route predicates.

    Every provider event except token deltas is first written to the lane inbox.
    The receive operation claims only one message; ack is a separate subsequent
    workflow node, after the relevant effect has become a durable step result.
    When control messaging is configured, the workflow polls one run-scoped
    queued input at each safe pre-request boundary, checkpoints it in state,
    then acknowledges the lane message before calling the provider. The host's
    ``build_request`` callback is responsible for incorporating that input.
    """

    if (control_messaging is None) != (control_spec is None):
        raise ValueError("control_messaging and control_spec must be configured together")
    if max_advisory_hooks_per_run < 0:
        raise ValueError("max_advisory_hooks_per_run cannot be negative")
    if not 0 <= max_advisory_continuations_per_run <= _MAX_ADVISORY_CONTINUATIONS_PER_RUN:
        raise ValueError(
            "max_advisory_continuations_per_run must be between 0 and "
            f"{_MAX_ADVISORY_CONTINUATIONS_PER_RUN}"
        )
    if any(not name.strip() for name in advisory_field_names):
        raise ValueError("advisory field names must be non-empty")

    if control_messaging is not None and control_spec is not None:
        register_control_point(
            resolver,
            control_messaging,
            control_spec,
            op="agent.event.control",
        )
        register_control_ack_step(
            resolver,
            control_messaging,
            op="agent.event.control_ack",
            pending_claims_key=control_spec.pending_claims_key,
        )
    else:
        @resolver.register("agent.event.control")
        async def control_disabled(_: StepContext) -> RunSuccess:
            return _state_update({"agent_control_input": None})

        @resolver.register("agent.event.control_ack")
        async def control_ack_disabled(_: StepContext) -> RunSuccess:
            return _state_update({})

    @resolver.register("agent.event.start")
    async def start(_: StepContext) -> RunSuccess:
        return _state_update({"agent_event_route": "prepare"})

    @resolver.register("agent.event.prepare")
    async def prepare(ctx: StepContext) -> RunSuccess:
        advice = await _call(callbacks.before_model, ctx)
        values: dict[str, JsonValue] = {"agent_event_route": "request"}
        if isinstance(advice, Mapping):
            values["agent_event_pre_model_advice"] = _bounded_annotations(advice)
        return _state_update(values)

    @resolver.register("agent.event.request")
    async def request(ctx: StepContext) -> RunSuccess | RunFailure:
        provider_request = callbacks.build_request(ctx.state_view)
        client_sequence = _state_integer(ctx.state_view, "agent_event_client_sse_sequence")
        semantic_count = 0
        provider_sequence = 0
        try:
            async for event in provider.stream(provider_request):
                if event.provider_sequence is not None:
                    provider_sequence = max(provider_sequence, event.provider_sequence)
                if isinstance(event, ProviderToolCallCompleted) and not provider.capabilities.tool_calls:
                    event = ProviderResponseFailed(
                        response_id=event.response_id,
                        provider_sequence=event.provider_sequence,
                        error_code="unsupported_tool_event",
                        message="provider emitted a tool call outside its declared capabilities",
                    )
                if isinstance(event, ProviderTextDelta):
                    client_sequence += 1
                    await _call(callbacks.publish_delta, ctx, event.delta, client_sequence)
                    continue
                trace_context = ctx.trace_ctx
                trace = EventTraceMetadata(
                    run_id=str(ctx.run_id),
                    turn_id=str(ctx.turn_node_id or ""),
                    workflow_node_id=str(ctx.workflow_node_id),
                    step_id=str(ctx.step_seq),
                    attempt=int(ctx.attempt),
                    trace_id=trace_context.trace_id,
                    span_id=trace_context.span_id,
                    parent_span_id=trace_context.parent_span_id,
                    provider_id=provider.provider_id,
                    provider_response_id=event.response_id,
                    provider_sequence=event.provider_sequence,
                    client_sse_sequence=client_sequence,
                )
                result = journal.publish(
                    event,
                    trace=trace,
                    conversation_id=conversation_id,
                    sender_id=sender_id,
                    security_scope=security_scope,
                )
                semantic_count += 1
                emitter = ctx.events
                if emitter is not None and ctx.trace_context is not None:
                    try:
                        emitter.emit(
                            type="agent_provider_semantic_event_persisted",
                            ctx=ctx.trace_context.child_span(node_id=ctx.workflow_node_id),
                            payload={
                                "provider_id": provider.provider_id,
                                "event_kind": event.kind,
                                "provider_event_id": event.provider_event_id,
                                "provider_response_id": event.response_id,
                                "provider_sequence": event.provider_sequence,
                                "durable_message_id": result.message_id,
                            },
                        )
                    except Exception:
                        pass
        except Exception as exc:
            failure = ProviderResponseFailed(
                error_code="provider_stream_error",
                message=f"provider stream failed: {type(exc).__name__}",
                retryable=False,
            )
            trace_context = ctx.trace_ctx
            trace = EventTraceMetadata(
                run_id=str(ctx.run_id),
                turn_id=str(ctx.turn_node_id or ""),
                workflow_node_id=str(ctx.workflow_node_id),
                step_id=str(ctx.step_seq),
                attempt=int(ctx.attempt),
                trace_id=trace_context.trace_id,
                span_id=trace_context.span_id,
                parent_span_id=trace_context.parent_span_id,
                provider_id=provider.provider_id,
                provider_sequence=provider_sequence,
                client_sse_sequence=client_sequence,
            )
            journal.publish(
                failure,
                trace=trace,
                conversation_id=conversation_id,
                sender_id=sender_id,
                security_scope=security_scope,
            )
            semantic_count += 1
        if semantic_count == 0:
            empty_failure = ProviderResponseFailed(
                error_code="empty_provider_stream",
                message="provider completed without semantic events",
            )
            trace_context = ctx.trace_ctx
            journal.publish(
                empty_failure,
                trace=EventTraceMetadata(
                    run_id=str(ctx.run_id),
                    turn_id=str(ctx.turn_node_id or ""),
                    workflow_node_id=str(ctx.workflow_node_id),
                    step_id=str(ctx.step_seq),
                    attempt=int(ctx.attempt),
                    trace_id=trace_context.trace_id,
                    span_id=trace_context.span_id,
                    parent_span_id=trace_context.parent_span_id,
                    provider_id=provider.provider_id,
                    client_sse_sequence=client_sequence,
                ),
                conversation_id=conversation_id,
                sender_id=sender_id,
                security_scope=security_scope,
            )
        return _state_update({
            "agent_event_route": "receive",
            "agent_event_provider_sequence": provider_sequence,
            "agent_event_client_sse_sequence": client_sequence,
            "agent_event_advisory_response_hook_count": 0,
        })

    @resolver.register("agent.event.receive")
    async def receive(ctx: StepContext) -> RunSuccess | RunFailure:
        run_id = str(ctx.run_id)
        claim = journal.claim_one(run_id=run_id)
        if claim is None:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["provider event inbox was empty before a terminal event"],
            )
        event = claim.durable_event.event
        hook_count = max(
            0, _state_integer(ctx.state_view, "agent_event_advisory_hook_count")
        )
        advisory_allowed = hook_count < max_advisory_hooks_per_run and (
            (isinstance(event, ProviderFieldCompleted) and event.field_name in advisory_field_names)
            or (isinstance(event, ProviderReasoningSummary) and allow_reasoning_summaries)
        )
        if advisory_allowed:
            route = "advisory"
        elif isinstance(event, ProviderToolCallCompleted):
            route = "tool"
        elif isinstance(event, ProviderResponseCompleted):
            route = "final"
        elif isinstance(event, ProviderResponseFailed):
            route = "failed"
        else:
            route = "no_action"
        return _state_update({
            "agent_event_current": cast(JsonObject, claim.durable_event.model_dump(mode="json")),
            "agent_event_claim": cast(JsonObject, claim.model_dump(mode="json")),
            "agent_event_route": route,
            "agent_event_handling_step_seq": ctx.step_seq,
        })

    @resolver.register("agent.event.hook")
    async def hook(ctx: StepContext) -> RunSuccess | RunFailure:
        event = _event_from_state(ctx.state_view).event
        if not isinstance(event, (ProviderFieldCompleted, ProviderReasoningSummary)):
            return RunFailure(conversation_node_id=None, state_update=[], errors=["advisory hook received wrong event kind"])
        claim = _claim_from_state(ctx.state_view)
        annotations = await _call(
            callbacks.on_advisory_event,
            ctx,
            event,
            idempotency_key=_advisory_idempotency_key(
                str(ctx.run_id), claim.durable_event.event_id
            ),
        )
        _emit_event_handled(ctx, _claim_from_state(ctx.state_view))
        safe_annotations = _bounded_annotations(annotations)
        return _state_update({
            "agent_event_hook_annotations": safe_annotations,
            "agent_event_advisory_hook_count": max(
                0, _state_integer(ctx.state_view, "agent_event_advisory_hook_count")
            ) + 1,
            "agent_event_advisory_response_hook_count": max(
                0,
                _state_integer(
                    ctx.state_view, "agent_event_advisory_response_hook_count"
                ),
            ) + 1,
            "agent_event_handling_step_seq": ctx.step_seq,
        })

    @resolver.register("agent.event.tool")
    async def tool(ctx: StepContext) -> RunSuccess | RunFailure:
        event = _event_from_state(ctx.state_view).event
        if not isinstance(event, ProviderToolCallCompleted):
            return RunFailure(conversation_node_id=None, state_update=[], errors=["tool node received a non-tool event"])
        claim = _claim_from_state(ctx.state_view)
        # The stable key lets an authorized host dispatcher deduplicate a replay.
        key = _tool_idempotency_key(str(ctx.run_id), provider.provider_id, event)
        try:
            pre_tool_annotations = _bounded_annotations(
                await _call(callbacks.before_tool_dispatch, ctx, event)
            )
            result = await _call(
                callbacks.authorize_and_dispatch_tool,
                ctx,
                event,
                idempotency_key=key,
            )
        except Exception as exc:
            journal.dead_letter(
                claim,
                error={"code": "tool_dispatch_failed_ambiguous", "message": type(exc).__name__},
            )
            return RunFailure(conversation_node_id=None, state_update=[], errors=["authorized tool dispatch failed"])
        if not isinstance(result, Mapping):
            return RunFailure(conversation_node_id=None, state_update=[], errors=["authorized tool dispatcher must return an object"])
        safe_result = _bounded_annotations(result)
        try:
            post_tool_annotations = _bounded_annotations(
                await _call(callbacks.after_tool_outcome, ctx, event, result)
            )
        except Exception as exc:
            journal.dead_letter(
                _claim_from_state(ctx.state_view),
                error={"code": "post_tool_hook_failed_after_effect", "message": type(exc).__name__},
            )
            return RunFailure(conversation_node_id=None, state_update=[], errors=["post-tool hook failed after dispatch"])
        _emit_event_handled(ctx, claim)
        return _state_update({
            "agent_event_tool_result": safe_result,
            "agent_event_pre_tool_advice": pre_tool_annotations,
            "agent_event_post_tool_advice": post_tool_annotations,
            "agent_event_handling_step_seq": ctx.step_seq,
        })

    @resolver.register("agent.event.ack")
    async def acknowledge(ctx: StepContext) -> RunSuccess:
        claim = _claim_from_state(ctx.state_view)
        journal.link_handling_step(
            claim,
            conversation_id=conversation_id,
            run_id=str(ctx.run_id),
            step_seq=_state_integer(
                ctx.state_view,
                "agent_event_handling_step_seq",
                default=-1,
            ),
        )
        journal.acknowledge(claim)
        route = str(ctx.state_view.get("agent_event_route") or "no_action")
        next_route = {
            "advisory": "receive",
            "no_action": "receive",
            "tool": "receive",
            "resume": "request",
            "final": "done",
            "failed": "failed_terminal",
        }.get(route, "failed_terminal")
        return _state_update({"agent_event_route": next_route, "agent_event_claim": None})

    @resolver.register("agent.event.finalize")
    async def finalize(ctx: StepContext) -> RunSuccess | RunFailure:
        event = _event_from_state(ctx.state_view).event
        if not isinstance(event, ProviderResponseCompleted):
            return RunFailure(conversation_node_id=None, state_update=[], errors=["finalize node received non-completion event"])
        if event.has_tool_calls:
            return _state_update({
                "agent_event_route": "resume",
                "agent_event_handling_step_seq": ctx.step_seq,
            })
        completed_advisory_hooks = max(
            0,
            _state_integer(ctx.state_view, "agent_event_advisory_response_hook_count"),
        )
        continuations = max(
            0, _state_integer(ctx.state_view, "agent_event_advisory_continuation_count")
        )
        if (
            completed_advisory_hooks > 0
            and continuations < max_advisory_continuations_per_run
        ):
            return _state_update({
                "agent_event_route": "resume",
                "agent_event_advisory_continuation_count": continuations + 1,
                "agent_event_handling_step_seq": ctx.step_seq,
            })
        await _call(
            callbacks.append_assistant_turn,
            ctx,
            event,
            idempotency_key=_event_effect_idempotency_key(
                str(ctx.run_id),
                _event_from_state(ctx.state_view).event_id,
                effect="assistant_turn",
            ),
        )
        _emit_event_handled(ctx, _claim_from_state(ctx.state_view))
        return _state_update({"agent_event_route": "final", "agent_event_handling_step_seq": ctx.step_seq})

    @resolver.register("agent.event.fail")
    async def fail(ctx: StepContext) -> RunSuccess | RunFailure:
        event = _event_from_state(ctx.state_view).event
        if not isinstance(event, ProviderResponseFailed):
            return RunFailure(conversation_node_id=None, state_update=[], errors=["failure node received non-failure event"])
        await _call(
            callbacks.record_failure,
            ctx,
            event,
            idempotency_key=_event_effect_idempotency_key(
                str(ctx.run_id),
                _event_from_state(ctx.state_view).event_id,
                effect="failure_record",
            ),
        )
        _emit_event_handled(ctx, _claim_from_state(ctx.state_view))
        return _state_update({"agent_event_route": "failed", "agent_event_handling_step_seq": ctx.step_seq})

    @resolver.register("agent.event.done")
    async def done(_: StepContext) -> RunSuccess:
        return _state_update({"agent_event_status": "succeeded"})

    @resolver.register("agent.event.failed_terminal")
    async def failed_terminal(_: StepContext) -> RunSuccess:
        return _state_update({"agent_event_status": "failed"})

    return {
        "event_is_advisory": lambda _edge, state, _result: state.get("agent_event_route") == "advisory",
        "event_is_tool_call": lambda _edge, state, _result: state.get("agent_event_route") == "tool",
        "event_is_response_complete": lambda _edge, state, _result: state.get("agent_event_route") == "final",
        "event_is_response_failed": lambda _edge, state, _result: state.get("agent_event_route") == "failed",
        "event_requires_no_action": lambda _edge, state, _result: state.get("agent_event_route") == "no_action",
        "event_stream_has_more": lambda _edge, state, _result: state.get("agent_event_route") == "receive",
        "event_turn_requires_model_resume": lambda _edge, state, _result: state.get("agent_event_route") == "request",
        "event_run_succeeded": lambda _edge, state, _result: state.get("agent_event_route") == "done",
        "event_run_failed": lambda _edge, state, _result: state.get("agent_event_route") == "failed_terminal",
    }


__all__ = [
    "EventWorkflowCallbacks",
    "_event_effect_idempotency_key",
    "register_event_workflow_steps",
]
