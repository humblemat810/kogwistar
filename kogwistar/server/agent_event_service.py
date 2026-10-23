"""Run-registry and scheduler binding for the opt-in event-agent API."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import TypeAdapter

from kogwistar.agent.event_control import (
    EventControlMode,
    EventControlSubmission,
)
from kogwistar.agent.streaming import ProviderCapabilities
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.server.run_registry import RunRegistry

from .agent_event_api import (
    EventAgentControlCapabilities,
    resolve_control_mode,
)

_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_MAX_EVENT_PAYLOAD_CHARS = 64_000
_MAX_TRANSIENT_RUNS = 64
_MAX_TRANSIENT_DELTAS_PER_RUN = 64
_MAX_TRANSIENT_DELTA_CHARS = 4_096
_SAFE_ERROR_CODE = re.compile(r"^[a-zA-Z0-9_.-]{1,80}$")
_SAFE_EVENT_TYPE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")
_TRACE_PROJECTION_FIELDS = (
    "turn_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "workflow_node_id",
    "workflow_step_seq",
    "producer_step_seq",
    "producer_attempt",
    "attempt",
    "provider_id",
    "provider_response_id",
    "provider_sequence",
    "provider_event_id",
    "durable_event_id",
    "durable_message_id",
)


def _project_trace_fields(payload: Mapping[str, object]) -> JsonObject:
    trace = payload.get("trace")
    sources = (payload, trace if isinstance(trace, Mapping) else {})
    projected: JsonObject = {}
    for key in _TRACE_PROJECTION_FIELDS:
        value = next((source[key] for source in sources if key in source), None)
        if isinstance(value, str) and value and len(value) <= 256:
            projected[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            projected[key] = value
    return projected


@dataclass(frozen=True, slots=True)
class EventAgentRunContext:
    run_id: str
    conversation_id: str
    workflow_id: str
    user_id: str
    user_turn_node_id: str
    input: str


class EventAgentRunScheduler(Protocol):
    def submit(
        self,
        *,
        run_id: str,
        priority_class: str,
        start_fn: Callable[[], None],
        max_retries: int = 0,
    ) -> Mapping[str, object]: ...


class EventAgentControlJournalLike(Protocol):
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
    ) -> EventControlSubmission: ...

    def get_submission(
        self, *, run_id: str, user_id: str, control_id: str
    ) -> EventControlSubmission | None: ...


CreateUserTurn = Callable[[str, str, str, str], str]
AuthorizeStart = Callable[[str, str, str], None]
RunExecutor = Callable[
    [EventAgentRunContext, "EventAgentPublishCallbacks"],
    Awaitable[Mapping[str, object]],
]
PublishSemanticEvent = Callable[[str, Mapping[str, object]], JsonObject]
PublishTransientDelta = Callable[[EventAgentRunContext, str, int], object]
ProviderCapabilitiesForRun = Callable[[str, str], ProviderCapabilities]
ControlRecipientForRun = Callable[[str, str], str]
SecurityScopeForRun = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class EventAgentPublishCallbacks:
    """Separate durable semantic events from ephemeral client token deltas."""

    publish_semantic_event: PublishSemanticEvent
    publish_transient_delta: Callable[[str, int], object]


class ScheduledEventAgentRunService:
    """Concrete opt-in service over existing registry, scheduler, and lane inbox.

    The host supplies conversation authorization, user-turn creation, provider
    workflow execution, and the run's recipient/security scope. The event loop
    does not call ``ChatRunService`` or run the conversation/RAG workflow.
    ``RunScheduler`` is an in-process scheduler; durable run/event records do
    not imply that queued executions resume automatically after process loss.
    """

    def __init__(
        self,
        *,
        registry: RunRegistry,
        scheduler: EventAgentRunScheduler,
        control_journal: EventAgentControlJournalLike,
        authorize_start: AuthorizeStart,
        create_user_turn: CreateUserTurn,
        execute_run: RunExecutor,
        publish_transient_delta: PublishTransientDelta | None = None,
        provider_capabilities: ProviderCapabilitiesForRun,
        control_recipient: ControlRecipientForRun,
        security_scope: SecurityScopeForRun,
        priority_class: str = "foreground",
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler
        self._control_journal = control_journal
        self._authorize_start = authorize_start
        self._create_user_turn = create_user_turn
        self._execute_run = execute_run
        self._publish_transient_delta = publish_transient_delta
        self._transient_lock = threading.Lock()
        self._transient_deltas: dict[str, deque[JsonObject]] = {}
        self._provider_capabilities = provider_capabilities
        self._control_recipient = control_recipient
        self._security_scope = security_scope
        self._priority_class = str(priority_class or "foreground")

    def start_run(
        self, *, conversation_id: str, workflow_id: str, user_id: str, input: str
    ) -> JsonObject:
        values = {
            "conversation_id": str(conversation_id).strip(),
            "workflow_id": str(workflow_id).strip(),
            "user_id": str(user_id).strip(),
            "input": str(input).strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("conversation, workflow, user, and input are required")
        self._authorize_start(
            values["conversation_id"], values["workflow_id"], values["user_id"]
        )
        run_id = str(uuid.uuid4())
        turn_id = str(
            self._create_user_turn(
                run_id,
                values["conversation_id"],
                values["user_id"],
                values["input"],
            )
        ).strip()
        if not turn_id:
            raise ValueError("host user-turn callback returned an empty turn ID")
        context = EventAgentRunContext(
            run_id=run_id,
            conversation_id=values["conversation_id"],
            workflow_id=values["workflow_id"],
            user_id=values["user_id"],
            user_turn_node_id=turn_id,
            input=values["input"],
        )
        self._registry.create_run(
            run_id=run_id,
            conversation_id=context.conversation_id,
            workflow_id=context.workflow_id,
            user_id=context.user_id,
            user_turn_node_id=context.user_turn_node_id,
            status="queued",
        )
        self._append_event(
            run_id,
            "run.created",
            {
                "run_id": run_id,
                "run_kind": "event_agent",
                "conversation_id": context.conversation_id,
                "workflow_id": context.workflow_id,
                "user_turn_node_id": context.user_turn_node_id,
                "status": "queued",
            },
        )
        schedule_failed = False
        try:
            admission = self._scheduler.submit(
                run_id=run_id,
                priority_class=self._priority_class,
                start_fn=lambda: self._execute(context),
                max_retries=0,
            )
        except Exception as exc:
            self._finish_failed(run_id, "scheduler_submit_failed", type(exc).__name__)
            admission = {"admission": "rejected", "reason": "scheduler_error"}
            schedule_failed = True
        if not schedule_failed and str(admission.get("admission") or "") != "accepted":
            reason = str(admission.get("reason") or "scheduler_rejected")
            self._finish_failed(run_id, "run_not_scheduled", reason)
        run = self._registry.get_run(run_id) or {}
        return {
            "run_id": run_id,
            "conversation_id": context.conversation_id,
            "workflow_id": context.workflow_id,
            "user_turn_node_id": context.user_turn_node_id,
            "status": str(run.get("status") or "queued"),
            "terminal": bool(run.get("terminal")),
            "admission": str(admission.get("admission") or "rejected"),
            **({"reason": str(admission["reason"])} if admission.get("reason") else {}),
        }

    def _execute(self, context: EventAgentRunContext) -> None:
        try:
            self._registry.update_status(context.run_id, status="running", started=True)
            self._append_event(
                context.run_id,
                "run.started",
                {"run_id": context.run_id, "run_kind": "event_agent", "status": "running"},
            )
            async def execute_workflow() -> Mapping[str, object]:
                return await self._execute_run(
                    context,
                    EventAgentPublishCallbacks(
                        publish_semantic_event=lambda event_type, payload: self._append_event(
                            context.run_id, event_type, payload
                        ),
                        publish_transient_delta=lambda delta, sequence: self._publish_delta(
                            context, delta, sequence
                        ),
                    ),
                )

            result = asyncio.run(execute_workflow())
            status = str(result.get("status") or result.get("workflow_status") or "")
            if status == "succeeded":
                self._registry.update_status(
                    context.run_id,
                    status="succeeded",
                    result={"status": "succeeded"},
                    finished=True,
                )
                self._append_event(
                    context.run_id,
                    "run.completed",
                    {"run_id": context.run_id, "status": "succeeded"},
                )
            elif status in {"failed", "cancelled"}:
                self._finish_failed(
                    context.run_id,
                    "event_workflow_failed" if status == "failed" else "event_workflow_cancelled",
                    status,
                    status=status,
                )
            else:
                self._finish_failed(context.run_id, "invalid_workflow_result", "invalid_status")
        except Exception as exc:
            self._finish_failed(
                context.run_id,
                "event_workflow_exception",
                type(exc).__name__,
            )

    def _finish_failed(
        self,
        run_id: str,
        code: str,
        detail: str,
        *,
        status: str = "failed",
    ) -> None:
        safe_code = code if _SAFE_ERROR_CODE.fullmatch(code) else "event_workflow_failed"
        self._registry.update_status(
            run_id,
            status=status,
            error={"code": safe_code, "detail": detail[:120]},
            finished=True,
        )
        self._append_event(
            run_id,
            "run.cancelled" if status == "cancelled" else "run.failed",
            {"run_id": run_id, "status": status, "error_code": safe_code},
        )

    def _append_event(
        self, run_id: str, event_type: str, payload: Mapping[str, object]
    ) -> JsonObject:
        if not _SAFE_EVENT_TYPE.fullmatch(event_type):
            raise ValueError("event type contains unsupported characters")
        try:
            safe_payload = _JSON_OBJECT_ADAPTER.validate_python(dict(payload))
            serialized = json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("event payload must be a JSON object") from exc
        if len(serialized) > _MAX_EVENT_PAYLOAD_CHARS:
            raise ValueError("event payload exceeds the 64,000-character limit")
        return self._registry.append_event(run_id, event_type, cast(JsonObject, safe_payload))

    async def _publish_delta(
        self, context: EventAgentRunContext, delta: str, sequence: int
    ) -> None:
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise ValueError("client delta sequence must be a positive integer")
        if not isinstance(delta, str) or len(delta) > _MAX_TRANSIENT_DELTA_CHARS:
            raise ValueError("transient text delta exceeds its size limit")
        row: JsonObject = {
            "client_sse_sequence": sequence,
            "delta": delta,
        }
        with self._transient_lock:
            if context.run_id not in self._transient_deltas and len(
                self._transient_deltas
            ) >= _MAX_TRANSIENT_RUNS:
                oldest_run_id = next(iter(self._transient_deltas))
                del self._transient_deltas[oldest_run_id]
            entries = self._transient_deltas.setdefault(
                context.run_id, deque(maxlen=_MAX_TRANSIENT_DELTAS_PER_RUN)
            )
            previous_sequence = (
                entries[-1].get("client_sse_sequence") if entries else None
            )
            if (
                isinstance(previous_sequence, int)
                and not isinstance(previous_sequence, bool)
                and sequence <= previous_sequence
            ):
                # Retries can replay prior deltas; don't duplicate them to clients.
                return
            entries.append(row)
        if self._publish_transient_delta is not None:
            try:
                result = self._publish_transient_delta(context, delta, sequence)
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            except Exception:
                # Client streaming is observational; it must not fail graph execution.
                pass

    def list_transient_deltas(
        self, *, run_id: str, user_id: str, after_sequence: int, limit: int
    ) -> list[JsonObject]:
        self._owned_run(run_id=run_id, user_id=user_id)
        with self._transient_lock:
            entries = tuple(self._transient_deltas.get(run_id, ()))
        output: list[JsonObject] = []
        if entries:
            oldest_sequence = entries[0].get("client_sse_sequence")
            if (
                isinstance(oldest_sequence, int)
                and not isinstance(oldest_sequence, bool)
                and after_sequence < oldest_sequence - 1
            ):
                output.append(
                    {
                        "event_type": "response.text.delta_gap",
                        "client_sse_sequence": oldest_sequence - 1,
                        "dropped_through_sequence": oldest_sequence - 1,
                    }
                )
        for row in entries:
            sequence = row.get("client_sse_sequence")
            if (
                isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence > max(0, after_sequence)
            ):
                output.append(dict(row))
                if len(output) >= min(max(1, limit), _MAX_TRANSIENT_DELTAS_PER_RUN + 1):
                    break
        return output

    def _owned_run(self, *, run_id: str, user_id: str) -> JsonObject:
        run = self._registry.get_run(str(run_id))
        if run is None:
            raise KeyError(run_id)
        if str(run.get("user_id") or "") != str(user_id):
            raise PermissionError("run owner mismatch")
        return run

    def get_run(self, *, run_id: str, user_id: str) -> JsonObject:
        run = self._owned_run(run_id=run_id, user_id=user_id)
        result: JsonObject = {
            key: cast(JsonValue, run[key])
            for key in (
                "run_id",
                "conversation_id",
                "workflow_id",
                "user_turn_node_id",
                "status",
                "terminal",
                "created_at_ms",
                "started_at_ms",
                "finished_at_ms",
            )
            if key in run
        }
        rows = self._registry.list_events(run_id, after_seq=0, limit=500)
        for row in reversed(rows):
            payload = row.get("payload")
            if isinstance(payload, Mapping):
                trace = _project_trace_fields(payload)
                if trace:
                    result["trace"] = trace
                    break
        return result

    def get_control_capabilities(
        self, *, run_id: str, user_id: str
    ) -> EventAgentControlCapabilities:
        run = self._owned_run(run_id=run_id, user_id=user_id)
        provider = self._provider_capabilities(str(run_id), str(user_id))
        # The v1 workflow only consumes queued input at a safe request boundary.
        return EventAgentControlCapabilities(
            provider=provider,
            live_steering=False,
            cancel_and_continue=False,
        )

    def submit_user_input(
        self, *, run_id: str, user_id: str, text: str, requested_mode: str
    ) -> EventControlSubmission:
        run = self._owned_run(run_id=run_id, user_id=user_id)
        if bool(run.get("terminal")):
            raise ValueError("event-agent run is terminal")
        if requested_mode not in {"steer", "queue", "cancel_and_continue"}:
            raise ValueError("unsupported control mode")
        effective_mode = resolve_control_mode(
            cast(EventControlMode, requested_mode),
            self.get_control_capabilities(run_id=run_id, user_id=user_id),
        )
        return self._control_journal.submit(
            run_id=str(run_id),
            conversation_id=str(run.get("conversation_id") or ""),
            user_id=str(user_id),
            recipient_id=self._control_recipient(str(run_id), str(user_id)),
            control_id=str(uuid.uuid4()),
            text=text,
            effective_mode=effective_mode,
            security_scope=self._security_scope(str(run_id), str(user_id)),
        )

    def get_control_receipt(
        self, *, run_id: str, control_id: str, user_id: str
    ) -> EventControlSubmission:
        self._owned_run(run_id=run_id, user_id=user_id)
        receipt = self._control_journal.get_submission(
            run_id=run_id, user_id=user_id, control_id=control_id
        )
        if receipt is None:
            raise KeyError(control_id)
        return receipt

    def list_events(
        self, *, run_id: str, user_id: str, after_seq: int, limit: int
    ) -> list[JsonObject]:
        self._owned_run(run_id=run_id, user_id=user_id)
        rows = self._registry.list_events(
            run_id,
            after_seq=max(0, int(after_seq)),
            limit=min(max(1, int(limit)), 500),
        )
        output: list[JsonObject] = []
        for row in rows:
            sequence = row.get("seq")
            if not isinstance(sequence, int) or isinstance(sequence, bool):
                continue
            event: JsonObject = {
                "sse_sequence": sequence,
                "event_type": str(row.get("event_type") or "message"),
                "payload": cast(JsonValue, row.get("payload") or {}),
            }
            payload = row.get("payload")
            if isinstance(payload, Mapping):
                event.update(_project_trace_fields(payload))
            output.append(event)
        return output


__all__ = [
    "EventAgentControlJournalLike",
    "EventAgentPublishCallbacks",
    "EventAgentRunContext",
    "EventAgentRunScheduler",
    "RunExecutor",
    "ScheduledEventAgentRunService",
]
