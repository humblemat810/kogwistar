"""Opt-in authenticated REST/SSE surface for graph-native event-agent runs."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Literal, Protocol, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from kogwistar.agent.streaming import ProviderCapabilities
from kogwistar.json_types import JsonObject, JsonValue

_SAFE_SSE_EVENT_TYPE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")


class StartEventAgentRunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    input: str = Field(min_length=1)


class SubmitEventAgentInputIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    requested_mode: Literal["steer", "queue", "cancel_and_continue"] = "queue"


class EventAgentControlReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted", "committed", "pending", "failed"]
    effective_mode: Literal["steer", "queue", "cancel_and_continue"]
    control_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    note: str | None = None


class EventAgentControlCapabilities(BaseModel):
    """Provider features intersected with features implemented by the run host."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderCapabilities
    live_steering: bool = False
    cancel_and_continue: bool = False


def resolve_control_mode(
    requested_mode: Literal["steer", "queue", "cancel_and_continue"],
    capabilities: EventAgentControlCapabilities,
) -> Literal["steer", "queue", "cancel_and_continue"]:
    """Allow live control only when both provider and event host support it."""

    if requested_mode == "steer" and not (
        capabilities.provider.steering and capabilities.live_steering
    ):
        return "queue"
    if requested_mode == "cancel_and_continue" and not (
        capabilities.provider.cancellation and capabilities.cancel_and_continue
    ):
        return "queue"
    return requested_mode


class EventAgentRunService(Protocol):
    """Injected run registry; tool dispatch and provider config stay server-side."""

    def start_run(self, *, conversation_id: str, workflow_id: str, user_id: str, input: str) -> object: ...

    def submit_user_input(
        self, *, run_id: str, user_id: str, text: str, requested_mode: str
    ) -> object: ...

    def get_control_receipt(
        self, *, run_id: str, control_id: str, user_id: str
    ) -> object: ...

    def get_control_capabilities(
        self, *, run_id: str, user_id: str
    ) -> EventAgentControlCapabilities: ...

    def get_run(self, *, run_id: str, user_id: str) -> Mapping[str, object]: ...

    def list_events(self, *, run_id: str, user_id: str, after_seq: int, limit: int) -> Sequence[Mapping[str, object]]: ...

    def list_transient_deltas(
        self, *, run_id: str, user_id: str, after_sequence: int, limit: int
    ) -> Sequence[Mapping[str, object]]: ...


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="event-agent run not found")
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail="event-agent access denied")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc)[:300])
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(status_code=500, detail="event-agent operation failed")


def _frame(*, event_type: str, sequence: str | int, payload: Mapping[str, JsonValue]) -> str:
    data = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    safe_type = event_type if _SAFE_SSE_EVENT_TYPE.fullmatch(event_type) else "message"
    return f"id: {sequence}\nevent: {safe_type}\ndata: {data}\n\n"


def _control_receipt(value: object) -> EventAgentControlReceipt:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    return EventAgentControlReceipt.model_validate(value)


def create_agent_event_router(
    *,
    get_service: Callable[[], EventAgentRunService],
    require_role: Callable[[str], None],
    require_namespace: Callable[[object], None],
    conversation_namespace: object,
    get_user_id: Callable[[], str | None],
) -> APIRouter:
    """Create independent ``/api/agent-events`` routes; they are not mounted by default."""

    router = APIRouter(prefix="/api/agent-events", tags=["agent-events"])

    def _user() -> str:
        user_id = str(get_user_id() or "").strip()
        if not user_id:
            raise HTTPException(status_code=401, detail="User identity required")
        return user_id

    @router.post("/runs", status_code=202)
    def start_run(inp: StartEventAgentRunIn) -> object:
        require_role("rw")
        require_namespace(conversation_namespace)
        try:
            return get_service().start_run(
                conversation_id=inp.conversation_id,
                workflow_id=inp.workflow_id,
                user_id=_user(),
                input=inp.input,
            )
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.post("/runs/{run_id}/input", status_code=202)
    def submit_input(run_id: str, inp: SubmitEventAgentInputIn) -> object:
        require_role("rw")
        require_namespace(conversation_namespace)
        try:
            service = get_service()
            user_id = _user()
            run = service.get_run(run_id=run_id, user_id=user_id)
            if bool(run.get("terminal")):
                raise HTTPException(status_code=409, detail="event-agent run is already terminal")
            capabilities = EventAgentControlCapabilities.model_validate(
                service.get_control_capabilities(run_id=run_id, user_id=user_id)
            )
            effective_mode = resolve_control_mode(inp.requested_mode, capabilities)
            result = service.submit_user_input(
                run_id=run_id,
                user_id=user_id,
                text=inp.text,
                requested_mode=effective_mode,
            )
            receipt = _control_receipt(result)
            if receipt.effective_mode != effective_mode:
                raise HTTPException(
                    status_code=502,
                    detail="run service control mode did not match negotiated provider capability",
                )
            if receipt.status == "failed":
                raise HTTPException(status_code=409, detail="input could not be applied")
            if effective_mode != inp.requested_mode and receipt.note is None:
                receipt = receipt.model_copy(
                    update={"note": f"{inp.requested_mode} is unsupported; input queued at a safe boundary"}
                )
            return receipt.model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/runs/{run_id}/input/{control_id}")
    def inspect_input(run_id: str, control_id: str) -> object:
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            service = get_service()
            user_id = _user()
            # Authorize ownership of the run before looking up its control ID.
            service.get_run(run_id=run_id, user_id=user_id)
            raw_receipt = service.get_control_receipt(
                run_id=run_id, control_id=control_id, user_id=user_id
            )
            if raw_receipt is None:
                raise HTTPException(status_code=404, detail="event-agent input not found")
            receipt = _control_receipt(raw_receipt)
            return receipt.model_dump(mode="json")
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/runs/{run_id}")
    def inspect_run(run_id: str) -> object:
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            return get_service().get_run(run_id=run_id, user_id=_user())
        except Exception as exc:
            raise _http_error(exc) from exc

    @router.get("/runs/{run_id}/events")
    async def stream_events(
        run_id: str, after_seq: int = 0, after_client_sequence: int = 0
    ) -> StreamingResponse:
        require_role("ro")
        require_namespace(conversation_namespace)
        user_id = _user()
        service = get_service()
        try:
            service.get_run(run_id=run_id, user_id=user_id)
        except Exception as exc:
            raise _http_error(exc) from exc

        async def stream() -> AsyncIterator[str]:
            last_sequence = max(int(after_seq), 0)
            last_client_sequence = max(int(after_client_sequence), 0)
            while True:
                events = service.list_events(
                    run_id=run_id,
                    user_id=user_id,
                    after_seq=last_sequence,
                    limit=500,
                )
                delivered = False
                for event in events:
                    sequence = event.get("sse_sequence")
                    if (
                        not isinstance(sequence, int)
                        or isinstance(sequence, bool)
                        or sequence <= last_sequence
                    ):
                        continue
                    event_type = str(event.get("event_type") or "message")
                    payload_raw = event.get("payload")
                    payload: dict[str, JsonValue] = {}
                    if isinstance(payload_raw, Mapping):
                        payload.update(cast(JsonObject, dict(payload_raw)))
                    payload["run_id"] = run_id
                    for sequence_key in (
                        "turn_id",
                        "provider_sequence",
                        "provider_event_id",
                        "durable_event_id",
                        "durable_message_id",
                        "response_id",
                        "trace_id",
                        "span_id",
                        "parent_span_id",
                        "workflow_node_id",
                        "workflow_step_seq",
                        "attempt",
                        "producer_step_seq",
                        "producer_attempt",
                        "provider_id",
                        "provider_response_id",
                    ):
                        value = event.get(sequence_key)
                        if isinstance(value, (str, int)) and not isinstance(value, bool):
                            payload[sequence_key] = value
                    last_sequence = sequence
                    delivered = True
                    yield _frame(event_type=event_type, sequence=sequence, payload=payload)
                deltas = service.list_transient_deltas(
                    run_id=run_id,
                    user_id=user_id,
                    after_sequence=last_client_sequence,
                    limit=500,
                )
                for delta_event in deltas:
                    sequence = delta_event.get("client_sse_sequence")
                    delta = delta_event.get("delta")
                    if (
                        not isinstance(sequence, int)
                        or isinstance(sequence, bool)
                        or sequence <= last_client_sequence
                    ):
                        continue
                    delta_event_type = str(
                        delta_event.get("event_type") or "response.text.delta"
                    )
                    if delta_event_type == "response.text.delta_gap":
                        dropped_through = delta_event.get("dropped_through_sequence")
                        last_client_sequence = sequence
                        delivered = True
                        yield _frame(
                            event_type=delta_event_type,
                            sequence=f"delta-gap-{sequence}",
                            payload={
                                "run_id": run_id,
                                "client_sse_sequence": sequence,
                                "dropped_through_sequence": (
                                    dropped_through
                                    if isinstance(dropped_through, int)
                                    and not isinstance(dropped_through, bool)
                                    else sequence
                                ),
                            },
                        )
                        continue
                    if not isinstance(delta, str):
                        continue
                    last_client_sequence = sequence
                    delivered = True
                    yield _frame(
                        event_type=delta_event_type,
                        sequence=f"delta-{sequence}",
                        payload={
                            "run_id": run_id,
                            "client_sse_sequence": sequence,
                            "delta": delta,
                        },
                    )
                run = service.get_run(run_id=run_id, user_id=user_id)
                if bool(run.get("terminal")) and not delivered:
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    return router


__all__ = [
    "EventAgentRunService",
    "EventAgentControlCapabilities",
    "EventAgentControlReceipt",
    "StartEventAgentRunIn",
    "SubmitEventAgentInputIn",
    "create_agent_event_router",
    "resolve_control_mode",
]
