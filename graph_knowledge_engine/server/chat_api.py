from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class CreateConversationIn(BaseModel):
    user_id: str
    conversation_id: str | None = None
    start_node_id: str | None = None


class SubmitAnswerIn(BaseModel):
    user_id: str | None = None
    text: str = Field(min_length=1)
    workflow_id: str = "agentic_answering.v2"


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(status_code=500, detail=str(exc))


def _sse_frame(*, event_type: str, seq: int, payload: dict[str, Any]) -> str:
    return f"id: {seq}\nevent: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_chat_router(
    *,
    get_service: Callable[[], Any],
    require_role: Callable[[str], None],
    require_namespace: Callable[[Any], None],
    conversation_namespace: Any,
    workflow_namespaces: Any,
    get_user_id: Callable[[], str | None] | None = None,
):
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.post("/conversations")
    def create_conversation(inp: CreateConversationIn):
        require_role("rw")
        require_namespace(conversation_namespace)
        try:
            effective_user_id = (get_user_id() if callable(get_user_id) else None) or inp.user_id
            return get_service().create_conversation(
                user_id=effective_user_id,
                conversation_id=inp.conversation_id,
                start_node_id=inp.start_node_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/conversations")
    def list_conversations():
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            effective_user_id = get_user_id() if callable(get_user_id) else None
            if not effective_user_id:
                raise HTTPException(status_code=401, detail="User Identity required")
            return {"conversations": get_service().list_conversations_for_user(user_id=effective_user_id)}
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            return get_service().get_conversation(conversation_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/conversations/{conversation_id}/turns")
    def get_transcript(conversation_id: str):
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            return {
                "conversation_id": conversation_id,
                "turns": get_service().list_transcript(conversation_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/conversations/{conversation_id}/turns:answer")
    def submit_answer(conversation_id: str, inp: SubmitAnswerIn):
        require_role("rw")
        require_namespace(conversation_namespace)
        try:
            effective_user_id = (get_user_id() if callable(get_user_id) else None) or inp.user_id
            payload = get_service().submit_turn_for_answer(
                conversation_id=conversation_id,
                user_id=effective_user_id,
                text=inp.text,
                workflow_id=inp.workflow_id,
            )
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/conversations/{conversation_id}/snapshots/latest")
    def latest_snapshot(conversation_id: str, run_id: str | None = None, stage: str | None = None):
        require_role("ro")
        require_namespace(workflow_namespaces)
        try:
            return get_service().latest_snapshot(conversation_id, run_id=run_id, stage=stage)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            return get_service().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/events")
    async def get_run_events(run_id: str, after_seq: int = 0):
        require_role("ro")
        require_namespace(conversation_namespace)
        try:
            get_service().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

        async def event_stream():
            last_seq = int(after_seq or 0)
            while True:
                service = get_service()
                events = service.list_run_events(run_id, after_seq=last_seq)
                for evt in events:
                    last_seq = int(evt["seq"])
                    payload = {
                        "run_id": run_id,
                        "event_type": evt["event_type"],
                        "created_at_ms": evt["created_at_ms"],
                        **(evt["payload"] or {}),
                    }
                    yield _sse_frame(event_type=evt["event_type"], seq=last_seq, payload=payload)
                run = service.get_run(run_id)
                if run["terminal"] and not events:
                    break
                await asyncio.sleep(0.1)

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    @router.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        require_role("rw")
        require_namespace(conversation_namespace)
        try:
            payload = get_service().cancel_run(run_id)
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/steps")
    def get_run_steps(run_id: str):
        require_role("ro")
        require_namespace(workflow_namespaces)
        try:
            return {
                "run_id": run_id,
                "steps": get_service().list_steps(run_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/checkpoints")
    def get_run_checkpoints(run_id: str):
        require_role("ro")
        require_namespace(workflow_namespaces)
        try:
            return {
                "run_id": run_id,
                "checkpoints": get_service().list_checkpoints(run_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/checkpoints/{step_seq}")
    def get_checkpoint(run_id: str, step_seq: int):
        require_role("ro")
        require_namespace(workflow_namespaces)
        try:
            return get_service().get_checkpoint(run_id, step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/replay")
    def replay_run(run_id: str, target_step_seq: int):
        require_role("ro")
        require_namespace(workflow_namespaces)
        try:
            return get_service().replay_run(run_id, target_step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    return router
