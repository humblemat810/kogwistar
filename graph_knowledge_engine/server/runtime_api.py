from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class SubmitWorkflowRunIn(BaseModel):
    workflow_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_node_id: str | None = None
    user_id: str | None = None
    initial_state: dict[str, Any] = Field(default_factory=dict)


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


def create_runtime_router(
    *,
    get_service: Callable[[], Any],
    require_role: Callable[[str], None],
    require_namespace: Callable[[Any], None],
    runtime_namespaces: Any,
):
    router = APIRouter(prefix="/api/workflow", tags=["runtime"])

    @router.post("/runs")
    def submit_workflow_run(inp: SubmitWorkflowRunIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service().submit_workflow_run(
                workflow_id=inp.workflow_id,
                conversation_id=inp.conversation_id,
                turn_node_id=inp.turn_node_id,
                user_id=inp.user_id,
                initial_state=inp.initial_state,
            )
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}")
    def get_workflow_run(run_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/events")
    async def get_workflow_run_events(run_id: str, after_seq: int = 0):
        require_role("ro")
        require_namespace(runtime_namespaces)
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
    def cancel_workflow_run(run_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service().cancel_run(run_id)
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/steps")
    def get_workflow_steps(run_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return {
                "run_id": run_id,
                "steps": get_service().list_steps(run_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/checkpoints")
    def get_workflow_checkpoints(run_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return {
                "run_id": run_id,
                "checkpoints": get_service().list_checkpoints(run_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/checkpoints/{step_seq}")
    def get_workflow_checkpoint(run_id: str, step_seq: int):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service().get_checkpoint(run_id, step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/replay")
    def replay_workflow_run(run_id: str, target_step_seq: int):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service().replay_run(run_id, target_step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    return router
