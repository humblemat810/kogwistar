from __future__ import annotations

import asyncio
import json
from typing import Any, cast, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from .chat_service import WorkflowProjectionRebuildingError, ChatRunService


class SubmitWorkflowRunIn(BaseModel):
    workflow_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_node_id: str | None = None
    user_id: str | None = None
    initial_state: dict[str, Any] = Field(default_factory=dict)


class UpsertWorkflowNodeIn(BaseModel):
    designer_id: str = Field(min_length=1)
    node_id: str | None = None
    label: str = Field(min_length=1)
    op: str | None = None
    start: bool = False
    terminal: bool = False
    fanout: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertWorkflowEdgeIn(BaseModel):
    designer_id: str = Field(min_length=1)
    edge_id: str | None = None
    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    relation: str = "wf_next"
    predicate: str | None = None
    priority: int = 100
    is_default: bool = False
    multiplicity: str = "one"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DesignActorIn(BaseModel):
    designer_id: str = Field(min_length=1)


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, WorkflowProjectionRebuildingError):
        return HTTPException(status_code=409, detail=str(exc))
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
    get_subject: Callable[[], str | None] | None = None,
):
    router = APIRouter(prefix="/api/workflow", tags=["runtime"])
    get_service_r = cast(Callable[[], ChatRunService], get_service)
    @router.post("/runs")
    def submit_workflow_run(inp: SubmitWorkflowRunIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service_r().submit_workflow_run(
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
            return get_service_r().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/events")
    async def get_workflow_run_events(run_id: str, after_seq: int = 0):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            get_service_r().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

        async def event_stream():
            last_seq = int(after_seq or 0)
            while True:
                service = get_service_r()
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
            payload = get_service_r().cancel_run(run_id)
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
                "steps": get_service_r().list_steps(run_id),
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
                "checkpoints": get_service_r().list_checkpoints(run_id),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/checkpoints/{step_seq}")
    def get_workflow_checkpoint(run_id: str, step_seq: int):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().get_checkpoint(run_id, step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/replay")
    def replay_workflow_run(run_id: str, target_step_seq: int):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().replay_run(run_id, target_step_seq)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/nodes")
    def upsert_workflow_node(workflow_id: str, inp: UpsertWorkflowNodeIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id=inp.designer_id,
                node_id=inp.node_id,
                label=inp.label,
                op=inp.op,
                start=inp.start,
                terminal=inp.terminal,
                fanout=inp.fanout,
                metadata=inp.metadata,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/edges")
    def upsert_workflow_edge(workflow_id: str, inp: UpsertWorkflowEdgeIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_upsert_edge(
                workflow_id=workflow_id,
                designer_id=inp.designer_id,
                edge_id=inp.edge_id,
                src=inp.src,
                dst=inp.dst,
                relation=inp.relation,
                predicate=inp.predicate,
                priority=inp.priority,
                is_default=inp.is_default,
                multiplicity=inp.multiplicity,
                metadata=inp.metadata,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.delete("/design/{workflow_id}/nodes/{node_id}")
    def delete_workflow_node(workflow_id: str, node_id: str, inp: DesignActorIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_delete_node(
                workflow_id=workflow_id,
                node_id=node_id,
                designer_id=inp.designer_id,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.delete("/design/{workflow_id}/edges/{edge_id}")
    def delete_workflow_edge(workflow_id: str, edge_id: str, inp: DesignActorIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_delete_edge(
                workflow_id=workflow_id,
                edge_id=edge_id,
                designer_id=inp.designer_id,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/design/{workflow_id}/history")
    def workflow_design_history(workflow_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().workflow_design_history(workflow_id=workflow_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/undo")
    def workflow_design_undo(workflow_id: str, inp: DesignActorIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_undo(
                workflow_id=workflow_id,
                designer_id=inp.designer_id,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/redo")
    def workflow_design_redo(workflow_id: str, inp: DesignActorIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            actor_sub = get_subject() if callable(get_subject) else None
            return get_service_r().workflow_design_redo(
                workflow_id=workflow_id,
                designer_id=inp.designer_id,
                actor_sub=actor_sub,
                source="rest",
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)
        
    @router.get("/api/workflow/design/{workflow_id}/graph")
    def workflow_design_graph(workflow_id: str, refresh: bool = False):
        chat_service = get_service_r()
        return chat_service.workflow_design_graph(workflow_id=workflow_id, refresh=refresh)

    @router.get("/api/workflow/catalog/ops")
    def workflow_catalog_ops():
        chat_service = get_service_r()
        return chat_service.workflow_catalog_ops()

    return router
