from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, cast, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from .chat_service import WorkflowProjectionRebuildingError, ChatRunService
from .error_reporting import internal_http_error


class SubmitWorkflowRunIn(BaseModel):
    workflow_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_node_id: str | None = None
    user_id: str | None = None
    initial_state: dict[str, Any] = Field(default_factory=dict)
    priority_class: str = "foreground"
    token_budget: int | None = None
    time_budget_ms: int | None = None


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


class ResumeRunIn(BaseModel):
    suspended_node_id: str = Field(min_length=1)
    suspended_token_id: str = Field(min_length=1)
    client_result: dict[str, Any] = Field(default_factory=dict)
    workflow_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn_node_id: str = Field(min_length=1)
    user_id: str | None = None


class CapabilityApproveIn(BaseModel):
    action: str = Field(min_length=1)
    capabilities: list[str] | str = Field(default_factory=list)
    subject: str | None = None


class CapabilityRevokeIn(BaseModel):
    capability: str = Field(min_length=1)
    subject: str | None = None


class ResourceSnapshotOut(BaseModel):
    scheduler: dict[str, Any]
    runs: dict[str, Any]
    budget_model: dict[str, Any]


class ToolAuditOut(BaseModel):
    conversation_id: str | None = None
    items: list[dict[str, Any]]


class ServiceTriggerSpecIn(BaseModel):
    type: str = Field(min_length=1)
    enabled: bool = True
    selector: dict[str, Any] = Field(default_factory=dict)
    debounce_ms: int = 0
    cooldown_ms: int = 0


class ServiceDeclareIn(BaseModel):
    service_id: str = Field(min_length=1)
    service_kind: str = "daemon"
    target_kind: str = "workflow"
    target_ref: str = Field(min_length=1)
    target_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    autostart: bool = False
    restart_policy: dict[str, Any] = Field(default_factory=dict)
    heartbeat_ttl_ms: int = 60_000
    trigger_specs: list[ServiceTriggerSpecIn] = Field(default_factory=list)


class ServiceHeartbeatIn(BaseModel):
    instance_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class ServiceTriggerIn(BaseModel):
    trigger_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


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
    return internal_http_error(exc)


def _sse_frame(*, event_type: str, seq: int, payload: dict[str, Any]) -> str:
    return f"id: {seq}\nevent: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _runtime_sse_debug_log(
    *,
    stage: str,
    run_id: str,
    after_seq: int | None = None,
    last_seq: int | None = None,
    event_type: str | None = None,
    event_count: int | None = None,
    terminal: bool | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    log_path = str(os.getenv("KOGWISTAR_RUNTIME_SSE_DEBUG_LOG") or "").strip()
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ts_ms": int(time.time() * 1000),
        "stage": stage,
        "run_id": str(run_id),
    }
    if after_seq is not None:
        payload["after_seq"] = int(after_seq)
    if last_seq is not None:
        payload["last_seq"] = int(last_seq)
    if event_type is not None:
        payload["event_type"] = str(event_type)
    if event_count is not None:
        payload["event_count"] = int(event_count)
    if terminal is not None:
        payload["terminal"] = bool(terminal)
    if detail:
        payload["detail"] = detail
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_runtime_router(
    *,
    get_service: Callable[[], Any],
    require_role: Callable[[str], None],
    require_namespace: Callable[[Any], None],
    runtime_namespaces: Any,
    get_subject: Callable[[], str | None] | None = None,
    get_user_id: Callable[[], str | None] | None = None,
    require_workflow_access: Callable[[str, str], None] | None = None,
):
    router = APIRouter(prefix="/api/workflow", tags=["runtime"])
    get_service_r = cast(Callable[[], ChatRunService], get_service)

    @router.post("/runs")
    def submit_workflow_run(inp: SubmitWorkflowRunIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        if require_workflow_access:
            require_workflow_access(inp.workflow_id, "rw")
        try:
            effective_user_id = (
                get_user_id() if callable(get_user_id) else None
            ) or inp.user_id
            payload = get_service_r().submit_workflow_run(
                workflow_id=inp.workflow_id,
                conversation_id=inp.conversation_id,
                turn_node_id=inp.turn_node_id,
                user_id=effective_user_id,
                initial_state=inp.initial_state,
                priority_class=inp.priority_class,
                token_budget=inp.token_budget,
                time_budget_ms=inp.time_budget_ms,
            )
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}")
    def get_workflow_run(run_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            run = get_service_r().get_run(run_id)
            if require_workflow_access:
                require_workflow_access(run["workflow_id"], "ro")
            return run
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/runs/{run_id}/events")
    async def get_workflow_run_events(run_id: str, after_seq: int = 0):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            await get_service_r().aget_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)
        _runtime_sse_debug_log(stage="open", run_id=run_id, after_seq=after_seq)

        async def event_stream():
            last_seq = int(after_seq or 0)
            while True:
                service = get_service_r()
                events = await service.alist_run_events(run_id, after_seq=last_seq)
                _runtime_sse_debug_log(
                    stage="batch",
                    run_id=run_id,
                    after_seq=after_seq,
                    last_seq=last_seq,
                    event_count=len(events),
                )
                for evt in events:
                    last_seq = int(evt["seq"])
                    payload = {
                        "run_id": run_id,
                        "event_type": evt["event_type"],
                        "created_at_ms": evt["created_at_ms"],
                        **(evt["payload"] or {}),
                    }
                    _runtime_sse_debug_log(
                        stage="yield",
                        run_id=run_id,
                        after_seq=after_seq,
                        last_seq=last_seq,
                        event_type=str(evt["event_type"]),
                    )
                    yield _sse_frame(
                        event_type=evt["event_type"], seq=last_seq, payload=payload
                    )
                run = await service.aget_run(run_id)
                _runtime_sse_debug_log(
                    stage="post_batch",
                    run_id=run_id,
                    after_seq=after_seq,
                    last_seq=last_seq,
                    event_count=len(events),
                    terminal=bool(run["terminal"]),
                )
                if run["terminal"] and not events:
                    _runtime_sse_debug_log(
                        stage="close",
                        run_id=run_id,
                        after_seq=after_seq,
                        last_seq=last_seq,
                        terminal=True,
                    )
                    break
                await asyncio.sleep(0.1)

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=headers
        )

    @router.post("/runs/{run_id}/cancel")
    def cancel_workflow_run(run_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service_r().cancel_run(run_id)
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/resources")
    def get_resource_snapshot():
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().resource_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/visibility")
    def get_visibility_snapshot():
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().visibility_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/scheduler/timeline")
    def get_scheduler_timeline(run_id: str | None = None, limit: int = 200):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return {
                "run_id": run_id,
                "events": get_service_r().list_scheduler_timeline(
                    run_id=run_id, limit=limit
                ),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/budget")
    def get_budget_snapshot():
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().budget_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/budget/history")
    def get_budget_history(limit: int = 200):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().budget_history(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/lane/progress")
    def get_lane_message_progress(
        run_id: str | None = None, conversation_id: str | None = None, limit: int = 200
    ):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().lane_message_progress(
                run_id=run_id, conversation_id=conversation_id, limit=limit
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/tools/audit")
    def get_tool_audit(conversation_id: str | None = None, limit: int = 200):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().list_tool_audit(
                conversation_id=conversation_id, limit=limit
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services")
    def declare_service(inp: ServiceDeclareIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().declare_service(
                service_id=inp.service_id,
                service_kind=inp.service_kind,
                target_kind=inp.target_kind,
                target_ref=inp.target_ref,
                target_config=inp.target_config,
                enabled=inp.enabled,
                autostart=inp.autostart,
                restart_policy=inp.restart_policy,
                heartbeat_ttl_ms=inp.heartbeat_ttl_ms,
                trigger_specs=[
                    spec.model_dump(mode="python") for spec in inp.trigger_specs
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/services")
    def list_services(limit: int = 200):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return {"services": get_service_r().list_services(limit=limit)}
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/services/{service_id}")
    def get_service(service_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().get_service(service_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/enable")
    def enable_service(service_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().enable_service(service_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/disable")
    def disable_service(service_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().disable_service(service_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/heartbeat")
    def record_service_heartbeat(service_id: str, inp: ServiceHeartbeatIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().record_service_heartbeat(
                service_id,
                instance_id=inp.instance_id,
                payload=inp.payload,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/trigger")
    def trigger_service(service_id: str, inp: ServiceTriggerIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().trigger_service(
                service_id,
                trigger_type=inp.trigger_type,
                payload=inp.payload,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/services/{service_id}/events")
    def get_service_events(service_id: str, limit: int = 500):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return {
                "service_id": service_id,
                "events": get_service_r().list_service_events(
                    service_id, limit=limit
                ),
            }
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/repair")
    def repair_service_projection(service_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_service_projection(service_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/repair")
    def repair_service_projections(limit: int = 10_000):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_service_projections(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/messages/repair-orphans")
    def repair_orphaned_claimed_messages(inbox_id: str | None = None, limit: int = 100):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_orphaned_claimed_messages(
                inbox_id=inbox_id, limit=limit
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/dead-letters")
    def dead_letter_snapshot(limit: int = 100):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().dead_letter_snapshot(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/dead-letters/{run_id}/replay")
    def replay_dead_letter(run_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().replay_dead_letter(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/{service_id}/repair")
    def repair_service_projection(service_id: str):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_service_projection(service_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/services/repair")
    def repair_service_projections(limit: int = 10_000):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_service_projections(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/messages/repair-orphans")
    def repair_orphaned_claimed_messages(inbox_id: str | None = None, limit: int = 100):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().repair_orphaned_claimed_messages(
                inbox_id=inbox_id, limit=limit
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/capabilities")
    def get_capabilities_snapshot():
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().capability_snapshot()
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.get("/operator/dashboard")
    def get_operator_dashboard(limit: int = 100):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().operator_dashboard(limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/capabilities/approve")
    def approve_capability(inp: CapabilityApproveIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service_r().capability_approve(
                action=inp.action,
                capabilities=inp.capabilities,
                subject=inp.subject,
            )
            return JSONResponse(status_code=202, content=payload)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/capabilities/revoke")
    def revoke_capability(inp: CapabilityRevokeIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        try:
            payload = get_service_r().capability_revoke(
                capability=inp.capability,
                subject=inp.subject,
            )
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

    @router.get("/runs/{run_id}/resume-contract")
    def get_workflow_resume_contract(run_id: str):
        require_role("ro")
        require_namespace(runtime_namespaces)
        try:
            return get_service_r().resume_contract(run_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/runs/{run_id}/resume")
    def resume_workflow_run(run_id: str, inp: ResumeRunIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        if require_workflow_access:
            require_workflow_access(inp.workflow_id, "rw")
        try:
            return get_service_r().resume_run(
                run_id=run_id,
                suspended_node_id=inp.suspended_node_id,
                suspended_token_id=inp.suspended_token_id,
                client_result=inp.client_result,
                workflow_id=inp.workflow_id,
                conversation_id=inp.conversation_id,
                turn_node_id=inp.turn_node_id,
                user_id=inp.user_id,
            )
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/nodes")
    def upsert_workflow_node(workflow_id: str, inp: UpsertWorkflowNodeIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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
        if require_workflow_access:
            require_workflow_access(workflow_id, "ro")
        try:
            return get_service_r().workflow_design_history(workflow_id=workflow_id)
        except Exception as exc:  # noqa: BLE001
            raise _as_http_error(exc)

    @router.post("/design/{workflow_id}/undo")
    def workflow_design_undo(workflow_id: str, inp: DesignActorIn):
        require_role("rw")
        require_namespace(runtime_namespaces)
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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
        if require_workflow_access:
            require_workflow_access(workflow_id, "rw")
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

    @router.get("/design/{workflow_id}/graph")
    def workflow_design_graph(workflow_id: str, refresh: bool = False):
        service = get_service_r()
        return service.workflow_design_graph(workflow_id=workflow_id, refresh=refresh)

    @router.get("/catalog/ops")
    def workflow_catalog_ops():
        service = get_service_r()
        return service.workflow_catalog_ops()

    return router
