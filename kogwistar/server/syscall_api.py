from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .chat_service import ChatRunService
from .error_reporting import internal_http_error


class SyscallRequest(BaseModel):
    version: str = "v1"
    op: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class SyscallResponse(BaseModel):
    version: str
    op: str
    status: str
    result: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, HTTPException):
        return exc
    return internal_http_error(exc)


def _ok(version: str, op: str, result: dict[str, Any]) -> SyscallResponse:
    return SyscallResponse(version=version, op=op, status="ok", result=result)


def _blocked(version: str, op: str, reason: str) -> SyscallResponse:
    return SyscallResponse(version=version, op=op, status="blocked", error={"reason": reason})


def _invoke_builtin_tool(service: ChatRunService, args: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(args.get("tool_name") or "").strip().lower()
    if not tool_name:
        raise ValueError("tool_name is required")
    if tool_name == "list_transcript":
        conversation_id = str(args["conversation_id"])
        return {
            "tool_name": tool_name,
            "conversation_id": conversation_id,
            "messages": service.list_transcript(conversation_id),
        }
    if tool_name == "latest_snapshot":
        conversation_id = str(args["conversation_id"])
        return {
            "tool_name": tool_name,
            "conversation_id": conversation_id,
            "snapshot": service.latest_snapshot(
                conversation_id,
                run_id=args.get("run_id"),
                stage=args.get("stage"),
            ),
        }
    if tool_name == "capability_snapshot":
        return {"tool_name": tool_name, "snapshot": service.capability_snapshot()}
    if tool_name == "resource_snapshot":
        return {"tool_name": tool_name, "snapshot": service.resource_snapshot()}
    raise KeyError(f"unknown tool: {tool_name}")


def _request_approval(service: ChatRunService, args: dict[str, Any]) -> dict[str, Any]:
    action = str(args.get("action") or "").strip().lower()
    subject = args.get("subject")
    capability = str(args.get("capability") or "").strip().lower()
    if not action:
        raise ValueError("action is required")
    if action == "grant":
        if not capability:
            raise ValueError("capability is required for grant")
        payload = service.capability_approve(
            action=str(args.get("approval_action") or "request_approval"),
            capabilities=[capability],
            subject=subject,
        )
        return {
            "status": "approved",
            "approval": payload,
        }
    if action == "revoke":
        if not capability:
            raise ValueError("capability is required for revoke")
        payload = service.capability_revoke(capability=capability, subject=subject)
        return {
            "status": "revoked",
            "revocation": payload,
        }
    if action == "deny":
        return {
            "status": "blocked",
            "reason": str(args.get("reason") or "request denied"),
        }
    return {
        "status": "requested",
        "reason": str(args.get("reason") or "approval pending"),
    }


def create_syscall_router(
    *,
    get_service: Callable[[], Any],
    require_role: Callable[[str], None],
    require_namespace: Callable[[Any], None],
    conversation_namespace: Any,
    workflow_namespaces: Any,
    get_user_id: Callable[[], str | None] | None = None,
):
    router = APIRouter(prefix="/api/syscall", tags=["syscall"])
    get_service_r = lambda: get_service()  # noqa: E731
    audit_log: list[dict[str, Any]] = []

    @router.get("/v1")
    def list_syscalls() -> dict[str, Any]:
        return {
            "version": "v1",
            "ops": [
                "spawn_process",
                "terminate_process",
                "send_message",
                "receive_message",
                "mount_memory",
                "project_view",
                "invoke_tool",
                "checkpoint",
                "resume",
                "request_approval",
            ],
        }

    @router.get("/v1/audit")
    def list_syscall_audit(limit: int = 200) -> dict[str, Any]:
        require_role("ro")
        require_namespace(workflow_namespaces)
        return {"version": "v1", "events": audit_log[-max(0, int(limit)) :]}

    @router.post("/v1/{op}")
    def dispatch(op: str, inp: SyscallRequest):
        service: ChatRunService = get_service_r()
        version = str(inp.version or "v1")
        op = str(op or inp.op or "").strip().lower()
        started_at = int(time.time() * 1000)
        try:
            if op == "spawn_process":
                require_role("rw")
                require_namespace(workflow_namespaces)
                args = inp.args
                payload = service.submit_workflow_run(
                    workflow_id=str(args["workflow_id"]),
                    conversation_id=str(args["conversation_id"]),
                    turn_node_id=args.get("turn_node_id"),
                    user_id=args.get("user_id") or (get_user_id() if callable(get_user_id) else None),
                    initial_state=dict(args.get("initial_state") or {}),
                    priority_class=str(args.get("priority_class") or "foreground"),
                    token_budget=args.get("token_budget"),
                    time_budget_ms=args.get("time_budget_ms"),
                )
                resp = _ok(version, op, payload)
                audit_log.append(
                    {
                        "ts_ms": started_at,
                        "version": version,
                        "op": op,
                        "status": "ok",
                    }
                )
                return resp
            if op == "terminate_process":
                require_role("rw")
                require_namespace(workflow_namespaces)
                payload = service.cancel_run(str(inp.args["run_id"]))
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "send_message":
                require_role("rw")
                require_namespace(conversation_namespace)
                payload = service.submit_turn_for_answer(
                    conversation_id=str(inp.args["conversation_id"]),
                    user_id=inp.args.get("user_id") or (get_user_id() if callable(get_user_id) else None),
                    text=str(inp.args["text"]),
                    workflow_id=str(inp.args.get("workflow_id") or "agentic_answering.v2"),
                )
                return _ok(version, op, payload)
            if op == "receive_message":
                require_role("ro")
                require_namespace(conversation_namespace)
                payload = service.list_transcript(str(inp.args["conversation_id"]))
                resp = _ok(version, op, {"conversation_id": inp.args["conversation_id"], "messages": payload})
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "mount_memory":
                require_role("ro")
                require_namespace(conversation_namespace)
                payload = service.latest_snapshot(
                    str(inp.args["conversation_id"]),
                    run_id=inp.args.get("run_id"),
                    stage=inp.args.get("stage"),
                )
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "project_view":
                require_role("ro")
                require_namespace(workflow_namespaces)
                payload = service.resource_snapshot()
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "invoke_tool":
                require_role("rw")
                require_namespace(conversation_namespace)
                payload = _invoke_builtin_tool(service, inp.args)
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "checkpoint":
                require_role("ro")
                require_namespace(workflow_namespaces)
                payload = service.get_checkpoint(
                    str(inp.args["run_id"]),
                    int(inp.args["step_seq"]),
                )
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            if op == "resume":
                require_role("rw")
                require_namespace(workflow_namespaces)
                payload = service.resume_run(
                    run_id=str(inp.args["run_id"]),
                    suspended_node_id=str(inp.args["suspended_node_id"]),
                    suspended_token_id=str(inp.args["suspended_token_id"]),
                    client_result=dict(inp.args.get("client_result") or {}),
                    workflow_id=str(inp.args["workflow_id"]),
                    conversation_id=str(inp.args["conversation_id"]),
                    turn_node_id=str(inp.args["turn_node_id"]),
                    user_id=inp.args.get("user_id") or (get_user_id() if callable(get_user_id) else None),
                )
                return _ok(version, op, payload)
            if op == "request_approval":
                require_role("rw")
                require_namespace(workflow_namespaces)
                payload = _request_approval(service, inp.args)
                status = str(payload.get("status") or "requested")
                if status == "blocked":
                    resp = _blocked(version, op, str(payload.get("reason") or "blocked"))
                    audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "blocked"})
                    return resp
                resp = _ok(version, op, payload)
                audit_log.append({"ts_ms": started_at, "version": version, "op": op, "status": "ok"})
                return resp
            raise KeyError(f"unknown syscall op: {op}")
        except Exception as exc:  # noqa: BLE001
            audit_log.append(
                {
                    "ts_ms": started_at,
                    "version": version,
                    "op": op,
                    "status": "error",
                    "error": str(exc),
                }
            )
            raise _as_http_error(exc)

    return router
