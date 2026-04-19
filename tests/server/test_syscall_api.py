from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import kogwistar.server_mcp_with_admin as server
from kogwistar.server.auth_middleware import claims_ctx
from tests.server.test_chat_server_api import _configure_server, _runtime_success_runner, _token_header, engine_triplet


pytestmark = pytest.mark.server


def _build_minimal_workflow(service, *, workflow_id: str, designer_id: str) -> None:
    service.workflow_design_upsert_node(
        workflow_id=workflow_id,
        designer_id=designer_id,
        node_id="start",
        label="Start",
        op="start",
        start=True,
    )
    service.workflow_design_upsert_node(
        workflow_id=workflow_id,
        designer_id=designer_id,
        node_id="end",
        label="End",
        op="end",
        terminal=True,
    )
    service.workflow_design_upsert_edge(
        workflow_id=workflow_id,
        designer_id=designer_id,
        edge_id="edge-start-end",
        src="start",
        dst="end",
        relation="wf_next",
        is_default=True,
    )


def test_syscall_surface_lists_versioned_ops(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    with TestClient(server.app) as client:
        resp = client.get("/api/syscall/v1")
        resp.raise_for_status()
        payload = resp.json()
        assert payload["version"] == "v1"
        assert "spawn_process" in payload["ops"]
        assert "resume" in payload["ops"]


def test_syscall_spawn_process_returns_ok_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": [
                "spawn_process",
                "workflow.run.write",
                "workflow.run.read",
                "project_view",
                "workflow.design.write",
                "write_graph",
            ],
            "sub": "tester",
        }
    )
    try:
        _build_minimal_workflow(service, workflow_id="wf.syscall.demo", designer_id="tester")
        created = service.create_conversation(user_id="user-syscall")
    finally:
        claims_ctx.reset(token)
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            capabilities="spawn_process,workflow.run.write,workflow.run.read,project_view,workflow.design.write,write_graph",
        )
        resp = client.post(
            "/api/syscall/v1/spawn_process",
            json={
                "version": "v1",
                "op": "spawn_process",
                "args": {
                    "workflow_id": "wf.syscall.demo",
                    "conversation_id": created["conversation_id"],
                    "turn_node_id": created["start_node_id"],
                    "user_id": "user-syscall",
                    "initial_state": {"user_id": "user-syscall", "conversation_id": created["conversation_id"]},
                },
            },
            headers=headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["version"] == "v1"
        assert payload["op"] == "spawn_process"
        assert payload["status"] == "ok"
        assert payload["result"]["workflow_id"] == "wf.syscall.demo"


def test_syscall_request_approval_is_blocked_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="rw", ns="workflow", capabilities="approve_action,project_view,read_security_scope")
        resp = client.post(
            "/api/syscall/v1/request_approval",
            json={"version": "v1", "op": "request_approval", "args": {"action": "deny", "reason": "need approval"}},
            headers=headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["status"] == "blocked"
        assert payload["error"]["reason"]


def test_syscall_invoke_tool_list_transcript_returns_messages(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    token = claims_ctx.set(
        {
            "ns": "conversation",
            "role": "rw",
            "capabilities": [
                "spawn_process",
                "send_message",
                "invoke_tool",
                "read_graph",
                "project_view",
                "read_security_scope",
                "workflow.run.write",
                "workflow.run.read",
            ],
            "sub": "tester",
        }
    )
    try:
        created = service.create_conversation(user_id="user-tool")
    finally:
        claims_ctx.reset(token)
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="conversation",
            capabilities="invoke_tool,read_graph,send_message,project_view,read_security_scope,spawn_process,workflow.run.write,workflow.run.read",
        )
        resp = client.post(
            "/api/syscall/v1/invoke_tool",
            json={
                "version": "v1",
                "op": "invoke_tool",
                "args": {
                    "tool_name": "list_transcript",
                    "conversation_id": created["conversation_id"],
                },
            },
            headers=headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["result"]["tool_name"] == "list_transcript"
        assert payload["result"]["messages"] == []


def test_syscall_request_approval_branches(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            capabilities="approve_action,project_view,read_security_scope",
        )
        granted = client.post(
            "/api/syscall/v1/request_approval",
            json={
                "version": "v1",
                "op": "request_approval",
                "args": {
                    "action": "grant",
                    "approval_action": "request_approval",
                    "subject": "tester",
                    "capability": "workflow.run.read",
                },
            },
            headers=headers,
        )
        granted.raise_for_status()
        granted_payload = granted.json()
        assert granted_payload["status"] == "ok"
        assert granted_payload["result"]["status"] == "approved"

        denied = client.post(
            "/api/syscall/v1/request_approval",
            json={
                "version": "v1",
                "op": "request_approval",
                "args": {"action": "deny", "reason": "manual deny"},
            },
            headers=headers,
        )
        denied.raise_for_status()
        denied_payload = denied.json()
        assert denied_payload["status"] == "blocked"
        assert denied_payload["error"]["reason"] == "manual deny"

        revoked = client.post(
            "/api/syscall/v1/request_approval",
            json={
                "version": "v1",
                "op": "request_approval",
                "args": {
                    "action": "revoke",
                    "subject": "tester",
                    "capability": "workflow.run.read",
                },
            },
            headers=headers,
        )
        revoked.raise_for_status()
        revoked_payload = revoked.json()
        assert revoked_payload["status"] == "ok"
        assert revoked_payload["result"]["status"] == "revoked"


def test_syscall_audit_endpoint_records_dispatches(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            capabilities="approve_action,project_view,read_security_scope",
        )
        client.get("/api/syscall/v1", headers=headers).raise_for_status()
        audit = client.get("/api/syscall/v1/audit", headers=headers)
        audit.raise_for_status()
        payload = audit.json()
        assert payload["version"] == "v1"
        assert payload["events"]
