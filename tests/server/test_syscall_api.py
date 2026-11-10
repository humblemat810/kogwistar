from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from kogwistar.conversation.models import ConversationNode
import kogwistar.server_mcp_with_admin as server
from kogwistar.server.auth_middleware import claims_ctx
from tests.server.auth_env import TEST_JWT_ALG, TEST_JWT_SECRET
from tests.server.test_chat_server_api import (
    _configure_server,
    _runtime_success_runner,
    _token_header,
)
from kogwistar.server.syscall_api import _invoke_builtin_tool, create_syscall_router
from tests.server.test_chat_server_api import (
    # Imported for pytest fixture discovery in this module.
    engine_triplet as engine_triplet,  # noqa: F401
)

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
        assert "create_conversation" in payload["ops"]
        assert "resume" in payload["ops"]


def test_syscall_create_conversation_uses_authenticated_owner(
    monkeypatch, engine_triplet
):
    _engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, _engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "owner-create",
            "user_id": "owner-create",
            "role": "rw",
            "ns": ["conversation"],
            "capabilities": ["spawn_process"],
            "iat": now,
            "exp": now + 3600,
            "iss": "local",
        },
        TEST_JWT_SECRET,
        algorithm=TEST_JWT_ALG,
    )
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(server.app) as client:
        request = {
            "version": "v1",
            "op": "create_conversation",
            "args": {
                "user_id": "spoofed-owner",
                "conversation_id": "syscall-owner-check",
                "start_node_id": "syscall-owner-check-start",
            },
        }
        denied = client.post(
            "/api/syscall/v1/create_conversation", json=request, headers=headers
        )
        assert denied.status_code == 403
        assert not conversation_engine.read.get_nodes(
            where={"conversation_id": "syscall-owner-check"},
            node_type=ConversationNode,
            limit=10,
        )

        request["args"]["user_id"] = "owner-create"
        created = client.post(
            "/api/syscall/v1/create_conversation", json=request, headers=headers
        )

    assert created.status_code == 200, created.text
    payload = created.json()
    assert payload["version"] == "v1"
    assert payload["op"] == "create_conversation"
    assert payload["status"] == "ok"
    assert payload["result"]["conversation_id"] == "syscall-owner-check"
    assert payload["result"]["start_node_id"] == "syscall-owner-check-start"
    assert payload["result"]["user_id"] == "owner-create"
    assert payload["result"]["turn_count"] == 0
    starts = conversation_engine.read.get_nodes(
        where={"conversation_id": "syscall-owner-check"},
        node_type=ConversationNode,
        limit=10,
    )
    assert len(starts) == 1
    assert starts[0].id == "syscall-owner-check-start"
    assert starts[0].user_id == "owner-create"


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


def test_syscall_receive_message_matches_shared_wire_fixture():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/receive-message-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    service = SimpleNamespace(
        list_transcript=lambda conversation_id: fixture["messages"]
        if conversation_id == fixture["conversation_id"]
        else pytest.fail("unexpected conversation ID")
    )
    authorized: list[tuple[str, object]] = []
    router = create_syscall_router(
        get_service=lambda: service,
        require_role=lambda role: authorized.append(("role", role)),
        require_namespace=lambda namespace: authorized.append(("namespace", namespace)),
        conversation_namespace="conversation",
        workflow_namespaces={"workflow"},
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            "/api/syscall/v1/receive_message",
            json={
                "version": "v1",
                "op": "receive_message",
                "args": {"conversation_id": fixture["conversation_id"], "limit": 0},
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["op"] == "receive_message"
    assert payload["status"] == "ok"
    assert payload["result"] == fixture
    assert authorized == [("role", "ro"), ("namespace", "conversation")]


def test_syscall_receive_message_missing_id_matches_shared_error_fixture():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/syscall-receive-message-missing-id-v1.json"
        ).read_text(encoding="utf-8")
    )
    service = SimpleNamespace(
        list_transcript=lambda _conversation_id: pytest.fail(
            "invalid receive_message input must not read the transcript"
        )
    )
    router = create_syscall_router(
        get_service=lambda: service,
        require_role=lambda _role: None,
        require_namespace=lambda _namespace: None,
        conversation_namespace="conversation",
        workflow_namespaces={"workflow"},
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            "/api/syscall/v1/receive_message",
            json={"version": "v1", "op": "receive_message", "args": {}},
        )

    assert response.status_code == fixture["http_status"]
    assert response.json() == fixture["body"]


@pytest.mark.parametrize(
    ("role_allowed", "namespace_allowed", "checks"),
    [
        (False, True, [("role", "ro")]),
        (True, False, [("role", "ro"), ("namespace", "conversation")]),
    ],
)
def test_syscall_receive_message_denies_before_service_read(
    role_allowed: bool,
    namespace_allowed: bool,
    checks: list[tuple[str, str]],
) -> None:
    service = SimpleNamespace(
        list_transcript=lambda _conversation_id: pytest.fail(
            "authorization denial must precede transcript access"
        )
    )
    observed: list[tuple[str, str]] = []

    def require_role(role: str) -> None:
        observed.append(("role", role))
        if not role_allowed:
            raise HTTPException(status_code=403, detail="role denied")

    def require_namespace(namespace: str) -> None:
        observed.append(("namespace", namespace))
        if not namespace_allowed:
            raise HTTPException(status_code=403, detail="namespace denied")

    router = create_syscall_router(
        get_service=lambda: service,
        require_role=require_role,
        require_namespace=require_namespace,
        conversation_namespace="conversation",
        workflow_namespaces={"workflow"},
    )
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post(
            "/api/syscall/v1/receive_message",
            json={"version": "v1", "op": "receive_message", "args": {"conversation_id": "c1"}},
        )

    assert response.status_code == 403
    assert observed == checks


def test_invoke_builtin_tool_results_match_shared_contract_fixture():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/invoke-tool-results-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["version"] == "v1"
    assert fixture["op"] == "invoke_tool"
    results = fixture["results"]
    service = SimpleNamespace(
        list_transcript=lambda conversation_id: [
            {"node_id": "turn-1", "role": "user", "content": "hello"}
        ],
        latest_snapshot=lambda conversation_id, *, run_id=None, stage=None: {
            "snapshot_node_id": "snapshot-1",
            "conversation_id": conversation_id,
            "metadata": {"source": "test"},
            "properties": {"snapshot_seq": 1},
        },
        capability_snapshot=lambda: {
            "specs": [],
            "approvals": [],
            "revoked": [],
            "audit_log": [],
            "current_subject": "alice",
            "effective_capabilities": ["project_view"],
        },
        resource_snapshot=lambda: {
            "scheduler": {},
            "runs": {},
            "services": {},
            "storage_usage_bytes": 0,
            "cost_ledger": {},
            "budget_model": {},
            "policy_infra": {},
        },
    )
    inputs = {
        "list_transcript": {"tool_name": "list_transcript", "conversation_id": "conversation-1"},
        "latest_snapshot": {
            "tool_name": "latest_snapshot",
            "conversation_id": "conversation-1",
            "run_id": "run-1",
            "stage": "answer",
        },
        "capability_snapshot": {"tool_name": "capability_snapshot"},
        "resource_snapshot": {"tool_name": "resource_snapshot"},
    }
    expected = {
        "list_transcript": {
            "tool_name": "list_transcript",
            "conversation_id": "conversation-1",
            "messages": [{"node_id": "turn-1", "role": "user", "content": "hello"}],
        },
        "latest_snapshot": {
            "tool_name": "latest_snapshot",
            "conversation_id": "conversation-1",
            "snapshot": {
                "snapshot_node_id": "snapshot-1",
                "conversation_id": "conversation-1",
                "metadata": {"source": "test"},
                "properties": {"snapshot_seq": 1},
            },
        },
        "capability_snapshot": {
            "tool_name": "capability_snapshot",
            "snapshot": service.capability_snapshot(),
        },
        "resource_snapshot": {
            "tool_name": "resource_snapshot",
            "snapshot": service.resource_snapshot(),
        },
    }
    for tool_name, args in inputs.items():
        result = _invoke_builtin_tool(service, args)
        assert result == expected[tool_name]
        assert set(result) == set(results[tool_name]["root_fields"])
        if "snapshot_fields" in results[tool_name]:
            assert set(result["snapshot"]) == set(results[tool_name]["snapshot_fields"])


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
