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
from kogwistar.engine_core.models import Grounding, Span
import kogwistar.server_mcp_with_admin as server
from kogwistar.server.auth_middleware import claims_ctx
from tests.server.auth_env import TEST_JWT_ALG, TEST_JWT_SECRET
from tests.server.chat_server_api_support import (
    _configure_server,
    _runtime_success_runner,
    _seed_checkpoint,
    _token_header,
)
from kogwistar.server.syscall_api import _invoke_builtin_tool, create_syscall_router
from tests.server.chat_server_api_support import (
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
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "syscall-catalog-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        resp = client.get(
            "/api/syscall/v1",
            headers=_token_header(client, role="ro", ns="workflow"),
        )
        assert resp.status_code == contract["success_status"], resp.text
        assert resp.json() == contract["response_body"]


def test_syscall_checkpoint_accepts_conversation_namespace_and_numeric_string_step_seq():
    calls = []
    request_namespaces = {"conversation"}
    workflow_namespaces = {"conversation", "workflow"}
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "syscall-checkpoint-step-seq-v1.json"
        ).read_text(encoding="utf-8")
    )

    def get_checkpoint(run_id, step_seq):
        calls.append((run_id, step_seq))
        return {"run_id": run_id, "step_seq": step_seq, "state": {"cursor": "x"}}

    def require_namespace(namespace):
        required = set(namespace) if isinstance(namespace, (set, frozenset, list, tuple)) else {namespace}
        if not request_namespaces.intersection(required):
            raise HTTPException(status_code=403)

    app = FastAPI()
    app.include_router(
        create_syscall_router(
            get_service=lambda: SimpleNamespace(get_checkpoint=get_checkpoint),
            require_role=lambda _role: None,
            require_namespace=require_namespace,
            conversation_namespace=object(),
            workflow_namespaces=workflow_namespaces,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/syscall/v1/checkpoint",
            json={
                "version": "v1",
                "op": "checkpoint",
            "args": {"run_id": "run-1", "step_seq": contract["step_seq_input"]},
            },
        )

    assert response.status_code == 200
    assert response.json()["result"] == {
        "run_id": "run-1",
        "step_seq": contract["step_seq_expected"],
        "state": {"cursor": "x"},
    }
    assert calls == [("run-1", contract["step_seq_expected"])]


def test_syscall_checkpoint_reads_persisted_python_checkpoint(monkeypatch, engine_triplet):
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/syscall-checkpoint-persisted-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    _seed_checkpoint(
        conversation_engine,
        run_id=contract["run_id"],
        workflow_id=contract["workflow_id"],
        step_seq=contract["step_seq_expected"],
        state=contract["state"],
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/syscall/v1/checkpoint",
            json={
                "version": contract["version"],
                "op": contract["op"],
                "args": {
                    "run_id": contract["run_id"],
                    "step_seq": contract["step_seq_input"],
                },
            },
            headers=_token_header(client, role="ro", ns="workflow"),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["result"] == contract["expected_result"]


def test_syscall_resume_returns_completed_result_contract(monkeypatch, engine_triplet):
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-resume-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    monkeypatch.setattr(
        service._run_execution,
        "_default_resume_runner",
        lambda request: {
            "workflow_status": "succeeded",
            "final_state": {"after_resume": request.client_result["approved"]},
            "budget": {},
        },
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/syscall/v1/resume",
            json={
                "version": "v1",
                "op": "resume",
                "args": {
                    "run_id": "syscall-resume-run",
                    "suspended_node_id": "approval-step",
                    "suspended_token_id": "approval-token",
                    "client_result": {"approved": True},
                    "workflow_id": "syscall-resume-workflow",
                    "conversation_id": "syscall-resume-conversation",
                    "turn_node_id": "syscall-resume-turn",
                },
            },
            headers=_token_header(client, role="rw", ns="workflow"),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["op"] == "resume"
    assert payload["status"] == "ok"
    assert set(contract["required_fields"]).issubset(payload["result"])
    assert payload["result"]["workflow_status"] in contract["completed_statuses"]
    assert payload["result"] == {
        "workflow_status": "succeeded",
        "final_state": {"after_resume": True},
        "budget": {},
    }


def test_syscall_project_view_matches_resource_snapshot_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-resources-http-v1.json"
        ).read_text(encoding="utf-8")
    )

    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities=",".join(contract["required_capabilities"]),
        )
        wrong_scope = _token_header(
            client,
            role="ro",
            ns="wisdom",
            capabilities=",".join(contract["required_capabilities"]),
        )
        denied = client.post(
            "/api/syscall/v1/project_view",
            json={"version": "v1", "op": "project_view", "args": {}},
            headers=wrong_scope,
        )
        conversation_scope = _token_header(
            client,
            role="ro",
            ns="conversation",
            capabilities=",".join(contract["required_capabilities"]),
        )
        conversation_result = client.post(
            "/api/syscall/v1/project_view",
            json={"version": "v1", "op": "project_view", "args": {}},
            headers=conversation_scope,
        )
        response = client.post(
            "/api/syscall/v1/project_view",
            json={"version": "v1", "op": "project_view", "args": {}},
            headers=headers,
        )

    assert denied.status_code == 403
    assert conversation_result.status_code == 200, conversation_result.text
    assert response.status_code == 200, response.text
    payload = response.json()["result"]
    assert all(field in payload for field in contract["required_fields"])
    for key, required_fields in contract["required_object_fields"].items():
        assert all(field in payload[key] for field in required_fields), key


def test_syscall_send_message_invalid_inputs_match_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    identity = claims_ctx.set(
        {
            "ns": "conversation",
            "role": "rw",
            "capabilities": ["create_conversation", "spawn_process"],
            "sub": "syscall-send-owner",
        }
    )
    try:
        service.create_conversation(
            user_id="syscall-send-owner", conversation_id="syscall-send-invalid"
        )
    finally:
        claims_ctx.reset(identity)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/syscall-send-message-invalid-v1.json"
        ).read_text(encoding="utf-8")
    )
    before = registry.meta_store.list_server_runs(
        conversation_id="syscall-send-invalid", limit=100
    )

    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="conversation",
            capabilities="send_message,spawn_process,read_graph",
        )
        for case in contract["cases"]:
            response = client.post(
                "/api/syscall/v1/send_message",
                json=case["request"],
                headers=headers,
            )
            assert response.status_code == case["http_status"], case["name"]
            assert response.json() == case["response"], case["name"]

    assert registry.meta_store.list_server_runs(
        conversation_id="syscall-send-invalid", limit=100
    ) == before


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
    expected_result = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-create-syscall-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    expected_result.update(
        conversation_id="syscall-owner-check",
        user_id="owner-create",
        start_node_id="syscall-owner-check-start",
        tail_node_id="syscall-owner-check-start",
    )
    assert payload["result"] == expected_result
    starts = conversation_engine.read.get_nodes(
        where={"conversation_id": "syscall-owner-check"},
        node_type=ConversationNode,
        limit=10,
    )
    assert len(starts) == 1
    assert starts[0].id == "syscall-owner-check-start"
    assert starts[0].user_id == "owner-create"


def test_syscall_create_conversation_router_rejects_owner_spoof_without_backend():
    calls = []

    def create_conversation(**kwargs):
        calls.append(kwargs)
        return {
            "conversation_id": kwargs["conversation_id"],
            "user_id": kwargs["user_id"],
            "status": "active",
            "start_node_id": kwargs["start_node_id"],
            "tail_node_id": kwargs["start_node_id"],
            "turn_count": 0,
        }

    service = SimpleNamespace(create_conversation=create_conversation)
    conversation_namespace = object()

    def require_role(role):
        if role != "rw":
            raise HTTPException(status_code=403)

    def require_namespace(namespace):
        if namespace is not conversation_namespace:
            raise HTTPException(status_code=403)

    app = FastAPI()
    app.include_router(
        create_syscall_router(
            get_service=lambda: service,
            require_role=require_role,
            require_namespace=require_namespace,
            conversation_namespace=conversation_namespace,
            workflow_namespaces=object(),
            get_user_id=lambda: "authenticated-owner",
        )
    )

    with TestClient(app) as client:
        denied = client.post(
            "/api/syscall/v1/create_conversation",
            json={
                "version": "v1",
                "op": "create_conversation",
                "args": {"user_id": "spoofed-owner"},
            },
        )
        assert denied.status_code == 403
        assert calls == []

        created = client.post(
            "/api/syscall/v1/create_conversation",
            json={
                "version": "v1",
                "op": "create_conversation",
                "args": {
                    "user_id": "authenticated-owner",
                    "conversation_id": "router-owner-check",
                    "start_node_id": "router-owner-start",
                },
            },
        )

    assert created.status_code == 200
    assert created.json()["result"]["user_id"] == "authenticated-owner"
    assert calls == [
        {
            "user_id": "authenticated-owner",
            "conversation_id": "router-owner-check",
            "start_node_id": "router-owner-start",
        }
    ]


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

        cancel_contract = json.loads(
            (Path(__file__).resolve().parents[2] / "contracts/golden/run-cancel-v1.json").read_text(
                encoding="utf-8"
            )
        )
        terminated = client.post(
            "/api/syscall/v1/terminate_process",
            json={
                "version": "v1",
                "op": "terminate_process",
                "args": {"run_id": payload["result"]["run_id"]},
            },
            headers=headers,
        )
        assert terminated.status_code == 200, terminated.text
        terminated_payload = terminated.json()
        assert terminated_payload["version"] == "v1"
        assert terminated_payload["op"] == "terminate_process"
        assert terminated_payload["status"] == "ok"
        terminated_run = terminated_payload["result"]
        for field in cancel_contract["required_fields"]:
            assert field in terminated_run, field
        assert terminated_run["status"] in cancel_contract["accepted_run_statuses"]


def test_syscall_request_approval_is_blocked_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/golden/syscall-request-approval-result-v1.json").read_text(
            encoding="utf-8"
        )
    )
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="rw", ns="workflow", capabilities="approve_action,project_view,read_security_scope")
        resp = client.post(
            "/api/syscall/v1/request_approval",
            json=contract["request"],
            headers=headers,
        )
        assert resp.status_code == contract["http_status"]
        assert resp.json() == contract["response"]

    requested_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/syscall-request-approval-unrecognized-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/api/syscall/v1/request_approval",
            json=requested_contract["request"],
            headers=_token_header(
                client,
                role="rw",
                ns="workflow",
                capabilities="",
            ),
        )
        assert response.status_code == requested_contract["http_status"]
        assert response.json() == requested_contract["response"]


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

        error_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/syscall-invoke-tool-errors-v1.json"
            ).read_text(encoding="utf-8")
        )
        graph_ids_before_errors = {
            node.id
            for node in conversation_engine.read.get_nodes(
                where={"conversation_id": created["conversation_id"]},
                node_type=ConversationNode,
                limit=1_000,
            )
        }
        for case in error_contract["cases"]:
            error_response = client.post(
                "/api/syscall/v1/invoke_tool",
                json=case["request"],
                headers=headers,
            )
            assert error_response.status_code == case["http_status"], case["name"]
            assert error_response.json() == case["response"], case["name"]
        graph_ids_after_errors = {
            node.id
            for node in conversation_engine.read.get_nodes(
                where={"conversation_id": created["conversation_id"]},
                node_type=ConversationNode,
                limit=1_000,
            )
        }
        assert graph_ids_after_errors == graph_ids_before_errors


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


def test_syscall_receive_message_reads_chroma_graph_and_matches_shared_fixture(
    monkeypatch, engine_triplet
):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/receive-message-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    _engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        _engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )

    for message in fixture["messages"]:
        content = message["content"]
        embedding = conversation_engine.iterative_defensive_emb(content)
        conversation_engine.write.add_node(
            ConversationNode(
                id=message["node_id"],
                label=f"Turn {message['turn_index']}",
                type="entity",
                summary=content,
                mentions=[
                    Grounding(spans=[Span.from_dummy_for_conversation("c1")])
                ],
                doc_id="conv:c1",
                metadata={
                    "entity_type": message["entity_type"],
                    "in_conversation_chain": True,
                    "level_from_root": 0,
                },
                role=message["role"],
                turn_index=message["turn_index"],
                conversation_id=fixture["conversation_id"],
                user_id="reader-1",
                embedding=embedding.tolist()
                if hasattr(embedding, "tolist")
                else embedding,
            )
        )
    empty_embedding = conversation_engine.iterative_defensive_emb("snapshot")
    conversation_engine.write.add_node(
        ConversationNode(
            id="receive-empty-snapshot",
            label="Empty conversation snapshot",
            type="entity",
            summary="snapshot",
            mentions=[
                Grounding(spans=[Span.from_dummy_for_conversation("c-empty")])
            ],
            doc_id="conv:c-empty",
            metadata={
                "entity_type": "context_snapshot",
                "in_conversation_chain": True,
                "level_from_root": 0,
            },
            conversation_id="c-empty",
            user_id="reader-1",
            embedding=empty_embedding.tolist()
            if hasattr(empty_embedding, "tolist")
            else empty_embedding,
        )
    )

    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="conversation")
        response = client.post(
            "/api/syscall/v1/receive_message",
            json={
                "version": "v1",
                "op": "receive_message",
                "args": {"conversation_id": fixture["conversation_id"]},
            },
            headers=headers,
        )
        unknown = client.post(
            "/api/syscall/v1/receive_message",
            json={
                "version": "v1",
                "op": "receive_message",
                "args": {"conversation_id": "c-unknown"},
            },
            headers=headers,
        )
        empty = client.post(
            "/api/syscall/v1/receive_message",
            json={
                "version": "v1",
                "op": "receive_message",
                "args": {"conversation_id": "c-empty"},
            },
            headers=headers,
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["op"] == "receive_message"
    assert payload["status"] == "ok"
    assert payload["result"] == fixture
    assert unknown.status_code == 404, unknown.text
    assert empty.status_code == 200, empty.text
    assert empty.json()["result"]["messages"] == []


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


def test_syscall_receive_message_unknown_conversation_is_not_found():
    def list_transcript(conversation_id: str) -> list[dict[str, object]]:
        raise KeyError(f"Unknown conversation_id: {conversation_id}")

    router = create_syscall_router(
        get_service=lambda: SimpleNamespace(list_transcript=list_transcript),
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
            json={
                "version": "v1",
                "op": "receive_message",
                "args": {"conversation_id": "missing"},
            },
        )

    assert response.status_code == 404


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
        assert any(
            approval["subject"] == "tester"
            and approval["action"] == "request_approval"
            and "workflow.run.read" in approval["capabilities"]
            for approval in granted_payload["result"]["approval"]["approvals"]
        )

        for action, missing_capability in (
            ("grant", "project_view"),
            ("grant", "read_security_scope"),
            ("revoke", "project_view"),
            ("revoke", "read_security_scope"),
        ):
            capabilities = {"approve_action", "project_view", "read_security_scope"}
            capabilities.remove(missing_capability)
            denied_mutation = client.post(
                "/api/syscall/v1/request_approval",
                json={
                    "version": "v1",
                    "op": "request_approval",
                    "args": {
                        "action": action,
                        "subject": "unauthorized-target",
                        "capability": "workflow.run.write",
                    },
                },
                headers=_token_header(
                    client,
                    role="rw",
                    ns="workflow",
                    capabilities=",".join(sorted(capabilities)),
                ),
            )
            assert denied_mutation.status_code == 403

        capability_snapshot = client.get(
            "/api/workflow/capabilities", headers=headers
        )
        capability_snapshot.raise_for_status()
        assert not any(
            approval["subject"] == "unauthorized-target"
            for approval in capability_snapshot.json()["approvals"]
        )
        assert not any(
            revoked["subject"] == "unauthorized-target"
            for revoked in capability_snapshot.json()["revoked"]
        )

        denied = client.post(
            "/api/syscall/v1/request_approval",
            json={
                "version": "v1",
                "op": "request_approval",
                "args": {"action": "deny", "reason": "manual deny"},
            },
            headers=_token_header(client, role="rw", ns="workflow", capabilities=""),
        )
        denied.raise_for_status()
        denied_payload = denied.json()
        assert denied_payload["status"] == "blocked"
        assert denied_payload["error"]["reason"] == "manual deny"

        requested_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/syscall-request-approval-unrecognized-v1.json"
            ).read_text(encoding="utf-8")
        )
        requested = client.post(
            "/api/syscall/v1/request_approval",
            json=requested_contract["request"],
            headers=_token_header(client, role="rw", ns="workflow", capabilities=""),
        )
        assert requested.status_code == requested_contract["http_status"], requested.text
        assert requested.json() == requested_contract["response"]

        invalid_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts"
                / "golden"
                / "syscall-request-approval-invalid-v1.json"
            ).read_text(encoding="utf-8")
        )
        for case in invalid_contract["cases"]:
            invalid_response = client.post(
                "/api/syscall/v1/request_approval",
                json=case["request"],
                headers=_token_header(client, role="rw", ns="workflow", capabilities=""),
            )
            assert invalid_response.status_code == case["http_status"], case["name"]
            assert invalid_response.json() == case["response"], case["name"]

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
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "syscall-audit-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            capabilities="approve_action,project_view,read_security_scope",
        )
        dispatched = client.post(
            "/api/syscall/v1/request_approval",
            json={
                "version": "v1",
                "op": "request_approval",
                "args": {"action": "deny", "reason": "manual deny"},
            },
            headers=headers,
        )
        assert dispatched.status_code == 200
        assert dispatched.json()["status"] == "blocked"

        audit = client.get(
            f"/api/syscall/v1/audit?limit={contract['limit']}", headers=headers
        )
        assert audit.status_code == contract["success_status"], audit.text
        payload = audit.json()
        for field in contract["required_fields"]:
            assert field in payload
        assert payload["version"] == "v1"
        assert len(payload["events"]) == 1
        event = payload["events"][0]
        for field in contract["required_event_fields"]:
            assert field in event
        for field, expected in contract["expected_event"].items():
            assert event[field] == expected
