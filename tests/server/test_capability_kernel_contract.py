from __future__ import annotations

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.runtime.runtime import WorkflowRuntime
from tests.server.test_chat_server_api import (
    _configure_server,
    engine_triplet,
    _token_header,
    _runtime_success_runner,
)
import kogwistar.server_mcp_with_admin as server


pytestmark = pytest.mark.server


def _wait_for_terminal(service, run_id: str) -> dict:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run.get("terminal"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not become terminal")


def test_capability_snapshot_route_exposes_effective_caps(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )

    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        resp = client.get("/api/workflow/capabilities", headers=headers)
        resp.raise_for_status()
        payload = resp.json()

        assert payload["current_subject"]
        assert "workflow.run.read" in payload["effective_capabilities"]
        assert "project_view" in payload["effective_capabilities"]
        assert payload["audit_log"]
        assert payload["audit_log"][-1]["action"] == "project_view"
        assert payload["audit_log"][-1]["outcome"] == "allow"


def test_capability_approval_revocation_and_audit(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _runtime_success_runner
    )

    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
                "workflow.run.read",
                "workflow.design.write",
                "write_graph",
            ],
            "sub": "tester-cap",
        }
    )
    try:
        snapshot = service.capability_snapshot()
        assert {
            "approve_action",
            "project_view",
            "read_security_scope",
            "workflow.run.read",
        }.issubset(set(snapshot["effective_capabilities"]))

        approved = service.capability_approve(
            action="spawn_process",
            capabilities=["spawn_process", "workflow.run.write"],
        )
        assert any(
            item["subject"] == "tester-cap"
            and item["action"] == "spawn_process"
            and "spawn_process" in item["capabilities"]
            for item in approved["approvals"]
        )

        created = service.create_conversation(user_id="user-cap")
        assert created["conversation_id"]
        service.workflow_design_upsert_node(
            workflow_id="wf.capability.demo",
            designer_id="tester-cap",
            node_id="start",
            label="Start",
            op="start",
            start=True,
        )
        service.workflow_design_upsert_node(
            workflow_id="wf.capability.demo",
            designer_id="tester-cap",
            node_id="end",
            label="End",
            op="end",
            terminal=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id="wf.capability.demo",
            designer_id="tester-cap",
            edge_id="edge-start-end",
            src="start",
            dst="end",
            relation="wf_next",
            is_default=True,
        )

        run = service.submit_workflow_run(
            workflow_id="wf.capability.demo",
            conversation_id=created["conversation_id"],
            turn_node_id=created["start_node_id"],
            user_id="user-cap",
            initial_state={"user_id": "user-cap", "conversation_id": created["conversation_id"]},
        )
        final = _wait_for_terminal(service, run["run_id"])
        assert final["status"] == "succeeded"

        revoked = service.capability_revoke(capability="spawn_process")
        assert any(
            row["subject"] == "tester-cap" and "spawn_process" in row["capabilities"]
            for row in revoked["revoked"]
        )

        with pytest.raises(HTTPException):
            service.submit_workflow_run(
                workflow_id="wf.capability.demo",
                conversation_id=created["conversation_id"],
                turn_node_id=created["start_node_id"],
                user_id="user-cap",
                initial_state={"user_id": "user-cap", "conversation_id": created["conversation_id"]},
            )

        audit = service.capability_snapshot()["audit_log"]
        assert any(item["outcome"] == "allow" for item in audit)
        assert any(item["outcome"] == "deny" for item in audit)
    finally:
        claims_ctx.reset(token)


def test_runtime_run_inherits_effective_capabilities_into_deps(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    seen: dict[str, object] = {}

    def _fake_run(self, *, workflow_id, conversation_id, turn_node_id, initial_state, run_id):
        deps = dict(initial_state.get("_deps") or {})
        seen["capabilities"] = tuple(deps.get("capabilities") or ())
        seen["subject"] = deps.get("capability_subject")
        return type(
            "RunResult",
            (),
            {"status": "succeeded", "final_state": {"deps_snapshot": deps}},
        )()

    monkeypatch.setattr(WorkflowRuntime, "run", _fake_run, raising=True)

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
            "sub": "tester-runtime",
        }
    )
    try:
        service.workflow_design_upsert_node(
            workflow_id="wf.capability.inherit",
            designer_id="tester-runtime",
            node_id="start",
            label="Start",
            op="start",
            start=True,
        )
        service.workflow_design_upsert_node(
            workflow_id="wf.capability.inherit",
            designer_id="tester-runtime",
            node_id="end",
            label="End",
            op="end",
            terminal=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id="wf.capability.inherit",
            designer_id="tester-runtime",
            edge_id="edge-start-end",
            src="start",
            dst="end",
            relation="wf_next",
            is_default=True,
        )
        created = service.create_conversation(user_id="user-runtime")
        run = service.submit_workflow_run(
            workflow_id="wf.capability.inherit",
            conversation_id=created["conversation_id"],
            turn_node_id=created["start_node_id"],
            user_id="user-runtime",
            initial_state={"user_id": "user-runtime", "conversation_id": created["conversation_id"]},
        )
        final = _wait_for_terminal(service, run["run_id"])
        assert final["status"] == "succeeded"
        assert "spawn_process" in seen["capabilities"]
        assert "workflow.run.write" in seen["capabilities"]
        assert seen["subject"] == "tester-runtime"
    finally:
        claims_ctx.reset(token)
