from __future__ import annotations

import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import kogwistar.server_mcp_with_admin as server
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.capability_kernel import CapabilityKernel
from tests.server.test_chat_server_api import (
    _configure_server,
    _runtime_success_runner,
    _token_header,
    engine_triplet,
)

pytestmark = pytest.mark.server



def _wait_for_terminal(service, run_id: str) -> dict:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        run = service.get_run(run_id)
        if run.get("terminal"):
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not become terminal")



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



def test_capability_snapshot_route_exposes_effective_caps(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
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
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
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
        _build_minimal_workflow(
            service,
            workflow_id="wf.capability.demo",
            designer_id="tester-cap",
        )

        run = service.submit_workflow_run(
            workflow_id="wf.capability.demo",
            conversation_id=created["conversation_id"],
            turn_node_id=created["start_node_id"],
            user_id="user-cap",
            initial_state={
                "user_id": "user-cap",
                "conversation_id": created["conversation_id"],
            },
        )
        final = _wait_for_terminal(service, run["run_id"])
        assert final["status"] == "succeeded"

        revoked = service.capability_revoke(capability="spawn_process")
        assert any(
            row["subject"] == "tester-cap"
            and "spawn_process" in row["capabilities"]
            for row in revoked["revoked"]
        )

        with pytest.raises(HTTPException):
            service.submit_workflow_run(
                workflow_id="wf.capability.demo",
                conversation_id=created["conversation_id"],
                turn_node_id=created["start_node_id"],
                user_id="user-cap",
                initial_state={
                    "user_id": "user-cap",
                    "conversation_id": created["conversation_id"],
                },
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
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
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
        _build_minimal_workflow(
            service,
            workflow_id="wf.capability.inherit",
            designer_id="tester-runtime",
        )
        created = service.create_conversation(user_id="user-runtime")
        run = service.submit_workflow_run(
            workflow_id="wf.capability.inherit",
            conversation_id=created["conversation_id"],
            turn_node_id=created["start_node_id"],
            user_id="user-runtime",
            initial_state={
                "user_id": "user-runtime",
                "conversation_id": created["conversation_id"],
            },
        )
        final = _wait_for_terminal(service, run["run_id"])
        assert final["status"] == "succeeded"
        assert "spawn_process" in seen["capabilities"]
        assert "workflow.run.write" in seen["capabilities"]
        assert seen["subject"] == "tester-runtime"
    finally:
        claims_ctx.reset(token)



def test_capability_kernel_normalizes_and_deduplicates_capabilities():
    kernel = CapabilityKernel()

    kernel.grant(
        subject="Alice",
        action="Spawn_Process",
        capabilities=[" Spawn_Process ", "spawn_process", "*", "WORKFLOW.RUN.WRITE"],
    )

    snapshot = kernel.snapshot()
    assert snapshot["approvals"] == [
        {
            "subject": "alice",
            "action": "spawn_process",
            "capabilities": ["spawn_process", "workflow.run.write"],
        }
    ]



def test_capability_kernel_approvals_are_subject_and_action_scoped():
    kernel = CapabilityKernel()
    kernel.grant(
        subject="alice",
        action="spawn_process",
        capabilities=["spawn_process", "workflow.run.write"],
    )

    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process", "workflow.run.write"],
    )
    assert allowed is True
    assert decision.outcome == "allow"

    other_action_allowed, other_action_decision = kernel.allowed(
        subject="alice",
        action="invoke_tool",
        required=["spawn_process"],
    )
    assert other_action_allowed is False
    assert other_action_decision.outcome == "deny"
    assert "missing=spawn_process" in other_action_decision.reason

    other_subject_allowed, other_subject_decision = kernel.allowed(
        subject="bob",
        action="spawn_process",
        required=["spawn_process"],
    )
    assert other_subject_allowed is False
    assert other_subject_decision.outcome == "deny"
    assert "missing=spawn_process" in other_subject_decision.reason



def test_capability_kernel_revocation_overrides_parent_and_action_grants():
    kernel = CapabilityKernel()
    kernel.grant(
        subject="alice",
        action="spawn_process",
        capabilities=["spawn_process", "workflow.run.write"],
    )
    kernel.revoke(subject="alice", capability="spawn_process")

    effective = kernel.materialize_capabilities(
        subject="alice",
        parent_capabilities=["spawn_process", "project_view"],
    )
    assert "spawn_process" not in effective
    assert "project_view" in effective
    assert "workflow.run.write" in effective

    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process", "workflow.run.write"],
        parent_capabilities=["spawn_process"],
    )
    assert allowed is False
    assert decision.outcome == "deny"
    assert "revoked=spawn_process" in decision.reason
    assert "missing=*" in decision.reason



def test_capability_kernel_require_appends_audit_log_on_deny():
    kernel = CapabilityKernel()

    with pytest.raises(HTTPException) as exc:
        kernel.require(
            subject="alice",
            action="spawn_process",
            required=["spawn_process", "workflow.run.write"],
        )

    assert exc.value.status_code == 403
    assert "requires capability" in str(exc.value.detail)
    assert len(kernel.audit_log) == 1
    decision = kernel.audit_log[-1]
    assert decision.subject == "alice"
    assert decision.action == "spawn_process"
    assert decision.outcome == "deny"
    assert decision.required == ("spawn_process", "workflow.run.write")



def test_capability_kernel_materialize_includes_parent_minus_revoked_plus_subject_grants():
    kernel = CapabilityKernel()
    kernel.grant(subject="alice", action="project_view", capabilities=["read_security_scope"])
    kernel.revoke(subject="alice", capability="workflow.run.read")

    effective = kernel.materialize_capabilities(
        subject="alice",
        parent_capabilities=["workflow.run.read", "project_view"],
    )

    assert effective == ("project_view", "read_security_scope")



def test_submit_workflow_run_requires_both_spawn_and_workflow_run_write(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )

    setup_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
                "workflow.design.write",
                "write_graph",
                "spawn_process",
                "workflow.run.write",
                "workflow.run.read",
            ],
            "sub": "setup-user",
        }
    )
    try:
        _build_minimal_workflow(
            service,
            workflow_id="wf.capability.requirements",
            designer_id="setup-user",
        )
        created = service.create_conversation(user_id="user-requirements")
    finally:
        claims_ctx.reset(setup_token)

    spawn_only_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["spawn_process", "workflow.run.read", "project_view"],
            "sub": "spawn-only",
        }
    )
    try:
        with pytest.raises(HTTPException):
            service.submit_workflow_run(
                workflow_id="wf.capability.requirements",
                conversation_id=created["conversation_id"],
                turn_node_id=created["start_node_id"],
                user_id="user-requirements",
                initial_state={
                    "user_id": "user-requirements",
                    "conversation_id": created["conversation_id"],
                },
            )
    finally:
        claims_ctx.reset(spawn_only_token)

    write_only_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.run.write", "workflow.run.read", "project_view"],
            "sub": "write-only",
        }
    )
    try:
        with pytest.raises(HTTPException):
            service.submit_workflow_run(
                workflow_id="wf.capability.requirements",
                conversation_id=created["conversation_id"],
                turn_node_id=created["start_node_id"],
                user_id="user-requirements",
                initial_state={
                    "user_id": "user-requirements",
                    "conversation_id": created["conversation_id"],
                },
            )
    finally:
        claims_ctx.reset(write_only_token)



def test_capability_approval_is_subject_local_and_does_not_leak_to_other_subjects(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )

    approver_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
                "workflow.design.write",
                "write_graph",
                "workflow.run.read",
            ],
            "sub": "subject-a",
        }
    )
    try:
        service.capability_approve(
            action="spawn_process",
            capabilities=["spawn_process", "workflow.run.write"],
        )
        snapshot = service.capability_snapshot()
        assert any(
            row["subject"] == "subject-a"
            and row["action"] == "spawn_process"
            and "spawn_process" in row["capabilities"]
            for row in snapshot["approvals"]
        )

        _build_minimal_workflow(
            service,
            workflow_id="wf.capability.subject_local",
            designer_id="subject-a",
        )
        created = service.create_conversation(user_id="user-subject-local")
    finally:
        claims_ctx.reset(approver_token)

    other_subject_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": [
                "project_view",
                "read_security_scope",
                "workflow.run.read",
            ],
            "sub": "subject-b",
        }
    )
    try:
        snapshot = service.capability_snapshot()

        assert not any(
            row["subject"] == "subject-b"
            and row["action"] == "spawn_process"
            for row in snapshot["approvals"]
        )

        with pytest.raises(HTTPException) as exc:
            service.submit_workflow_run(
                workflow_id="wf.capability.subject_local",
                conversation_id=created["conversation_id"],
                user_id="user-subject-local",
            )
        assert exc.value.status_code == 403
    finally:
        claims_ctx.reset(other_subject_token)


def test_capability_allow_with_direct_grant():
    kernel = CapabilityKernel()
    kernel.grant(
        subject="alice",
        action="spawn_process",
        capabilities=["spawn_process", "workflow.run.write"],
    )
    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process", "workflow.run.write"],
    )
    assert allowed is True
    assert decision.outcome == "allow"


def test_capability_deny_when_required_missing():
    kernel = CapabilityKernel()
    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process", "workflow.run.write"],
    )
    assert allowed is False
    assert decision.outcome == "deny"
    assert "missing=spawn_process,workflow.run.write" in decision.reason


def test_capability_snapshot_requires_project_view():
    kernel = CapabilityKernel()
    with pytest.raises(HTTPException):
        kernel.require(
            subject="alice",
            action="project_view",
            required=["project_view"],
        )


def test_capability_action_scope_isolated_between_actions():
    kernel = CapabilityKernel()
    kernel.grant(
        subject="alice",
        action="spawn_process",
        capabilities=["spawn_process", "workflow.run.write"],
    )
    allowed, decision = kernel.allowed(
        subject="alice",
        action="invoke_tool",
        required=["spawn_process"],
    )
    assert allowed is False
    assert decision.outcome == "deny"


def test_capability_revocation_overrides_grant():
    kernel = CapabilityKernel()
    kernel.grant(
        subject="alice",
        action="spawn_process",
        capabilities=["spawn_process", "workflow.run.write"],
    )
    kernel.revoke(subject="alice", capability="spawn_process")
    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process"],
        parent_capabilities=["spawn_process"],
    )
    assert allowed is False
    assert decision.outcome == "deny"
    assert "revoked=spawn_process" in decision.reason


def test_capability_revocation_overrides_action_specific_approval():
    kernel = CapabilityKernel()
    kernel.grant(subject="alice", action="spawn_process", capabilities=["spawn_process"])
    kernel.revoke(subject="alice", capability="spawn_process")
    allowed, _decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process"],
    )
    assert allowed is False


def test_capability_deny_appends_audit():
    kernel = CapabilityKernel()
    with pytest.raises(HTTPException):
        kernel.require(
            subject="alice",
            action="spawn_process",
            required=["spawn_process"],
        )
    assert kernel.audit_log
    assert kernel.audit_log[-1].outcome == "deny"


def test_capability_allow_appends_audit():
    kernel = CapabilityKernel()
    kernel.grant(subject="alice", action="spawn_process", capabilities=["spawn_process"])
    kernel.require(
        subject="alice",
        action="spawn_process",
        required=["spawn_process"],
    )
    assert kernel.audit_log
    assert kernel.audit_log[-1].outcome == "allow"


def test_submit_turn_for_answer_requires_send_message(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )
    token = claims_ctx.set(
        {
            "ns": "conversation",
            "role": "rw",
            "capabilities": ["project_view", "spawn_process"],
            "sub": "msg-tester",
        }
    )
    try:
        created = service.create_conversation(user_id="user-msg")
        with pytest.raises(HTTPException):
            service.submit_turn_for_answer(
                conversation_id=created["conversation_id"],
                user_id="user-msg",
                text="hello",
            )
    finally:
        claims_ctx.reset(token)


def test_submit_workflow_run_requires_spawn_process_and_workflow_run_write(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )
    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["spawn_process", "workflow.run.read", "project_view"],
            "sub": "runner",
        }
    )
    try:
        with pytest.raises(HTTPException):
            service.submit_workflow_run(
                workflow_id="wf.missing.write",
                conversation_id="conv-x",
                user_id="user-x",
            )
    finally:
        claims_ctx.reset(token)


def test_get_run_requires_workflow_run_read(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )
    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["spawn_process", "workflow.run.write", "project_view"],
            "sub": "runner",
        }
    )
    try:
        with pytest.raises(HTTPException):
            service.get_run("run-x")
    finally:
        claims_ctx.reset(token)


def test_capability_snapshot_requires_projection_and_scope_read(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
    )
    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["project_view"],
            "sub": "tester-scope",
        }
    )
    try:
        with pytest.raises(HTTPException):
            service.capability_snapshot()
    finally:
        claims_ctx.reset(token)


def test_parent_capabilities_do_not_bypass_missing_required_capability():
    kernel = CapabilityKernel()
    kernel.grant(subject="alice", action="spawn_process", capabilities=["spawn_process"])
    kernel.revoke(subject="alice", capability="spawn_process")
    allowed, decision = kernel.allowed(
        subject="alice",
        action="spawn_process",
        required=["spawn_process"],
        parent_capabilities=["spawn_process"],
    )
    assert allowed is False
    assert decision.outcome == "deny"
    assert "revoked=spawn_process" in decision.reason
