"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_workflow_process_views(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-blocked-runs-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    run = fixture["run"]
    process_table_fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-process-table-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    service.run_registry.create_run(
        run_id=run["run_id"],
        conversation_id=run["conversation_id"],
        workflow_id=run["workflow_id"],
        user_id=run["user_id"],
        user_turn_node_id=run["user_turn_node_id"],
        status="running",
    )
    service.run_registry.append_event(
        run["run_id"], "run.stage", {"stage": "execute"}
    )
    service.run_registry.update_status(
        run["run_id"],
        status=run["status"],
        started=True,
    )
    from fastapi import HTTPException

    process_reads: list[int] = []
    meta_store = service._execution_meta_store()
    original_list_server_runs = meta_store.list_server_runs

    def track_run_store_read(
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        process_reads.append(limit)
        return original_list_server_runs(
            status=status,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            limit=limit,
        )

    monkeypatch.setattr(meta_store, "list_server_runs", track_run_store_read)
    with _claims(
        "ro", "workflow", sub="blocked-runs-no-cap", scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                fixture["operation"], fixture["arguments"], enforce_visibility=True
            )
    assert denied.value.status_code == 403
    with _claims(
        "ro", "workflow", sub="process-table-no-cap", scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as denied_process_table:
            await server.mcp._execute(
                process_table_fixture["operation"],
                process_table_fixture["arguments"],
                enforce_visibility=True,
            )
    assert denied_process_table.value.status_code == 403
    assert process_reads == []

    with _claims("ro", "workflow"):
        processes = _structured(
            await server.mcp.call_tool(
                process_table_fixture["operation"],
                process_table_fixture["arguments"],
            )
        )["processes"]
        actual_processes = [
            {
                field: process[field]
                for field in (
                    "process_id",
                    "process_kind",
                    "status",
                    "workflow_id",
                    "conversation_id",
                )
            }
            for process in processes
        ]
        assert actual_processes == process_table_fixture["expected_processes"]

        blocked = _structured(
            await server.mcp.call_tool(
                fixture["operation"], fixture["arguments"]
            )
        )["processes"]
        actual = [
            {
                field: item[field]
                for field in ("process_id", "process_kind", "status")
            }
            for item in blocked
            if item["process_id"] == run["run_id"]
        ]
        assert actual == fixture["expected_processes"]

        timeline = _structured(
            await server.mcp.call_tool(
                "workflow.process_timeline",
                {"run_id": run["run_id"], "limit": 10},
            )
        )
        assert timeline["run_id"] == run["run_id"]
        assert timeline["events"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_tools_round_trip(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
            "project_view",
            "service.manage",
            "service.inspect",
            "service.heartbeat",
        ]
        created = service.create_conversation(user_id="svc-mcp-user")

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "service.manage",
            "service.inspect",
            "service.heartbeat",
            "project_view",
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
        ]
        declared = _structured(
            await server.mcp.call_tool(
                "workflow.service_declare",
                {
                    "service_id": "svc.mcp.demo",
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.service.mcp",
                    "target_config": {"conversation_id": created["conversation_id"]},
                },
            )
        )
        assert declared["service_id"] == "svc.mcp.demo"

        heartbeat = _structured(
            await server.mcp.call_tool(
                "workflow.service_heartbeat",
                {
                    "service_id": "svc.mcp.demo",
                    "instance_id": "mcp-1",
                    "payload": {"beat": 1},
                },
            )
        )
        assert heartbeat["health_status"] == "healthy"

        listed = _structured(
            await server.mcp.call_tool("workflow.service_list", {"limit": 20})
        )["services"]
        assert any(item["service_id"] == "svc.mcp.demo" for item in listed)

        triggered = _structured(
            await server.mcp.call_tool(
                "workflow.service_trigger",
                {"service_id": "svc.mcp.demo", "trigger_type": "external event"},
            )
        )
        assert triggered["current_child_run_id"]

        events = _structured(
            await server.mcp.call_tool(
                "workflow.service_events", {"service_id": "svc.mcp.demo", "limit": 50}
            )
        )["events"]
        assert any(evt["event_type"] == "service.triggered" for evt in events)


def test_chat_run_service_execution_meta_comes_from_conversation_engine(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    assert service._execution_meta_store() is conversation_engine.meta_sqlite
    assert service.run_registry.meta_store is conversation_engine.meta_sqlite


def test_chat_run_service_process_table_uses_claim_scope_not_backend_namespace(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.namespace = "backend-tenant"
    service.run_registry.create_run(
        run_id="run-scope-1",
        conversation_id="conv-scope-1",
        workflow_id="wf-scope-1",
        user_id="user-scope-1",
        user_turn_node_id="turn-scope-1",
        status="running",
    )
    token = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "store-tenant",
            "execution_ns": "exec-tenant",
            "security_scope": "tenant-a",
        }
    )
    try:
        rows = service.list_process_table(limit=10)
    finally:
        claims_ctx.reset(token)
    assert rows
    row = rows[0]
    assert row["namespace"] == "exec-tenant"
    assert row["storage_namespace"] == "store-tenant"
    assert row["security_scope"] == "tenant-a"


def test_chat_run_service_resume_contract_exposes_checkpoint_metadata(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.write.add_node(
        WorkflowCheckpointNode(
            id="wf_ckpt|run-resume-1|0",
            label="Checkpoint 0",
            type="entity",
            doc_id="wf_ckpt|run-resume-1|0",
            summary="checkpoint",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata={
                "entity_type": "workflow_checkpoint",
                "run_id": "run-resume-1",
                "workflow_id": "wf-resume-1",
                "step_seq": 0,
                "checkpoint_schema_version": 1,
                "state_json": "{\"foo\": \"bar\"}",
            },
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )
    contract = service.resume_contract("run-resume-1")
    assert contract["run_id"] == "run-resume-1"
    assert contract["latest_checkpoint_step_seq"] == 0
    assert contract["checkpoint_schema_version"] == 1
    assert "checkpoint_schema_version" in contract["persisted_keys"]
    assert "_rt_join" in contract["ephemeral_keys"]


def test_chat_run_service_operator_inbox_respects_security_scope(
    monkeypatch, engine_triplet
):
    limit_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-operator-inbox-limit-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service._execution_meta_store().project_lane_message(
        message_id="orphaned-projection",
        namespace="conversation",
        purpose="operator",
        inbox_id="inbox:ops",
        conversation_id="conv-scope-msg",
        recipient_id="lane:b",
        sender_id="lane:a",
        msg_type="request.orphaned",
        status="pending",
        created_at=1,
        available_at=1,
        run_id=None,
        step_id=None,
        correlation_id=None,
        payload_json='{"secret":"must-not-leak"}',
    )
    token_send = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-a",
        }
    )
    try:
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.private",
            payload={"hello": "world"},
            security_scope="tenant-a",
        )
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.shared",
            payload={"hello": "shared"},
            security_scope="tenant-a",
            shared_scope=True,
        )
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.visible",
            payload={"hello": "visible"},
            security_scope="tenant-a",
            shared_scope=True,
        )
    finally:
        claims_ctx.reset(token_send)

    token_read = claims_ctx.set(
        {
            "ns": "workflow",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-b",
        }
    )
    try:
        rows = service.list_operator_inbox(inbox_id="inbox:ops", limit=1)
        limited_rows = [
            service.list_operator_inbox(inbox_id="inbox:ops", limit=case["limit"])
            for case in limit_contract["cases"]
        ]
    finally:
        claims_ctx.reset(token_read)
    assert len(rows) == 1
    assert rows[0]["msg_type"] == "request.shared"
    assert rows[0]["visibility"] == "shared"
    assert rows[0]["message_id"] != "orphaned-projection"
    assert [len(messages) for messages in limited_rows] == [
        case["expected_message_count"] for case in limit_contract["cases"]
    ]


@pytest.mark.asyncio
async def test_mcp_chat_submit_cancel_and_workflow_diagnostics(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )

    with _claims("rw", "conversation"):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-mcp"})
        )
        conversation_id = created["conversation_id"]
        asked = _structured(
            await server.mcp.call_tool(
                "conversation.ask",
                {
                    "conversation_id": conversation_id,
                    "user_id": "u-mcp",
                    "text": "cancel this",
                },
            )
        )
        run_id = asked["run_id"]

        cancelled = _structured(
            await server.mcp.call_tool("conversation.cancel_run", {"run_id": run_id})
        )
        assert cancelled["status"] == "cancelling"
        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await server.mcp.call_tool(
                    "conversation.run_status", {"run_id": run_id}
                )
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Run did not reach cancelled status")

        transcript = _structured(
            await server.mcp.call_tool(
                "conversation.get_transcript", {"conversation_id": conversation_id}
            )
        )
        assert [turn["role"] for turn in transcript["turns"]] == ["user"]

    checkpoint = WorkflowCheckpointNode(
        id="wf_ckpt|diag-run|0",
        label="Checkpoint 0",
        type="entity",
        doc_id="wf_ckpt|diag-run|0",
        summary="workflow checkpoint",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_checkpoint",
            "run_id": "diag-run",
            "workflow_id": "wf.diag",
            "step_seq": 0,
            "state_json": json.dumps({"answer": "draft"}),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.write.add_node(checkpoint)

    with _claims("ro", "workflow"):
        checkpoint_args = {"run_id": "diag-run", "step_seq": 0}
        got = _structured(
            await server.mcp.call_tool(
                "workflow.run_checkpoint_get", checkpoint_args
            )
        )
        checkpoint_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-run-checkpoint-get-mcp-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert checkpoint_contract["operation"] == "workflow.run_checkpoint_get"
        assert set(checkpoint_args) == set(checkpoint_contract["request_fields"])
        assert set(got) == set(checkpoint_contract["response_fields"])
        assert got["run_id"] == "diag-run"
        assert got["step_seq"] == 0
        assert isinstance(got["state"], dict)
        assert got["state"]["answer"] == "draft"

        # replay_to needs at least one checkpoint and no subsequent steps for target=0
        replay = _structured(
            await server.mcp.call_tool(
                "workflow.run_replay", {"run_id": "diag-run", "target_step_seq": 0}
            )
        )
        assert replay["state"]["answer"] == "draft"

        resume_contract = _structured(
            await server.mcp.call_tool(
                "workflow.run_resume_contract", {"run_id": "missing-mcp-run"}
            )
        )
        fixture = json.loads(
            (Path(__file__).resolve().parents[2] / "contracts/golden/run-resume-contract-v1.json")
            .read_text(encoding="utf-8")
        )
        assert set(resume_contract) == set(fixture["required_fields"])
        assert resume_contract["run_id"] == "missing-mcp-run"
        for field, expected in fixture["no_checkpoint"].items():
            assert resume_contract[field] == expected
        assert resume_contract["supported_wait_reasons"] == fixture[
            "supported_wait_reasons"
        ]


@pytest.mark.asyncio
async def test_mcp_workflow_runtime_submit_cancel(monkeypatch, engine_triplet):
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles

    submit_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-submit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    expected_owner = submit_contract["python_service_arguments"]["user_id"]
    cancel_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/run-cancel-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_status_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-status-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_cancel_runner,
    )
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=lambda _workflow_id, _role: None,
    )

    with _claims("rw", "conversation", scope_parts={"user_id": expected_owner}):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-wf-mcp"})
        )
        conversation_id = created["conversation_id"]

    with _claims("rw", "workflow", scope_parts={"user_id": expected_owner}):
        submitted = _structured(
            await workflow_mcp.call_tool(
                submit_contract["tool"],
                {
                    **submit_contract["arguments"],
                    "conversation_id": conversation_id,
                    "user_id": "caller-controlled-owner",
                },
            )
        )
        run_id = submitted["run_id"]
        stored_run = server.run_registry.get_run(run_id)
        assert stored_run is not None
        assert stored_run["user_id"] == expected_owner
        cancelling = _structured(
            await workflow_mcp.call_tool("workflow.run_cancel", {"run_id": run_id})
        )
        assert cancelling["status"] == "cancelling"

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await workflow_mcp.call_tool("workflow.run_status", {"run_id": run_id})
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Workflow run did not reach cancelled status")
        assert set(run_status_contract["required_fields"]).issubset(status)
        assert status["status"] in run_status_contract["terminal_statuses"]
        assert status["terminal"] is True

        events = _structured(
            await workflow_mcp.call_tool("workflow.run_events", {"run_id": run_id})
        )
        names = [str(evt.get("event_type") or "") for evt in events.get("events", [])]
        assert cancel_contract["nonterminal_cancel_event"] in names
        assert "run.cancelled" in names

        retried = _structured(
            await workflow_mcp.call_tool("workflow.run_cancel", {"run_id": run_id})
        )
        assert retried["status"] == "cancelled"
        replayed_events = _structured(
            await workflow_mcp.call_tool("workflow.run_events", {"run_id": run_id})
        )
        replayed_names = [
            str(evt.get("event_type") or "") for evt in replayed_events.get("events", [])
        ]
        assert cancel_contract["terminal_retry_appends_event"] is False
        assert replayed_names == names

        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1


@pytest.mark.asyncio
async def test_mcp_workflow_run_cancel_denies_missing_write_capability(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service.run_registry.create_run(
        run_id="mcp-cancel-auth-run",
        conversation_id="mcp-cancel-auth-conversation",
        workflow_id="mcp-cancel-auth-workflow",
        user_id="alice",
        user_turn_node_id="mcp-cancel-auth-turn",
        status="running",
    )

    with _claims("rw", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.run_cancel",
                {"run_id": "mcp-cancel-auth-run"},
                enforce_visibility=True,
            )
    assert denied.value.status_code == 403

    run = service.run_registry.get_run("mcp-cancel-auth-run")
    assert run is not None
    assert run["status"] == "running"
    assert run["cancel_requested"] is False

    service.run_registry.create_run(
        run_id="mcp-cancel-terminal-run",
        conversation_id="mcp-cancel-terminal-conversation",
        workflow_id="mcp-cancel-terminal-workflow",
        user_id="alice",
        user_turn_node_id="mcp-cancel-terminal-turn",
        status="cancelled",
    )
    with _claims("rw", "workflow", sub="alice", scope_parts={"capabilities": []}):
        terminal_retry = _structured(
            await server.mcp._execute(
                "workflow.run_cancel",
                {"run_id": "mcp-cancel-terminal-run"},
                enforce_visibility=True,
            )
        )
    assert terminal_retry["status"] == "cancelled"


@pytest.mark.asyncio
async def test_mcp_workflow_run_submit_denies_missing_capabilities_before_admission(
    monkeypatch, engine_triplet
):
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    submissions: list[dict] = []
    monkeypatch.setattr(
        service._run_execution,
        "submit_workflow_run",
        lambda **kwargs: submissions.append(kwargs) or {"status": "queued"},
    )
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=lambda _workflow_id, _role: None,
    )
    arguments = {
        "workflow_id": "unauthorized-submit-workflow",
        "conversation_id": "unauthorized-submit-conversation",
    }

    for capabilities in (["workflow.run.write"], ["spawn_process"]):
        with _claims(
            "rw",
            "workflow",
            sub="alice",
            scope_parts={"capabilities": capabilities},
        ):
            denied = await workflow_mcp._handle_call_tool(
                None,
                types.CallToolRequestParams(
                    name="workflow.run_submit",
                    arguments=arguments,
                ),
            )
        assert denied.is_error is True
        assert "403" in denied.content[0].text

    assert submissions == []
    assert (
        service.run_registry.meta_store.list_server_runs(
            workflow_id="unauthorized-submit-workflow",
            conversation_id="unauthorized-submit-conversation",
            limit=20,
        )
        == []
    )


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_redo(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    workflow_id = "wf.design.mcp.undo_redo"
    designer_id = "designer-mcp"
    with _claims("rw", "workflow", sub=designer_id):
        n1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|start",
                    "label": "Start",
                    "op": "start",
                    "start": True,
                },
            )
        )
        assert n1["node_id"] == f"wf|{workflow_id}|start"
        n2 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|end",
                    "label": "End",
                    "op": "end",
                    "terminal": True,
                },
            )
        )
        assert n2["node_id"] == f"wf|{workflow_id}|end"
        e1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "src": f"wf|{workflow_id}|start",
                    "dst": f"wf|{workflow_id}|end",
                    "relation": "wf_next",
                    "is_default": True,
                },
            )
        )
        assert e1["edge_id"] == f"wf|{workflow_id}|e|start_end"

        hist_before = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        assert int(hist_before["current_version"]) >= 3
        assert bool(hist_before["can_undo"]) is True

        undone = _structured(
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert undone["status"] in {"ok", "noop"}

        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )

        redone = _structured(
            await server.mcp.call_tool(
                "workflow.design_redo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert redone["status"] in {"ok", "noop"}

        deleted_after_redo = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )
        )
        assert bool(deleted_after_redo.get("deleted")) is True
        hist_after = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        timeline_ops = [
            str(item.get("op") or "") for item in hist_after.get("timeline", [])
        ]
        assert "UNDO_APPLIED" in timeline_ops
        assert "REDO_APPLIED" in timeline_ops


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def delete_edge(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_delete_edge", delete_edge)
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [
        {
            **contract["arguments"],
            "actor_sub": contract["arguments"]["designer_id"],
            "source": "mcp",
        }
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_delete_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_delete_edge",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def upsert_edge(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_upsert_edge", upsert_edge)
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [
        {
            **contract["arguments"],
            "actor_sub": contract["arguments"]["designer_id"],
            "source": "mcp",
        }
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_design_edge_upsert_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_upsert_edge",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []

    def history(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service._workflow_design, "workflow_design_history", history)
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [contract["arguments"]]


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_empty_state_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_design_history_denies_before_read(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_history",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": []},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "rw",
        "conversation",
        sub="alice",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    workflow_id = contract["arguments"]["workflow_id"]
    node_id = contract["arguments"]["node_id"]
    designer_id = contract["arguments"]["designer_id"]
    with _claims(
        "rw",
        "workflow",
        sub=designer_id,
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": workflow_id,
                "designer_id": designer_id,
                "node_id": node_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_delete_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-delete-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_delete_node",
        lambda **kwargs: calls.append(kwargs) or contract["result"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    with _claims(
        "rw",
        "workflow",
        sub=contract["arguments"]["designer_id"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected
    for field in contract["integer_result_fields"]:
        assert isinstance(body[field], int)


@pytest.mark.asyncio
async def test_mcp_workflow_design_node_upsert_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_upsert_node",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-undo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    arguments = contract["arguments"]
    with _claims(
        "rw",
        "workflow",
        sub=arguments["designer_id"],
        scope_parts={
            "capabilities": [*contract["caller_capabilities"], "write_graph"]
        },
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": arguments["workflow_id"],
                "designer_id": arguments["designer_id"],
                "node_id": "undo-node-contract",
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        result = await server.mcp.call_tool(contract["tool"], arguments)

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected


@pytest.mark.asyncio
async def test_mcp_workflow_design_redo_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-redo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    arguments = contract["arguments"]
    with _claims(
        "rw",
        "workflow",
        sub=arguments["designer_id"],
        scope_parts={
            "capabilities": [*contract["caller_capabilities"], "write_graph"]
        },
    ):
        created = await server.mcp.call_tool(
            "workflow.design_node_upsert",
            {
                "workflow_id": arguments["workflow_id"],
                "designer_id": arguments["designer_id"],
                "node_id": "redo-node-contract",
                "label": "Start",
                "op": "start",
                "start": True,
            },
        )
        assert created.is_error is False
        undone = await server.mcp.call_tool(
            "workflow.design_undo", arguments
        )
        assert undone.is_error is False
        result = await server.mcp.call_tool(contract["tool"], arguments)

    assert result.is_error is False
    body = _structured(result)
    for field, expected in contract["fixed_result_fields"].items():
        assert body[field] == expected, f"{field}: {body!r}"
    for field in contract["integer_result_fields"]:
        assert isinstance(body[field], int)
    for field, length in contract["array_lengths"].items():
        assert len(body[field]) == length
    assert body["commits"]["1"]["action"] == contract["version_one_action"]
    for field in contract["absent_result_fields"]:
        assert field not in body


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-undo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_undo",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    with _claims(
        "rw",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.design.write"]},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_design_redo_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-redo-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_redo",
        lambda **kwargs: calls.append(kwargs) or contract["fixed_result_fields"],
    )

    for missing_capability in contract["caller_capabilities"]:
        capabilities = [
            capability
            for capability in contract["caller_capabilities"]
            if capability != missing_capability
        ]
        with _claims(
            "rw", "workflow", sub="alice", scope_parts={"capabilities": capabilities}
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": contract["caller_capabilities"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert calls == []
