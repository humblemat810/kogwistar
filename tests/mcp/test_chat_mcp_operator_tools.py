"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_workflow_message_orphans_repair_denies_before_dispatch(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    meta_store = service._execution_meta_store()
    accesses: list[str] = []
    original_list = meta_store.list_projected_lane_messages
    original_update = meta_store.update_projected_lane_message_status

    def track_list(**kwargs: Any) -> Any:
        accesses.append("list")
        return original_list(**kwargs)

    def track_update(**kwargs: Any) -> Any:
        accesses.append("update")
        return original_update(**kwargs)

    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_list)
    monkeypatch.setattr(meta_store, "update_projected_lane_message_status", track_update)
    arguments = {"inbox_id": "repair-auth", "limit": 10}

    with _claims("rw", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.message_orphans_repair", arguments, enforce_visibility=True
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "workflow"), ("rw", "conversation")):
        with _claims(
            role,
            namespace,
            sub="alice",
            scope_parts={"capabilities": ["project_view"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.message_orphans_repair", arguments, enforce_visibility=True
            )
    assert accesses == []


@pytest.mark.asyncio
async def test_mcp_workflow_message_orphans_repair_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-repair-orphans-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    meta_store = service._execution_meta_store()
    inbox_id = contract["mcp_arguments"]["inbox_id"]
    for message_id, lease_seconds in (
        (contract["mcp_expected_repaired_message_ids"][0], -1),
        (contract["mcp_active_message_id"], 60),
    ):
        meta_store.project_lane_message(
            message_id=message_id,
            namespace="workflow",
            purpose="system",
            inbox_id=inbox_id,
            conversation_id="mcp-repair-contract",
            recipient_id="python-worker",
            sender_id="scheduler",
            msg_type="workflow.run.execute",
            status="pending",
            created_at=1,
            available_at=0,
            run_id=f"run-{message_id}",
            step_id="start",
            correlation_id=f"corr-{message_id}",
            payload_json="{}",
        )
        claimed = meta_store.claim_projected_lane_messages(
            namespace="workflow",
            inbox_id=inbox_id,
            claimed_by="repair-contract-worker",
            limit=1,
            lease_seconds=lease_seconds,
            message_ids=[message_id],
        )
        assert [row.message_id for row in claimed] == [message_id]

    with _claims(
        "rw",
        "workflow",
        sub="orphan-repair-contract",
        scope_parts={"capabilities": ["project_view"]},
    ):
        result = await server.mcp._execute(
            contract["mcp_operation"],
            contract["mcp_arguments"],
            enforce_visibility=True,
        )

    assert _structured(result) == {
        "repaired_message_ids": contract["mcp_expected_repaired_message_ids"]
    }
    active = meta_store.list_projected_lane_messages(
        namespace="workflow", inbox_id=inbox_id, status="claimed", limit=10
    )
    assert [row.message_id for row in active] == [contract["mcp_active_message_id"]]


@pytest.mark.asyncio
async def test_mcp_workflow_operator_dashboard_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    required = {"project_view", "workflow.run.read", "read_security_scope"}
    meta_store = service._execution_meta_store()
    reads: list[str] = []
    original_list_runs = meta_store.list_server_runs
    original_list_messages = meta_store.list_projected_lane_messages

    def track_runs(**kwargs: Any) -> Any:
        reads.append("runs")
        return original_list_runs(**kwargs)

    def track_messages(**kwargs: Any) -> Any:
        reads.append("messages")
        return original_list_messages(**kwargs)

    monkeypatch.setattr(meta_store, "list_server_runs", track_runs)
    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_messages)
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "operator-dashboard-v1.json"
        ).read_text(encoding="utf-8")
    )

    for missing in required:
        capabilities = sorted(required - {missing})
        with _claims(
            "ro",
            "workflow",
            sub=f"dashboard-missing-{missing}",
            scope_parts={"capabilities": capabilities},
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                contract["mcp_operation"],
                contract["mcp_arguments"],
                enforce_visibility=True,
            )
        assert denied.value.status_code == 403

    for role, namespace in (("ro", "conversation"),):
        with _claims(
            role,
            namespace,
            sub="dashboard-scope-denied",
            scope_parts={"capabilities": sorted(required)},
        ):
            denied = await server.mcp._handle_call_tool(
                None,
                types.CallToolRequestParams(
                    name=contract["mcp_operation"],
                    arguments=contract["mcp_arguments"],
                ),
            )
        assert denied.is_error is True
    assert reads == []

    with _claims(
        "ro", "workflow", sub="dashboard-authorized", scope_parts={"capabilities": sorted(required)}
    ):
        result = await server.mcp._execute(
            contract["mcp_operation"],
            contract["mcp_arguments"],
            enforce_visibility=True,
        )
    dashboard = _structured(result)
    assert all(field in dashboard for field in contract["dashboard_fields"])


@pytest.mark.asyncio
async def test_mcp_workflow_operator_inbox_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-operator-inbox-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "ro",
        "workflow",
        sub="operator-inbox-contract",
        scope_parts={"capabilities": ["project_view"]},
    ):
        result = await server.mcp._execute(
            contract["operation"], contract["arguments"], enforce_visibility=True
        )
    assert _structured(result) == contract["expected_result"]


@pytest.mark.asyncio
async def test_mcp_workflow_operator_inbox_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    meta_store = service._execution_meta_store()
    reads: list[str] = []
    original_list_messages = meta_store.list_projected_lane_messages
    original_get_nodes = conversation_engine.read.get_nodes

    def track_messages(**kwargs: Any) -> Any:
        reads.append("messages")
        return original_list_messages(**kwargs)

    def track_nodes(*args: Any, **kwargs: Any) -> Any:
        reads.append("nodes")
        return original_get_nodes(*args, **kwargs)

    monkeypatch.setattr(meta_store, "list_projected_lane_messages", track_messages)
    monkeypatch.setattr(conversation_engine.read, "get_nodes", track_nodes)
    arguments = {"inbox_id": "auth-deny", "limit": 10}

    with _claims(
        "ro", "workflow", sub="inbox-no-cap", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            "workflow.operator_inbox", arguments, enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="inbox-wrong-scope",
        scope_parts={"capabilities": ["project_view"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            "workflow.operator_inbox", arguments, enforce_visibility=True
        )
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_budget_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "workflow-budget-http-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "ro",
        "workflow",
        sub="budget-snapshot-contract",
        scope_parts={"capabilities": contract["required_capabilities"]},
    ):
        result = await server.mcp._execute(
            "workflow.budget_snapshot", {}, enforce_visibility=True
        )

    payload = _structured(result)
    for field in contract["required_fields"]:
        assert field in payload
    for field in contract["required_cost_ledger_fields"]:
        assert field in payload["cost_ledger"]
    assert payload["cost_ledger"] == contract["expected_cost_ledger"]
    assert payload["budget_model"] == contract["budget_model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("workflow.budget_snapshot", {}),
        ("workflow.budget_history", {"limit": 10}),
    ],
)
async def test_mcp_workflow_budget_reads_deny_before_ledger_read(
    monkeypatch, engine_triplet, operation, arguments
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    ledger_reads: list[bool] = []
    original_snapshot = service.cost_ledger.snapshot

    def track_snapshot() -> Any:
        ledger_reads.append(True)
        return original_snapshot()

    monkeypatch.setattr(service.cost_ledger, "snapshot", track_snapshot)
    required = {"project_view", "read_security_scope"}
    for missing in required:
        with _claims(
            "ro",
            "workflow",
            sub=f"budget-missing-{missing}",
            scope_parts={"capabilities": sorted(required - {missing})},
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                operation, arguments, enforce_visibility=True
            )
        assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="budget-wrong-scope",
        scope_parts={"capabilities": sorted(required)},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(operation, arguments, enforce_visibility=True)

    assert ledger_reads == []
    with _claims(
        "rw",
        "workflow",
        sub="budget-authorized-rw",
        scope_parts={"capabilities": sorted(required)},
    ):
        authorized = await server.mcp._execute(
            operation, arguments, enforce_visibility=True
        )
    assert _structured(authorized)
    assert ledger_reads == [True]


@pytest.mark.asyncio
async def test_mcp_workflow_design_requires_designer_id_and_enforces_subject(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with _claims("rw", "workflow", sub="alice"):
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.missing_designer"},
            )
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.mismatch", "designer_id": "bob"},
            )
