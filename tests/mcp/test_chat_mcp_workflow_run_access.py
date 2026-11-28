"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_chat_tool_visibility_by_namespace(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with _claims("rw", "conversation"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "conversation.create" in names
        assert "conversation.ask" in names
        assert "conversation.cancel_run" in names
        assert "workflow.run_replay" not in names
        assert "doc_parse" not in names

    with _claims("rw", "docs"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "doc_parse" in names
        assert "kg_extract" in names
        assert "store_document" in names

    with _claims("ro", "workflow"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "workflow.run_status" in names
        assert "workflow.run_events" in names
        assert "workflow.run_submit" not in names
        assert "workflow.design_history" in names
        assert "workflow.design_undo" not in names
        assert "workflow.run_checkpoint_get" in names
        assert "workflow.run_replay" in names
        assert "workflow.process_table" in names
        assert "workflow.operator_inbox" in names
        assert "workflow.blocked_runs" in names
        assert "workflow.process_timeline" in names
        assert "conversation.ask" not in names


@pytest.mark.asyncio
async def test_mcp_workflow_run_status_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(service, "get_run", lambda run_id: reads.append(run_id))

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_status", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_status_uses_capability_without_workflow_acl(
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

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    access_checks: list[tuple[str, str]] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: {"run_id": run_id, "workflow_id": "private-workflow"},
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
        require_workflow_access=lambda workflow_id, role: access_checks.append(
            (workflow_id, role)
        ),
    )

    with _claims(
        "ro",
        "workflow",
        sub="outsider",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await workflow_mcp._execute(
                "workflow.run_status",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
        )

    assert result["run_id"] == "private-run"
    assert access_checks == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(
        service, "list_run_events", lambda run_id, **_kwargs: reads.append(run_id)
    )

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_events", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_denies_missing_read_capability_before_read(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    def record_events(run_id: str, **_kwargs: object) -> list[dict[str, object]]:
        reads.append(run_id)
        return []

    monkeypatch.setattr(service._run_execution, "list_run_events", record_events)

    with _claims("ro", "workflow", sub="outsider", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.run_events",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )

    assert denied.value.status_code == 403
    assert reads == []

    with _claims(
        "ro",
        "workflow",
        sub="authorized-reader",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = await server.mcp.call_tool(
            "workflow.run_events", {"run_id": "authorized-run"}
        )
    assert _structured(result) == {"run_id": "authorized-run", "events": []}
    assert reads == ["authorized-run"]

    with _claims("ro", "workflow", sub="legacy-reader"):
        legacy_result = await server.mcp.call_tool(
            "workflow.run_events", {"run_id": "legacy-run"}
        )
    assert _structured(legacy_result) == {"run_id": "legacy-run", "events": []}
    assert reads == ["authorized-run", "legacy-run"]


@pytest.mark.asyncio
async def test_mcp_workflow_run_events_matches_shared_contract(
    monkeypatch, engine_triplet
):
    from pathlib import Path

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-events-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    event = {
        "seq": 7,
        "run_id": contract["arguments"]["run_id"],
        "event_type": "run.started",
        "payload": {"status": "running"},
        "created_at_ms": 1700000000000,
    }
    calls: list[tuple[str, int, int]] = []

    def list_events(run_id: str, *, after_seq: int, limit: int) -> list[dict]:
        calls.append((run_id, after_seq, limit))
        return [event]

    monkeypatch.setattr(service._run_execution, "list_run_events", list_events)
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["operation"], contract["arguments"])
        )

    assert set(contract["response_fields"]).issubset(result)
    assert result == {"run_id": event["run_id"], "events": [event]}
    assert set(contract["required_event_fields"]).issubset(result["events"][0])
    assert calls == [(event["run_id"], 4, 200)]


@pytest.mark.asyncio
async def test_mcp_workflow_run_evidence_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/run-evidence-v1.json"
    )
    expected = json.loads(fixture.read_text(encoding="utf-8"))
    snapshot = {
        "snapshot_node_id": expected["snapshot_node_id"],
        "metadata": expected["snapshot_metadata"],
        "properties": {
            "evidence_pack_digest": json.dumps(
                expected["source_summary"]["evidence_pack_digest"]
            )
        },
    }
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: {
            "run_id": run_id,
            "workflow_id": "evidence-workflow",
            "conversation_id": expected["conversation_id"],
        },
    )
    monkeypatch.setattr(
        service._conversation_queries,
        "latest_snapshot",
        lambda conversation_id, *, run_id, stage=None: snapshot,
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read", "project_view"]},
    ):
        result = _structured(
            await server.mcp.call_tool(
                "workflow.run_evidence", {"run_id": expected["run_id"]}
            )
        )

    assert result == expected


@pytest.mark.asyncio
async def test_mcp_workflow_run_evidence_denies_scope_and_missing_capabilities(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_reads: list[str] = []
    snapshot_reads: list[str] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: run_reads.append(run_id)
        or {
            "workflow_id": "evidence-workflow",
            "conversation_id": "conversation-1",
        },
    )
    monkeypatch.setattr(
        service._conversation_queries,
        "latest_snapshot",
        lambda conversation_id, **_kwargs: snapshot_reads.append(conversation_id)
        or {},
    )

    with _claims("ro", "conversation", sub="outsider"):
        wrong_scope = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_evidence", arguments={"run_id": "private-run"}
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert run_reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as missing_run_read:
            await server.mcp._execute(
                "workflow.run_evidence",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert missing_run_read.value.status_code == 403
    assert run_reads == []

    run_reads.clear()
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(HTTPException) as missing_project_view:
            await server.mcp._execute(
                "workflow.run_evidence",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert missing_project_view.value.status_code == 403
    assert run_reads == ["private-run", "private-run"]
    assert snapshot_reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_lineage_matches_shared_contract(
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

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-lineage-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    access_checks: list[tuple[str, str]] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: {
            "run_id": run_id,
            "workflow_id": "wf-child",
            "conversation_id": "lineage-conversation",
        },
    )
    monkeypatch.setattr(
        service,
        "workflow_run_lineage",
        lambda run_id: contract["response"],
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
        require_workflow_access=lambda workflow_id, role: access_checks.append(
            (workflow_id, role)
        ),
    )

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await workflow_mcp.call_tool(
                "workflow.run_lineage",
                {"run_id": contract["response"]["run_id"], "limit": 2},
            )
        )

    assert result == {**contract["response"], "truncated": False}
    assert access_checks == [
        ("wf-child", "ro"),
        ("wf-child", "ro"),
        ("wf-parent", "ro"),
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_run_lineage_denies_scope_capability_and_child_acl(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
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
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-lineage-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_reads: list[str] = []
    lineage_reads: list[str] = []
    access_checks: list[str] = []
    monkeypatch.setattr(
        service._run_execution,
        "get_run",
        lambda run_id: run_reads.append(run_id)
        or {"workflow_id": "wf-child", "conversation_id": "lineage-conversation"},
    )
    monkeypatch.setattr(
        service,
        "workflow_run_lineage",
        lambda run_id: lineage_reads.append(run_id) or contract["response"],
    )

    def check_workflow_access(workflow_id: str, _role: str) -> None:
        access_checks.append(workflow_id)
        if workflow_id == "wf-parent":
            raise HTTPException(status_code=403, detail="workflow denied")

    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=check_workflow_access,
    )

    with _claims("ro", "conversation", sub="alice"):
        wrong_scope = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_lineage",
                arguments={"run_id": contract["response"]["run_id"]},
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert run_reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as missing_capability:
            await workflow_mcp._execute(
                "workflow.run_lineage",
                {"run_id": contract["response"]["run_id"]},
                enforce_visibility=True,
            )
    assert missing_capability.value.status_code == 403
    assert run_reads == []
    assert lineage_reads == []

    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        denied_child = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_lineage",
                arguments={"run_id": contract["response"]["run_id"]},
            ),
        )
    assert denied_child.is_error is True
    assert denied_child.content[0].text == "403: workflow denied"
    assert "wf-parent" in access_checks
    assert lineage_reads == [contract["response"]["run_id"]]


@pytest.mark.asyncio
async def test_mcp_workflow_process_timeline_matches_shared_contract(
    monkeypatch, engine_triplet
):
    from pathlib import Path
    from types import SimpleNamespace

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-process-timeline-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    event = {
        "seq": 9,
        "run_id": contract["arguments"]["run_id"],
        "event_type": "run.stage",
        "payload": {"stage": "execute"},
        "created_at_ms": 1700000000000,
    }
    calls: list[tuple[str, int, int]] = []

    def list_events(run_id: str, *, after_seq: int, limit: int) -> list[dict]:
        calls.append((run_id, after_seq, limit))
        return [event]

    monkeypatch.setattr(
        service,
        "_execution_meta_store",
        lambda: SimpleNamespace(list_server_run_events=list_events),
    )
    with _claims(
        "ro",
        "workflow",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["operation"], contract["arguments"])
        )

    assert set(contract["response_fields"]).issubset(result)
    assert result == {"run_id": event["run_id"], "events": [event]}
    assert set(contract["required_event_fields"]).issubset(result["events"][0])
    assert calls == [(event["run_id"], 4, 200)]


@pytest.mark.asyncio
async def test_mcp_workflow_process_timeline_denies_wrong_scope_before_read(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[bool] = []
    monkeypatch.setattr(
        service,
        "_execution_meta_store",
        lambda: reads.append(True),
    )

    with _claims(
        "ro",
        "conversation",
        sub="outsider",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.process_timeline",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )

    with _claims("ro", "workflow", sub="outsider", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            await server.mcp._execute(
                "workflow.process_timeline",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )
    assert denied.value.status_code == 403
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(service, "get_run", lambda run_id: reads.append(run_id))

    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_steps", arguments={"run_id": "private-run"}
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_preserves_role_namespace_policy_without_capability(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []

    def get_run(run_id: str) -> dict[str, str]:
        reads.append(f"run:{run_id}")
        return {"workflow_id": "workflow-1"}

    def list_steps(run_id: str) -> list[dict[str, str]]:
        reads.append(f"steps:{run_id}")
        return []

    monkeypatch.setattr(service, "get_run", get_run)
    monkeypatch.setattr(service, "list_steps", list_steps)

    with _claims("ro", "workflow", sub="reader", scope_parts={"capabilities": []}):
        result = await server.mcp.call_tool(
            "workflow.run_steps", {"run_id": "authorized-run"}
        )

    assert _structured(result) == {"run_id": "authorized-run", "steps": []}
    assert reads == ["run:authorized-run", "steps:authorized-run"]


@pytest.mark.asyncio
async def test_chat_service_steps_still_require_read_capability_after_mcp_admission(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    monkeypatch.setattr(
        service._run_inspection,
        "list_steps",
        lambda run_id: reads.append(run_id) or [],
    )

    with _claims("ro", "workflow", sub="reader", scope_parts={"capabilities": []}):
        with pytest.raises(HTTPException) as denied:
            service.list_steps("private-run")

    assert denied.value.status_code == 403
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_enforces_per_workflow_acl_before_returning_steps(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[str] = []
    access_checks: list[tuple[str, str]] = []

    def get_run(run_id: str) -> dict[str, str]:
        reads.append(f"run:{run_id}")
        return {"workflow_id": "private-workflow", "user_id": "run-owner"}

    def list_steps(run_id: str) -> list[dict[str, str]]:
        reads.append(f"steps:{run_id}")
        return [{"step_seq": 1}]

    def deny_workflow_access(workflow_id: str, role: str) -> None:
        access_checks.append((workflow_id, role))
        raise HTTPException(status_code=403, detail="workflow denied")

    monkeypatch.setattr(service, "get_run", get_run)
    monkeypatch.setattr(service, "list_steps", list_steps)
    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=deny_workflow_access,
    )

    with _claims("ro", "workflow", sub="outsider"):
        with pytest.raises(HTTPException) as denied:
            await workflow_mcp._execute(
                "workflow.run_steps",
                {"run_id": "private-run"},
                enforce_visibility=True,
            )

    assert denied.value.status_code == 403
    assert access_checks == [("private-workflow", "ro")]
    assert reads == ["run:private-run"]

    def allow_workflow_access(workflow_id: str, role: str) -> None:
        access_checks.append((workflow_id, role))

    delegated_workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=allow_workflow_access,
    )
    with _claims("ro", "workflow", sub="steps-reader"):
        delegated = await delegated_workflow_mcp._execute(
            "workflow.run_steps",
            {"run_id": "private-run"},
            enforce_visibility=True,
        )

    assert _structured(delegated) == {
        "run_id": "private-run",
        "steps": [{"step_seq": 1}],
    }
    assert access_checks[-1] == ("private-workflow", "ro")
    assert reads[-2:] == ["run:private-run", "steps:private-run"]


@pytest.mark.asyncio
async def test_mcp_workflow_checkpoint_get_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def read_checkpoint(run_id: str, step_seq: int) -> dict[str, Any]:
        reads.append((run_id, step_seq))
        return {}

    monkeypatch.setattr(service, "get_checkpoint", read_checkpoint)
    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_checkpoint_get",
                arguments={"run_id": "private-run", "step_seq": 4},
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_replay_denies_wrong_namespace_before_read(
    monkeypatch, engine_triplet
):
    from mcp import types

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def replay_run(run_id: str, target_step_seq: int) -> dict[str, Any]:
        reads.append((run_id, target_step_seq))
        return {}

    monkeypatch.setattr(service, "replay_run", replay_run)
    with _claims("ro", "conversation", sub="outsider"):
        result = await server.mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_replay",
                arguments={"run_id": "private-run", "target_step_seq": 4},
            ),
        )

    assert result.is_error is True
    assert "403 Forbidden" in result.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_resume_contract_denies_scope_and_missing_capability(
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
    reads: list[str] = []

    def resume_contract(run_id: str) -> dict:
        reads.append(run_id)
        return {"run_id": run_id}

    monkeypatch.setattr(service._run_inspection, "resume_contract", resume_contract)
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

    with _claims("ro", "conversation", sub="alice"):
        wrong_scope = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_resume_contract",
                arguments={"run_id": "private-run"},
            ),
        )
    assert wrong_scope.is_error is True
    assert "403 Forbidden" in wrong_scope.content[0].text
    assert reads == []

    with _claims("ro", "workflow", sub="alice", scope_parts={"capabilities": []}):
        missing_capability = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name="workflow.run_resume_contract",
                arguments={"run_id": "private-run"},
            ),
        )
    assert missing_capability.is_error is True
    assert "403" in missing_capability.content[0].text
    assert reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_resume_dispatches_request_and_returns_result(
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
    resume_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-resume-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    request = {
        "run_id": "resume-mcp-run",
        "suspended_node_id": "approval",
        "suspended_token_id": "approval-token",
        "client_result": {"status": "success", "state_update": [["u", {"approved": True}]]},
        "workflow_id": "resume-mcp-workflow",
        "conversation_id": "resume-mcp-conversation",
        "turn_node_id": "resume-mcp-turn",
        "user_id": "authenticated-owner",
    }
    expected = {
        "workflow_status": "succeeded",
        "final_state": {"approved": True},
        "budget": {},
    }
    calls: list[dict[str, Any]] = []

    def resume_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(service, "resume_run", resume_run)
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

    with _claims(
        "rw",
        "workflow",
        sub="authenticated-owner",
        scope_parts={"capabilities": ["workflow.run.write"]},
    ):
        result = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(name="workflow.run_resume", arguments=request),
        )

    payload = _structured(result)
    assert result.is_error is False
    assert calls == [request]
    assert set(resume_contract["required_fields"]).issubset(payload)
    assert payload == expected
    assert payload["workflow_status"] in resume_contract["completed_statuses"]


@pytest.mark.asyncio
async def test_mcp_workflow_run_resume_denies_missing_write_capability_before_effect(
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

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-resume-auth-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    effects: list[Any] = []
    monkeypatch.setattr(
        service._run_execution,
        "_default_resume_runner",
        lambda request: effects.append(request) or {},
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

    denied_principal = contract["denied_principal"]
    with _claims(
        denied_principal["role"],
        denied_principal["namespace"],
        sub="authenticated-owner",
        scope_parts={"capabilities": denied_principal["capabilities"]},
    ):
        result = await workflow_mcp._handle_call_tool(
            None,
            types.CallToolRequestParams(
                name=contract["tool_name"],
                arguments={
                    "run_id": "private-run",
                    "suspended_node_id": "approval",
                    "suspended_token_id": "approval-token",
                    "client_result": {"status": "success"},
                    "workflow_id": "private-workflow",
                    "conversation_id": "private-conversation",
                    "turn_node_id": "private-turn",
                },
            ),
        )

    assert result.is_error is True
    assert "403:" in result.content[0].text
    assert contract["required_capability"] in result.content[0].text
    if contract["must_not_execute"]:
        assert effects == []


@pytest.mark.asyncio
async def test_mcp_workflow_run_resume_continues_persisted_suspended_workflow(
    monkeypatch, engine_triplet
):
    import asyncio
    import threading

    import kogwistar.server.chat_service_run_execution as run_execution_module
    from kogwistar.runtime.models import RunSuccess, RunSuspended
    from kogwistar.runtime.resolvers import MappingStepResolver
    from kogwistar.runtime.runtime import WorkflowRuntime
    from kogwistar.runtime.runtime_checkpoints import WorkflowRunOwnershipError
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "mcp-resume-integration-workflow"
    run_id = "mcp-resume-integration-run"
    conversation_id = "mcp-resume-integration-conversation"
    turn_node_id = "mcp-resume-integration-turn"
    owner = "mcp-resume-integration-owner"

    with _claims(
        "rw",
        "workflow",
        sub=owner,
        scope_parts={"capabilities": ["workflow.design.write", "write_graph"]},
    ):
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="start",
            label="Start",
            op="mcp-resume-start",
            start=True,
        )
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="approval",
            label="Wait for approval",
            op="mcp-resume-approval",
        )
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="finish",
            label="Finish",
            op="mcp-resume-finish",
            terminal=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=owner,
            edge_id="start-approval",
            src="start",
            dst="approval",
            relation="wf_next",
            is_default=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=owner,
            edge_id="approval-finish",
            src="approval",
            dst="finish",
            relation="wf_next",
            is_default=True,
        )

    resolver = MappingStepResolver()
    finish_calls = []
    finish_calls_lock = threading.Lock()

    @resolver.register("mcp-resume-start")
    def _start(_context):
        return RunSuccess(state_update=[("u", {"started": True})])

    @resolver.register("mcp-resume-approval")
    def _approval(_context):
        return RunSuspended(resume_payload={"question": "approve?"})

    @resolver.register("mcp-resume-finish")
    def _finish(context):
        with finish_calls_lock:
            finish_calls.append(context.state_view.get("approved"))
        return RunSuccess(
            state_update=[
                ("u", {"finished": True, "approved": context.state_view.get("approved")})
            ]
        )

    monkeypatch.setattr("kogwistar.conversation.resolvers.default_resolver", resolver)
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )
    suspended = runtime.run(
        workflow_id=workflow_id,
        conversation_id=conversation_id,
        turn_node_id=turn_node_id,
        initial_state={
            "conversation_id": conversation_id,
            "user_id": owner,
            "turn_node_id": turn_node_id,
            "turn_index": 0,
            "role": "user",
            "user_text": "approve",
            "mem_id": "memory",
        },
        run_id=run_id,
    )
    assert suspended.status == "suspended"

    checkpoints = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    assert checkpoints
    latest = max(
        checkpoints,
        key=lambda node: int(getattr(node, "metadata", {}).get("step_seq", -1)),
    )
    state = latest.metadata["state_json"]
    if isinstance(state, str):
        state = json.loads(state)
    suspended_node_id, _, suspended_token_id, _ = state["_rt_join"]["suspended"][0]

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
    resume_arguments = {
        "run_id": run_id,
        "suspended_node_id": suspended_node_id,
        "suspended_token_id": suspended_token_id,
        "client_result": {
            "status": "success",
            "state_update": [["u", {"approved": True}]],
        },
        "workflow_id": workflow_id,
        "conversation_id": conversation_id,
        "turn_node_id": turn_node_id,
        "user_id": owner,
    }
    with _claims(
        "rw",
        "workflow",
        sub=owner,
        scope_parts={"capabilities": ["workflow.run.write", "approve_action"]},
    ):
        original_load_checkpoint = run_execution_module.load_checkpoint
        after_checkpoint_read = threading.Barrier(2)

        def synchronized_load_checkpoint(*args, **kwargs):
            checkpoint = original_load_checkpoint(*args, **kwargs)
            after_checkpoint_read.wait(timeout=5)
            return checkpoint

        monkeypatch.setattr(
            run_execution_module, "load_checkpoint", synchronized_load_checkpoint
        )

        def concurrent_resume():
            return asyncio.run(
                workflow_mcp.call_tool("workflow.run_resume", resume_arguments)
            )

        results = await asyncio.gather(
            asyncio.to_thread(concurrent_resume),
            asyncio.to_thread(concurrent_resume),
            return_exceptions=True,
        )
        monkeypatch.setattr(
            run_execution_module, "load_checkpoint", original_load_checkpoint
        )
        successful_results = [
            result for result in results if not isinstance(result, BaseException)
        ]
        rejected_results = [
            result for result in results if isinstance(result, BaseException)
        ]
        assert len(successful_results) == 1, results
        assert len(rejected_results) == 1, results
        assert isinstance(rejected_results[0], WorkflowRunOwnershipError), rejected_results
        assert finish_calls == [True]
        checkpoint_count = len(
            conversation_engine.read.get_nodes(
                where={
                    "$and": [
                        {"entity_type": "workflow_checkpoint"},
                        {"run_id": run_id},
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="not found in suspended runtime state"):
            await workflow_mcp.call_tool("workflow.run_resume", resume_arguments)

    payload = _structured(successful_results[0])
    assert successful_results[0].is_error is False
    assert payload["workflow_status"] == "succeeded"
    assert payload["final_state"]["started"] is True
    assert payload["final_state"]["approved"] is True
    assert payload["final_state"]["finished"] is True
    assert len(
        conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_checkpoint"},
                    {"run_id": run_id},
                ]
            }
        )
    ) == checkpoint_count


@pytest.mark.asyncio
async def test_mcp_workflow_run_replay_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, int]] = []

    def replay_run(run_id: str, target_step_seq: int) -> dict:
        calls.append((run_id, target_step_seq))
        return contract["result"]

    monkeypatch.setattr(service, "replay_run", replay_run)
    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [(contract["arguments"]["run_id"], 3)]


@pytest.mark.asyncio
async def test_mcp_workflow_run_cancel_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-cancel-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def cancel_run(run_id: str) -> dict:
        calls.append(run_id)
        return contract["result"]

    monkeypatch.setattr(service, "cancel_run", cancel_run)
    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert calls == [contract["arguments"]["run_id"]]


@pytest.mark.asyncio
async def test_mcp_workflow_run_steps_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-steps-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    run_id = contract["arguments"]["run_id"]
    monkeypatch.setattr(
        service,
        "get_run",
        lambda requested_id: {"run_id": requested_id, "workflow_id": "workflow-1"},
    )
    monkeypatch.setattr(
        service,
        "list_steps",
        lambda requested_id: contract["result"]["steps"]
        if requested_id == run_id
        else [],
    )

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_run_submit_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-submit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def submit_workflow_run(**kwargs):
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service, "submit_workflow_run", submit_workflow_run)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_run_submit_uses_authenticated_user_over_argument(
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

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "submit_workflow_run",
        lambda **kwargs: calls.append(kwargs) or {"run_id": "run-auth-owner"},
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

    with _claims(
        "rw",
        "workflow",
        scope_parts={"user_id": "authenticated-owner"},
    ):
        result = await workflow_mcp.call_tool(
            "workflow.run_submit",
            {
                "workflow_id": "wf-auth-owner",
                "conversation_id": "conv-auth-owner",
                "user_id": "caller-controlled-owner",
            },
        )

    assert result.is_error is False
    assert len(calls) == 1
    assert calls[0]["user_id"] == "authenticated-owner"


@pytest.mark.asyncio
async def test_mcp_workflow_run_submit_denies_workflow_acl_before_admission(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException
    from kogwistar.server.auth_middleware import (
        NameSpace,
        Role,
        get_current_subject,
        get_current_user_id,
    )
    from kogwistar.server.chat_mcp import build_workflow_mcp
    from kogwistar.server.mcp_tools import require_ns, tool_roles

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    submissions: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "submit_workflow_run",
        lambda **kwargs: submissions.append(kwargs) or {"run_id": "unexpected"},
    )
    access_checks: list[tuple[str, str]] = []

    def deny_workflow_access(workflow_id: str, role: str) -> None:
        access_checks.append((workflow_id, role))
        raise HTTPException(status_code=403, detail="workflow denied")

    workflow_mcp = build_workflow_mcp(
        get_service=lambda: service,
        tool_roles=tool_roles,
        require_ns=require_ns,
        role_ro=Role.RO,
        role_rw=Role.RW,
        ns_workflow=NameSpace.WORKFLOW,
        get_subject=get_current_subject,
        get_user_id=get_current_user_id,
        require_workflow_access=deny_workflow_access,
    )

    with _claims("rw", "workflow", sub="alice"):
        with pytest.raises(HTTPException) as denied:
            await workflow_mcp._execute(
                "workflow.run_submit",
                {
                    "workflow_id": "acl-denied-workflow",
                    "conversation_id": "acl-denied-conversation",
                },
                enforce_visibility=True,
            )

    assert denied.value.status_code == 403
    assert access_checks == [("acl-denied-workflow", "rw")]
    assert submissions == []
    assert (
        service.run_registry.meta_store.list_server_runs(
            workflow_id="acl-denied-workflow",
            conversation_id="acl-denied-conversation",
            limit=20,
        )
        == []
    )
