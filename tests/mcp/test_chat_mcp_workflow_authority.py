"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_workflow_tool_audit_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-tool-audit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str | None, int]] = []

    def list_tool_audit(*, conversation_id, limit):
        calls.append((conversation_id, limit))
        return contract["result"]

    monkeypatch.setattr(service, "list_tool_audit", list_tool_audit)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["conversation_id"],
            contract["python_service_arguments"]["limit"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_tool_audit_denies_before_reads(monkeypatch, engine_triplet):
    from types import SimpleNamespace

    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-tool-audit-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    reads: list[dict[str, object]] = []
    read_engine = SimpleNamespace(
        read=SimpleNamespace(
            get_nodes=lambda *, where, limit: reads.append(
                {"where": where, "limit": limit}
            )
            or []
        )
    )
    monkeypatch.setattr(service, "_conversation_engine", lambda: read_engine)

    for role, namespace, capabilities in (
        ("ro", "conversation", ["project_view", "read_graph"]),
        ("ro", "workflow", ["read_graph"]),
        ("rw", "workflow", ["project_view"]),
    ):
        try:
            with _claims(
                role,
                namespace,
                scope_parts={"capabilities": capabilities},
            ):
                await server.mcp._execute(
                    contract["tool"], contract["arguments"], enforce_visibility=True
                )
        except (HTTPException, PermissionError):
            pass
        else:
            pytest.fail(
                f"unauthorized tool-audit call was allowed: {role=} {namespace=} {capabilities=}"
            )
    assert reads == []

    with _claims(
        "ro",
        "workflow",
        scope_parts={"capabilities": ["project_view", "read_graph"]},
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert len(reads) == 2
    assert all(
        call["limit"] == contract["python_service_arguments"]["limit"]
        for call in reads
    )
    assert _structured(result) == {
        "conversation_id": contract["python_service_arguments"]["conversation_id"],
        "items": [],
    }


@pytest.mark.asyncio
async def test_mcp_workflow_budget_history_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-budget-history-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def budget_history(*, limit):
        calls.append(limit)
        return contract["result"]

    monkeypatch.setattr(service, "budget_history", budget_history)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_visibility_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-visibility-snapshot-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[None] = []

    def visibility_snapshot():
        calls.append(None)
        return contract["result"]

    monkeypatch.setattr(service, "visibility_snapshot", visibility_snapshot)

    with _claims(
        "ro",
        "workflow",
        sub=contract["result"]["current_subject"],
        scope_parts={"capabilities": contract["result"]["current_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [None]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_visibility_snapshot_denies_missing_capability_and_namespace(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    scope_reads: list[None] = []
    original_scope_snapshot = service._scope_snapshot

    def track_scope_snapshot():
        scope_reads.append(None)
        return original_scope_snapshot()

    monkeypatch.setattr(service, "_scope_snapshot", track_scope_snapshot)

    with _claims("ro", "workflow", sub="visibility-no-cap", scope_parts={"capabilities": ["project_view"]}):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "workflow.visibility_snapshot", {}, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="visibility-wrong-namespace",
        scope_parts={"capabilities": ["project_view", "read_security_scope"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.visibility_snapshot", {}, enforce_visibility=True
            )

    assert scope_reads == []


@pytest.mark.asyncio
async def test_mcp_workflow_capabilities_snapshot_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capabilities-snapshot-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    snapshot_calls: list[None] = []

    def capability_kernel_snapshot():
        snapshot_calls.append(None)
        return {
            field: contract["result"][field]
            for field in ("specs", "approvals", "revoked", "audit_log")
        }

    monkeypatch.setattr(service.capability_kernel, "snapshot", capability_kernel_snapshot)

    with _claims(
        "ro",
        "workflow",
        sub=contract["result"]["current_subject"],
        scope_parts={"capabilities": contract["result"]["effective_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert snapshot_calls == [None]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_capabilities_snapshot_denies_before_snapshot(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    snapshot_calls: list[None] = []
    original_snapshot = service.capability_kernel.snapshot

    def track_snapshot():
        snapshot_calls.append(None)
        return original_snapshot()

    monkeypatch.setattr(service.capability_kernel, "snapshot", track_snapshot)
    for capabilities in (["project_view"], ["read_security_scope"]):
        with _claims(
            "ro",
            "workflow",
            sub=f"capabilities-snapshot-denied-{len(capabilities)}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as denied:
                await server.mcp._execute(
                    "workflow.capabilities_snapshot", {}, enforce_visibility=True
                )
            assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="capabilities-snapshot-wrong-namespace",
        scope_parts={"capabilities": ["project_view", "read_security_scope"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "workflow.capabilities_snapshot", {}, enforce_visibility=True
            )

    assert snapshot_calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_capability_approve_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-approve-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "rw",
        "workflow",
        sub=contract["expected"]["current_subject"],
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    payload = _structured(result)
    for field in contract["expected"]["required_fields"]:
        assert field in payload
    assert payload["current_subject"] == contract["expected"]["current_subject"]
    assert contract["expected"]["approval"] in payload["approvals"]
    assert set(contract["expected"]["effective_capabilities_include"]).issubset(
        payload["effective_capabilities"]
    )
    audit = payload["audit_log"][-2:]
    assert [entry["action"] for entry in audit] == contract["expected"]["audit_actions"]
    assert [entry["required"] for entry in audit] == contract["expected"]["audit_required"]
    assert all(entry["outcome"] == "allow" for entry in audit)


@pytest.mark.asyncio
async def test_mcp_workflow_capability_approve_denies_without_mutation(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-approve-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    prior_approvals = {
        key: set(capabilities)
        for key, capabilities in service.capability_kernel.approvals.items()
    }
    mutation_calls: list[dict[str, Any]] = []
    original_grant = service.capability_kernel.grant

    def track_grant(**kwargs: Any) -> None:
        mutation_calls.append(kwargs)
        original_grant(**kwargs)

    monkeypatch.setattr(service.capability_kernel, "grant", track_grant)
    for index, capabilities in enumerate(
        (
            ["project_view"],
            ["approve_action"],
            ["approve_action", "project_view"],
            ["approve_action", "read_security_scope"],
        )
    ):
        with _claims(
            "rw",
            "workflow",
            sub=f"approval-missing-capability-{index}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as missing_capability:
                await server.mcp._execute(
                    fixture["tool"], fixture["arguments"], enforce_visibility=True
                )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="approval-read-only-role",
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                fixture["tool"], fixture["arguments"], enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="approval-wrong-namespace",
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                fixture["tool"], fixture["arguments"], enforce_visibility=True
            )

    assert mutation_calls == []
    assert service.capability_kernel.approvals == prior_approvals


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-revoke-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    service.capability_kernel.grant(
        subject=contract["expected"]["current_subject"],
        action="seed",
        capabilities="spawn_process",
    )

    with _claims(
        "rw",
        "workflow",
        sub=contract["expected"]["current_subject"],
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    payload = _structured(result)
    for field in contract["expected"]["required_fields"]:
        assert field in payload
    assert payload["current_subject"] == contract["expected"]["current_subject"]
    assert contract["expected"]["revocation"] in payload["revoked"]
    assert set(contract["expected"]["effective_capabilities_exclude"]).isdisjoint(
        payload["effective_capabilities"]
    )
    audit = payload["audit_log"][-2:]
    assert [entry["action"] for entry in audit] == contract["expected"]["audit_actions"]
    assert [entry["required"] for entry in audit] == contract["expected"]["audit_required"]
    assert all(entry["outcome"] == "allow" for entry in audit)


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_denies_without_mutation(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-capability-revoke-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    prior_revoked = {
        subject: set(capabilities)
        for subject, capabilities in service.capability_kernel.revoked.items()
    }
    mutation_calls: list[dict[str, Any]] = []
    original_revoke = service.capability_kernel.revoke

    def track_revoke(**kwargs: Any) -> None:
        mutation_calls.append(kwargs)
        return original_revoke(**kwargs)

    monkeypatch.setattr(service.capability_kernel, "revoke", track_revoke)
    for index, capabilities in enumerate(
        (
            ["project_view"],
            ["approve_action"],
            ["approve_action", "project_view"],
            ["approve_action", "read_security_scope"],
        )
    ):
        with _claims(
            "rw",
            "workflow",
            sub=f"revoke-missing-capability-{index}",
            scope_parts={"capabilities": capabilities},
        ):
            with pytest.raises(HTTPException) as denied:
                await server.mcp._execute(
                    contract["tool"], contract["arguments"], enforce_visibility=True
                )
        assert denied.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="revoke-read-only-role",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="revoke-wrong-namespace",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )

    assert mutation_calls == []
    assert service.capability_kernel.revoked == prior_revoked


@pytest.mark.asyncio
async def test_mcp_workflow_capability_revoke_can_revoke_own_snapshot_access(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    subject = "self-revoking-admin"

    with _claims(
        "rw",
        "workflow",
        sub=subject,
        scope_parts={
            "capabilities": [
                "approve_action",
                "project_view",
                "read_security_scope",
            ]
        },
    ):
        result = await server.mcp.call_tool(
            "workflow.capability_revoke",
            {"capability": "project_view", "subject": subject},
        )

    assert result.is_error is False
    payload = _structured(result)
    assert {"subject": subject, "capabilities": ["project_view"]} in payload["revoked"]
    assert "project_view" not in payload["effective_capabilities"]
    assert [entry["action"] for entry in payload["audit_log"][-2:]] == [
        "approve_action",
        "project_view",
    ]
    assert all(entry["outcome"] == "allow" for entry in payload["audit_log"][-2:])


@pytest.mark.asyncio
async def test_mcp_workflow_service_trigger_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-trigger-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def trigger_service(service_id, *, trigger_type, payload):
        calls.append((service_id, trigger_type, payload))
        return contract["result"]

    monkeypatch.setattr(service, "trigger_service", trigger_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["trigger_type"],
            contract["python_service_arguments"]["payload"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_trigger_denies_unauthorized_effects(
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
            / "contracts/golden/workflow-service-trigger-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    effects: list[tuple[str, str, dict[str, object]]] = []

    def trigger_service(
        service_id,
        *,
        trigger_type,
        payload,
        authenticated_user_id=None,
        enforce_authenticated_owner=False,
    ):
        effects.append((service_id, trigger_type, payload))
        return contract["result"]

    monkeypatch.setattr(service.service_supervisor, "trigger_service", trigger_service)

    for role, namespace, capabilities in (
        ("ro", "workflow", ["service.manage"]),
        ("rw", "conversation", ["service.manage"]),
        ("rw", "workflow", []),
    ):
        with _claims(
            role,
            namespace,
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert effects == []

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert effects == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["trigger_type"],
            contract["python_service_arguments"]["payload"],
        )
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letter_replay_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letter-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims(
        "rw",
        "workflow",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letter_replay_dispatches_effect_once(
    monkeypatch, engine_triplet
):
    import asyncio
    import threading

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "dead-letter-mcp-replay"
    attempts = 0
    replay_completed = threading.Event()

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("initial attempt fails")
        replay_completed.set()

    admission = service.scheduler.submit(
        run_id=run_id,
        priority_class="foreground",
        start_fn=fail_once,
        max_retries=0,
    )
    assert admission["admission"] == "accepted"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not service.scheduler.dead_letter():
        time.sleep(0.01)
    dead_rows = [row for row in service.scheduler.dead_letter() if row["run_id"] == run_id]
    assert dead_rows, "initial failed attempt did not enter dead-letter queue"
    dead = dead_rows[0]
    assert attempts == 1

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        results = await asyncio.gather(
            *(
                server.mcp.call_tool(
                    "workflow.dead_letter_replay", {"run_id": run_id}
                )
                for _ in range(2)
            )
        )
        responses = [_structured(result) for result in results]

    outcomes = sorted(response["replayed"] for response in responses)
    assert outcomes == [False, True]
    assert all(response["run_id"] == run_id for response in responses)
    assert next(response for response in responses if response["replayed"]) == {
        "run_id": run_id,
        "replayed": True,
        "dead_letter": dead,
    }
    assert replay_completed.wait(timeout=5.0)
    assert attempts == 2
    assert all(row["run_id"] != run_id for row in service.scheduler.dead_letter())


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letter_replay_denies_without_dispatch(
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
            / "contracts/golden/workflow-dead-letter-replay-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "replay_dead_letter",
        lambda run_id: scheduler_calls.append(run_id),
    )

    with _claims(
        "rw",
        "workflow",
        scope_parts={"capabilities": []},
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )

    assert denied.value.status_code == 403
    assert scheduler_calls == []


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letters_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letters-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "dead_letter",
        lambda: scheduler_calls.append("read") or [],
    )

    with _claims(
        "ro",
        "workflow",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert scheduler_calls == ["read"]


@pytest.mark.asyncio
async def test_mcp_workflow_dead_letters_denies_before_read(monkeypatch, engine_triplet):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letters-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    scheduler_calls: list[str] = []
    monkeypatch.setattr(
        service.scheduler,
        "dead_letter",
        lambda: scheduler_calls.append("read") or [],
    )

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "rw",
        "conversation",
        scope_parts={"capabilities": contract["caller_capabilities"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert scheduler_calls == []

    assert scheduler_calls == []
