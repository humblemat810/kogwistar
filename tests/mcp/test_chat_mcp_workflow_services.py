"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_workflow_scheduler_timeline_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-scheduler-timeline-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def list_scheduler_timeline(**kwargs):
        calls.append(kwargs)
        return contract["result"]["events"]

    monkeypatch.setattr(service, "list_scheduler_timeline", list_scheduler_timeline)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_scheduler_timeline_denies_before_reads(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    timeline_reads: list[dict[str, object]] = []

    def track_timeline(**kwargs: object) -> list[dict[str, object]]:
        timeline_reads.append(kwargs)
        return []

    monkeypatch.setattr(service.scheduler, "timeline", track_timeline)
    arguments = {"run_id": None, "limit": 10}
    with _claims(
        "ro", "workflow", sub="scheduler-no-cap", scope_parts={"capabilities": []}
    ), pytest.raises(HTTPException) as denied:
        await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert denied.value.status_code == 403

    with _claims(
        "ro",
        "conversation",
        sub="scheduler-wrong-scope",
        scope_parts={"capabilities": ["project_view"]},
    ), pytest.raises(PermissionError):
        await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert timeline_reads == []

    with _claims(
        "rw",
        "workflow",
        sub="scheduler-authorized",
        scope_parts={"capabilities": ["project_view"]},
    ):
        result = await server.mcp._execute(
            "workflow.scheduler_timeline", arguments, enforce_visibility=True
        )
    assert timeline_reads == [{"run_id": None, "limit": 10}]
    assert _structured(result) == {"run_id": None, "events": []}


@pytest.mark.asyncio
async def test_mcp_workflow_service_declare_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-declare-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[dict[str, object]] = []

    def declare_service(**kwargs):
        calls.append(kwargs)
        return contract["result"]

    monkeypatch.setattr(service, "declare_service", declare_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_declare_denies_unauthorized_effects(
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
            / "contracts/golden/workflow-service-declare-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        service.service_supervisor,
        "declare_service",
        lambda **kwargs: writes.append(kwargs) or contract["result"],
    )

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
    assert writes == []

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert writes == [contract["python_service_arguments"]]


@pytest.mark.asyncio
async def test_mcp_workflow_service_disable_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-disable-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def disable_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "disable_service", disable_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_enable_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-enable-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def enable_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "enable_service", enable_service)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "method", "fixture_name"),
    [
        (
            "workflow.service_enable",
            "enable_service",
            "workflow-service-enable-mcp-v1.json",
        ),
        (
            "workflow.service_disable",
            "disable_service",
            "workflow-service-disable-mcp-v1.json",
        ),
    ],
)
async def test_mcp_workflow_service_lifecycle_denies_unauthorized_effects(
    monkeypatch, engine_triplet, tool, method, fixture_name
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (Path(__file__).resolve().parents[2] / f"contracts/golden/{fixture_name}").read_text(
            encoding="utf-8"
        )
    )
    effects: list[str] = []
    monkeypatch.setattr(
        service.service_supervisor,
        method,
        lambda service_id: effects.append(service_id) or contract["result"],
    )

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
                tool, contract["arguments"], enforce_visibility=True
            )
    assert effects == []

    with _claims(
        "rw", "workflow", scope_parts={"capabilities": ["service.manage"]}
    ):
        result = await server.mcp._execute(
            tool, contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert effects == [contract["python_service_arguments"]["service_id"]]


@pytest.mark.asyncio
async def test_mcp_workflow_service_events_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-events-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, int]] = []

    def list_service_events(service_id, *, limit):
        calls.append((service_id, limit))
        return contract["result"]["events"]

    monkeypatch.setattr(service, "list_service_events", list_service_events)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["limit"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_events_deny_before_read(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    reads: list[tuple[str, int]] = []

    def list_service_events(service_id: str, *, limit: int) -> list[dict[str, str]]:
        reads.append((service_id, limit))
        return [{"event_id": "evt-1"}]

    monkeypatch.setattr(
        service.service_supervisor, "list_service_events", list_service_events
    )
    arguments = {"service_id": "svc-1", "limit": 5}

    for claims_ns, capabilities in [
        ("workflow", []),
        ("workflow", ["service.inspect"]),
        ("workflow", ["project_view"]),
        ("conversation", ["service.inspect", "project_view"]),
    ]:
        reads_before = len(reads)
        with _claims(
            "ro",
            claims_ns,
            sub=f"service-events-denied-{reads_before}",
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                "workflow.service_events", arguments, enforce_visibility=True
            )
        assert len(reads) == reads_before

    with _claims(
        "ro",
        "workflow",
        sub="service-events-authorized",
        scope_parts={"capabilities": ["service.inspect", "project_view"]},
    ):
        result = await server.mcp._execute(
            "workflow.service_events", arguments, enforce_visibility=True
        )

    assert result.is_error is False
    assert _structured(result) == {
        "service_id": "svc-1",
        "events": [{"event_id": "evt-1"}],
    }
    assert reads == [("svc-1", 5)]


@pytest.mark.asyncio
async def test_mcp_workflow_service_get_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-get-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def get_service(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "get_service", get_service)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_heartbeat_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-heartbeat-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def record_service_heartbeat(service_id, *, instance_id, payload):
        calls.append((service_id, instance_id, payload))
        return contract["result"]

    monkeypatch.setattr(service, "record_service_heartbeat", record_service_heartbeat)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["instance_id"],
            contract["python_service_arguments"]["payload"],
        )
    ]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_heartbeat_denies_unauthorized_effects(
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
            / "contracts/golden/workflow-service-heartbeat-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    effects: list[tuple[str, str, dict[str, object]]] = []
    monkeypatch.setattr(
        service.service_supervisor,
        "record_service_heartbeat",
        lambda service_id, *, instance_id, payload: (
            effects.append((service_id, instance_id, payload)) or contract["result"]
        ),
    )

    for role, namespace, capabilities in (
        ("ro", "workflow", ["service.heartbeat", "service.manage"]),
        ("rw", "conversation", ["service.heartbeat", "service.manage"]),
        ("rw", "workflow", []),
        ("rw", "workflow", ["service.heartbeat"]),
        ("rw", "workflow", ["service.manage"]),
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
        "rw",
        "workflow",
        scope_parts={"capabilities": ["service.heartbeat", "service.manage"]},
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert _structured(result) == contract["result"]
    assert effects == [
        (
            contract["python_service_arguments"]["service_id"],
            contract["python_service_arguments"]["instance_id"],
            contract["python_service_arguments"]["payload"],
        )
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_service_list_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-list-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def list_services(*, limit):
        calls.append(limit)
        return contract["result"]["services"]

    monkeypatch.setattr(service, "list_services", list_services)

    with _claims("ro", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_reads_deny_before_lookup(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    lookups: list[tuple[str, str]] = []

    def get_service(service_id: str) -> dict[str, str]:
        lookups.append(("get", service_id))
        return {"service_id": service_id}

    def list_services(*, limit: int) -> list[dict[str, int]]:
        lookups.append(("list", str(limit)))
        return []

    monkeypatch.setattr(service.service_supervisor, "get_service", get_service)
    monkeypatch.setattr(service.service_supervisor, "list_services", list_services)
    operations = [
        ("workflow.service_get", {"service_id": "svc-1"}),
        ("workflow.service_list", {"limit": 20}),
    ]

    for tool, arguments in operations:
        reads_before = len(lookups)
        with _claims(
            "ro", "workflow", sub=f"{tool}-no-cap", scope_parts={"capabilities": []}
        ), pytest.raises(HTTPException) as denied:
            await server.mcp._execute(tool, arguments, enforce_visibility=True)
        assert denied.value.status_code == 403
        assert len(lookups) == reads_before

        with _claims(
            "ro",
            "conversation",
            sub=f"{tool}-wrong-scope",
            scope_parts={"capabilities": ["service.inspect"]},
        ), pytest.raises(PermissionError):
            await server.mcp._execute(tool, arguments, enforce_visibility=True)
        assert len(lookups) == reads_before

        for capabilities in (["service.inspect"], ["project_view"]):
            with _claims(
                "ro",
                "workflow",
                sub=f"{tool}-missing-{capabilities[0]}",
                scope_parts={"capabilities": capabilities},
            ), pytest.raises(HTTPException) as denied:
                await server.mcp._execute(tool, arguments, enforce_visibility=True)
            assert denied.value.status_code == 403
            assert len(lookups) == reads_before

        with _claims(
            "ro",
            "workflow",
            sub=f"{tool}-authorized",
            scope_parts={"capabilities": ["service.inspect", "project_view"]},
        ):
            result = await server.mcp._execute(
                tool, arguments, enforce_visibility=True
            )
        assert result.is_error is False

    assert lookups == [
        ("get", "svc-1"),
        ("list", "20"),
    ]


@pytest.mark.asyncio
async def test_mcp_workflow_service_repair_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-service-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[str] = []

    def repair_service_projection(service_id):
        calls.append(service_id)
        return contract["result"]

    monkeypatch.setattr(service, "repair_service_projection", repair_service_projection)

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["service_id"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_repair_denies_unauthorized_effects(
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
            / "contracts/golden/workflow-service-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    service_id = contract["python_service_arguments"]["service_id"]
    with _claims("rw", "workflow", scope_parts={"capabilities": ["service.manage"]}):
        service.declare_service(
            service_id=service_id,
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.repair-auth",
            enabled=False,
        )

    repairs: list[str] = []
    monkeypatch.setattr(
        service.service_supervisor,
        "_rebuild_projection",
        lambda sid: repairs.append(sid),
    )
    for role, namespace, capabilities in (
        ("ro", "workflow", ["service.manage"]),
        ("rw", "conversation", ["service.manage"]),
        ("rw", "workflow", ["service.inspect", "project_view"]),
    ):
        with _claims(
            role,
            namespace,
            scope_parts={"capabilities": capabilities},
        ), pytest.raises((HTTPException, PermissionError)):
            await server.mcp._execute(
                contract["tool"], contract["arguments"], enforce_visibility=True
            )
    assert repairs == []

    with _claims(
        "rw",
        "workflow",
        scope_parts={
            "capabilities": ["service.manage", "service.inspect", "project_view"]
        },
    ):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert repairs == [service_id]
    assert _structured(result)["service_id"] == service_id


@pytest.mark.asyncio
async def test_mcp_workflow_services_repair_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-services-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls: list[int] = []

    def repair_service_projections(*, limit):
        calls.append(limit)
        return contract["result"]

    monkeypatch.setattr(
        service, "repair_service_projections", repair_service_projections
    )

    with _claims("rw", "workflow"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert result.is_error is False
    assert calls == [contract["python_service_arguments"]["limit"]]
    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_workflow_services_repair_denies_unauthorized_effects(
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
            / "contracts/golden/workflow-services-repair-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    listed: list[int] = []
    repairs: list[str] = []
    monkeypatch.setattr(
        service.service_supervisor,
        "list_services",
        lambda *, limit: listed.append(limit)
        or [{"service_id": "svc-a"}, {"service_id": "svc-b"}],
    )
    monkeypatch.setattr(
        service.service_supervisor,
        "_rebuild_projection",
        lambda service_id: repairs.append(service_id),
    )

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
    assert listed == []
    assert repairs == []

    with _claims("rw", "workflow", scope_parts={"capabilities": ["service.manage"]}):
        result = await server.mcp._execute(
            contract["tool"], contract["arguments"], enforce_visibility=True
        )
    assert result.is_error is False
    assert listed == [contract["python_service_arguments"]["limit"]]
    assert repairs == ["svc-a", "svc-b"]
    assert _structured(result) == {"repaired_service_ids": repairs}
