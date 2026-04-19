from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import pytest
from fastapi.testclient import TestClient

import kogwistar.server_mcp_with_admin as server
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.server.auth_middleware import claims_ctx
import kogwistar.server.service_daemon as service_daemon
from kogwistar.server.service_daemon import SERVICE_PROJECTION_NAMESPACE
from tests.server.test_chat_server_api import (
    FakeEmbeddingFunction,
    _configure_server,
    _runtime_success_runner,
    _token_header,
)
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.server


@pytest.fixture()
def engine_triplet():
    root = Path(".tmp_service_daemon_tests") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        yield (
            GraphKnowledgeEngine(
                persist_directory=str(root / "kg"),
                kg_graph_type="knowledge",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "conversation"),
                kg_graph_type="conversation",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "workflow"),
                kg_graph_type="workflow",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            ),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _service_claims():
    token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "storage_ns": "default",
            "execution_ns": "default",
            "security_scope": "tenant/demo",
            "capabilities": [
                "service.manage",
                "service.inspect",
                "service.heartbeat",
                "project_view",
                "spawn_process",
                "workflow.run.read",
                "workflow.run.write",
                "workflow.design.write",
                "workflow.design.inspect",
                "read_graph",
                "write_graph",
            ],
            "sub": "svc-tester",
        }
    )
    try:
        yield
    finally:
        claims_ctx.reset(token)


def _create_conversation(service) -> dict[str, str]:
    with _service_claims():
        return service.create_conversation(user_id="svc-user")


def _build_minimal_workflow(service, *, workflow_id: str, designer_id: str = "svc-tester") -> None:
    with _service_claims():
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


def _attach_fake_service_spawn(service, *, statuses: list[str] | None = None) -> None:
    plan = list(statuses or ["succeeded"])
    state = {"n": 0}

    def _spawn_workflow_run(**kwargs):
        idx = min(state["n"], len(plan) - 1)
        status = str(plan[idx] or "succeeded")
        state["n"] += 1
        run_id = f"svc-run-{uuid.uuid4().hex}"
        service.run_registry.create_run(
            run_id=run_id,
            conversation_id=str(kwargs.get("conversation_id") or "svc-conv"),
            workflow_id=str(kwargs.get("workflow_id") or "wf.service"),
            user_id=str(kwargs.get("user_id") or "svc-user"),
            user_turn_node_id=str(kwargs.get("turn_node_id") or "svc-turn"),
            status="running" if status == "running" else status,
        )
        service.run_registry.update_status(
            run_id,
            status=status,
            started=True,
            finished=True,
            result={"workflow_status": status} if status == "succeeded" else None,
            error={"message": "boom"} if status == "failed" else None,
        )
        return {
            "run_id": run_id,
            "workflow_id": str(kwargs.get("workflow_id") or "wf.service"),
            "conversation_id": str(kwargs.get("conversation_id") or "svc-conv"),
            "status": status,
        }

    service.service_supervisor.spawn_workflow_run = _spawn_workflow_run


def _wait_for_terminal(service, run_id: str) -> dict:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        with _service_claims():
            run = service.get_run(run_id)
        if run["terminal"]:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach terminal state")


def _replace_projection(service, service_id: str, payload: dict) -> None:
    workflow_engine = service._workflow_engine()
    latest_seq = workflow_engine.meta_sqlite.get_latest_entity_event_seq(
        namespace=getattr(workflow_engine, "namespace", "default")
    )
    workflow_engine.meta_sqlite.replace_named_projection(
        SERVICE_PROJECTION_NAMESPACE,
        service_id,
        payload,
        last_authoritative_seq=latest_seq,
        last_materialized_seq=latest_seq,
        projection_schema_version=1,
        materialization_status="ready",
    )


def _service_projection(service, service_id: str) -> dict:
    projection = service._workflow_engine().meta_sqlite.get_named_projection(
        SERVICE_PROJECTION_NAMESPACE, service_id
    )
    if projection is None:
        raise AssertionError(f"missing projection for {service_id}")
    return dict(projection["payload"])


def test_service_declaration_health_disable_and_projection_rebuild(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    base_ms = 1_000_000
    monkeypatch.setattr(service_daemon, "_now_ms", lambda: base_ms)
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.health")
    _attach_fake_service_spawn(service, statuses=["succeeded"])

    with _service_claims():
        declared = service.declare_service(
            service_id="svc.health.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.health",
            target_config={"conversation_id": created["conversation_id"]},
            enabled=True,
            heartbeat_ttl_ms=60_000,
        )
        assert declared["service_id"] == "svc.health.demo"

        defs = workflow_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "service_definition"},
                    {"service_id": "svc.health.demo"},
                ]
            }
        )
        assert defs

        healthy = service.record_service_heartbeat(
            "svc.health.demo", instance_id="inst-1", payload={"beat": 1}
        )
        assert healthy["health_status"] == "healthy"

        monkeypatch.setattr(service_daemon, "_now_ms", lambda: base_ms + 120_000)
        service.service_supervisor.tick()
        degraded = _service_projection(service, "svc.health.demo")
        assert degraded["health_status"] == "degraded"

        disabled = service.disable_service("svc.health.demo")
        assert disabled["lifecycle_status"] == "stopped"
        assert disabled["health_status"] == "stopped"

        workflow_engine.meta_sqlite.clear_named_projection(
            SERVICE_PROJECTION_NAMESPACE, "svc.health.demo"
        )
        service.service_supervisor._rebuild_projection("svc.health.demo")
        rebuilt = _service_projection(service, "svc.health.demo")
        assert rebuilt["service_id"] == "svc.health.demo"
        assert rebuilt["enabled"] is False


def test_service_restart_policy_respects_backoff_and_max_restarts(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    base_ms = 2_000_000
    monkeypatch.setattr(service_daemon, "_now_ms", lambda: base_ms)

    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.restart")
    _attach_fake_service_spawn(service, statuses=["failed", "succeeded"])

    with _service_claims():
        service.declare_service(
            service_id="svc.restart.demo",
            service_kind="worker",
            target_kind="workflow",
            target_ref="wf.service.restart",
            target_config={"conversation_id": created["conversation_id"]},
            restart_policy={
                "mode": "on_failure",
                "max_restarts": 1,
                "restart_backoff_ms": 30,
            },
        )
        first = service.trigger_service(
            "svc.restart.demo",
            trigger_type="external event",
            payload={},
        )
        first_run_id = str(first["current_child_run_id"] or "")
        assert first_run_id
    with _service_claims():
        service.service_supervisor.tick()
        waiting = _service_projection(service, "svc.restart.demo")
        assert waiting["lifecycle_status"] == "restarting"
        assert waiting["restart_count"] == 1
        assert str(waiting["current_child_run_id"] or "") == first_run_id

        monkeypatch.setattr(service_daemon, "_now_ms", lambda: base_ms + 40)
        service.service_supervisor.tick()
        restarted = _service_projection(service, "svc.restart.demo")
        second_run_id = str(restarted["current_child_run_id"] or "")
        assert second_run_id
        assert second_run_id != first_run_id

    with _service_claims():
        monkeypatch.setattr(service_daemon, "_now_ms", lambda: base_ms + 100)
        service.service_supervisor.tick()
        stable = _service_projection(service, "svc.restart.demo")
        assert str(stable["current_child_run_id"] or "") == second_run_id
        assert stable["restart_count"] == 1


def test_service_schedule_autostart_and_due_window(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.schedule")
    _attach_fake_service_spawn(service, statuses=["succeeded", "succeeded"])

    with _service_claims():
        service.declare_service(
            service_id="svc.schedule.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.schedule",
            target_config={"conversation_id": created["conversation_id"]},
            enabled=True,
            autostart=True,
            trigger_specs=[
                {
                    "type": "schedule",
                    "enabled": True,
                    "selector": {"interval_ms": 10_000},
                    "cooldown_ms": 10_000,
                }
            ],
        )
        service.service_supervisor.bootstrap()
        first = _service_projection(service, "svc.schedule.demo")
        first_run_id = str(first["current_child_run_id"] or "")
        assert first["last_trigger_type"] == "autostart"
        assert first_run_id

    _wait_for_terminal(service, first_run_id)

    with _service_claims():
        service.service_supervisor.tick()
        still_first = _service_projection(service, "svc.schedule.demo")
        assert str(still_first["current_child_run_id"] or "") == first_run_id

        projection = workflow_engine.meta_sqlite.get_named_projection(
            SERVICE_PROJECTION_NAMESPACE, "svc.schedule.demo"
        )
        payload = dict(projection["payload"])
        payload["next_due_at_ms"] = int(time.time() * 1000) - 1
        payload["last_triggered_at_ms"] = 0
        _replace_projection(service, "svc.schedule.demo", payload)
        service.service_supervisor.tick()
        second = _service_projection(service, "svc.schedule.demo")
        second_run_id = str(second["current_child_run_id"] or "")
        assert second["last_trigger_type"] == "schedule"
        assert second_run_id and second_run_id != first_run_id


def test_service_message_graph_and_external_triggers(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.msg")
    _build_minimal_workflow(service, workflow_id="wf.service.graph")
    _build_minimal_workflow(service, workflow_id="wf.service.external")
    _attach_fake_service_spawn(service, statuses=["succeeded", "succeeded", "succeeded"])

    with _service_claims():
        service.declare_service(
            service_id="svc.msg.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.msg",
            target_config={"conversation_id": created["conversation_id"]},
            trigger_specs=[
                {
                    "type": "message arrival",
                    "enabled": True,
                    "selector": {"inbox_id": "inbox:svc:demo"},
                }
            ],
        )
    conversation_engine.meta_sqlite.project_lane_message(
        message_id="msg-service-demo",
        namespace="default",
        inbox_id="inbox:svc:demo",
        conversation_id=created["conversation_id"],
        recipient_id="svc",
        sender_id="user",
        msg_type="request.demo",
        status="pending",
        created_at=int(time.time()),
        available_at=int(time.time()),
        run_id=None,
        step_id=None,
        correlation_id="corr-service-demo",
        payload_json='{"x":1}',
    )
    with _service_claims():
        service.service_supervisor.tick()
        msg_triggered = _service_projection(service, "svc.msg.demo")
        assert msg_triggered["last_trigger_type"] == "message arrival"
        assert msg_triggered["current_child_run_id"]

        service.declare_service(
            service_id="svc.graph.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.graph",
            target_config={"conversation_id": created["conversation_id"]},
            trigger_specs=[
                {
                    "type": "graph change",
                    "enabled": True,
                    "selector": {"engine": "workflow", "namespace": "default"},
                }
            ],
        )
    workflow_engine.write.add_node(
        Node(
            id="service-graph-node",
            label="service graph node",
            type="entity",
            summary="graph trigger node",
            mentions=[Grounding(spans=[Span.from_dummy_for_workflow("graph")])],
            metadata={"entity_type": "service_test"},
        )
    )
    with _service_claims():
        service.service_supervisor.tick()
        graph_triggered = _service_projection(service, "svc.graph.demo")
        assert graph_triggered["last_trigger_type"] == "graph change"
        assert graph_triggered["current_child_run_id"]

        service.declare_service(
            service_id="svc.external.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.external",
            target_config={"conversation_id": created["conversation_id"]},
        )
        external = service.trigger_service(
            "svc.external.demo",
            trigger_type="external event",
            payload={"source": "test"},
        )
        assert external["last_trigger_type"] == "external event"
        events = service.list_service_events("svc.external.demo")
        assert any(evt["event_type"] == "service.triggered" for evt in events)


def test_service_process_table_shows_service_and_child_run(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.process")
    _attach_fake_service_spawn(service, statuses=["succeeded"])

    with _service_claims():
        service.declare_service(
            service_id="svc.process.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.service.process",
            target_config={"conversation_id": created["conversation_id"]},
        )
        triggered = service.trigger_service(
            "svc.process.demo",
            trigger_type="external event",
            payload={},
        )
        run_id = str(triggered["current_child_run_id"] or "")
        rows = service.list_process_table(limit=20)

    service_row = next(row for row in rows if row["process_kind"] == "service")
    run_row = next(row for row in rows if row["process_kind"] == "workflow_run")
    assert service_row["process_id"] == "svc.process.demo"
    assert service_row["current_child_run_id"] == run_id
    assert run_row["process_id"] == run_id


def test_service_runtime_api_round_trip(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )
    created = _create_conversation(service)
    _build_minimal_workflow(service, workflow_id="wf.service.rest")
    _attach_fake_service_spawn(service, statuses=["succeeded"])

    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            capabilities="service.manage,service.inspect,service.heartbeat,project_view,spawn_process,workflow.run.read,workflow.run.write",
        )
        declared = client.post(
            "/api/workflow/services",
            json={
                "service_id": "svc.rest.demo",
                "service_kind": "daemon",
                "target_kind": "workflow",
                "target_ref": "wf.service.rest",
                "target_config": {"conversation_id": created["conversation_id"]},
                "enabled": True,
            },
            headers=headers,
        )
        declared.raise_for_status()
        assert declared.json()["service_id"] == "svc.rest.demo"

        beat = client.post(
            "/api/workflow/services/svc.rest.demo/heartbeat",
            json={"instance_id": "rest-1", "payload": {"ok": True}},
            headers=headers,
        )
        beat.raise_for_status()
        assert beat.json()["health_status"] == "healthy"

        listed = client.get("/api/workflow/services", headers=headers)
        listed.raise_for_status()
        assert any(
            item["service_id"] == "svc.rest.demo"
            for item in listed.json()["services"]
        )

        triggered = client.post(
            "/api/workflow/services/svc.rest.demo/trigger",
            json={"trigger_type": "external event", "payload": {}},
            headers=headers,
        )
        triggered.raise_for_status()
        run_id = str(triggered.json()["current_child_run_id"] or "")
        assert run_id

        events = client.get(
            "/api/workflow/services/svc.rest.demo/events", headers=headers
        )
        events.raise_for_status()
        names = [evt["event_type"] for evt in events.json()["events"]]
        assert "service.heartbeat" in names
        assert "service.triggered" in names
