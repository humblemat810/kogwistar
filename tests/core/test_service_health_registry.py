from __future__ import annotations

from kogwistar.engine_core import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.service_health import SERVICE_HEALTH_PROJECTION_NAMESPACE


def _engine(tmp_path):
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
    )


def test_service_health_declare_writes_sparse_truth_and_latest_state(tmp_path):
    engine = _engine(tmp_path)

    payload = engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
        operator_tags=["maintenance"],
    )

    assert payload["service_id"] == "svc.demo"
    assert payload["operator_tags"] == ["maintenance"]
    row = engine.meta_sqlite.get_named_projection(
        SERVICE_HEALTH_PROJECTION_NAMESPACE,
        "demo|ws:demo:ops|svc.demo",
    )
    assert row is not None
    assert row["payload"]["llm_assisted"] is True
    events = engine.read.get_nodes(where={"entity_type": "service_health_event"})
    assert [node.metadata["service_event_type"] for node in events] == [
        "service.registered"
    ]
    assert engine.meta_sqlite.list_named_projections("service_registry") == []
    assert not engine.read.get_nodes(where={"entity_type": "service_definition"})
    assert not engine.read.get_nodes(where={"entity_type": "service_event"})


def test_service_health_repeated_declare_converges_latest_state(tmp_path):
    engine = _engine(tmp_path)
    kwargs = dict(
        service_id="svc.demo",
        service_kind="projection_daemon",
        owner_app="demo-app",
        deterministic=True,
        llm_assisted=False,
        workspace_id="demo",
        namespace="ws:demo:ops",
        operator_tags=["projection"],
    )

    engine.service_health.declare_service(**kwargs)
    engine.service_health.declare_service(**kwargs)

    rows = engine.meta_sqlite.list_named_projections(SERVICE_HEALTH_PROJECTION_NAMESPACE)
    assert [row["key"] for row in rows] == ["demo|ws:demo:ops|svc.demo"]
    events = engine.read.get_nodes(where={"entity_type": "service_health_event"})
    assert [node.metadata["service_event_type"] for node in events] == [
        "service.registered"
    ]


def test_service_health_heartbeat_updates_latest_state_without_graph_spam(tmp_path):
    engine = _engine(tmp_path)
    engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    engine.service_health.start_instance(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
    )
    before_events = engine.read.get_nodes(where={"entity_type": "service_health_event"})

    engine.service_health.heartbeat(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        status="healthy",
    )
    engine.service_health.heartbeat(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        status="healthy",
    )

    payload = engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    assert payload is not None
    assert payload["instance_id"] == "inst-1"
    assert payload["last_seen_ms"] is not None
    assert payload["status"] == "healthy"
    after_events = engine.read.get_nodes(where={"entity_type": "service_health_event"})
    assert len(after_events) == len(before_events)


def test_service_health_recovery_reports_stale_without_mutating_state(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
        heartbeat_ttl_ms=10,
    )
    engine.service_health.start_instance(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        started_at_ms=1,
    )
    payload = engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    assert payload is not None
    payload["last_seen_ms"] = 1
    engine.meta_sqlite.replace_named_projection(
        SERVICE_HEALTH_PROJECTION_NAMESPACE,
        "demo|ws:demo:ops|svc.demo",
        payload,
        last_authoritative_seq=0,
        last_materialized_seq=0,
        projection_schema_version=1,
        materialization_status="ready",
    )

    report = engine.recovery.inspect(
        workspace_id="demo",
        namespaces=["ws:demo:ops"],
    )

    assert any(item.daemon_id == "svc.demo" and item.observed_state == "stale" for item in report.daemon_health)
    assert any(finding.surface == "service_health" for finding in report.findings)
    unchanged = engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    assert unchanged is not None
    assert unchanged["status"] == "starting"


def test_service_health_same_service_id_is_scoped_by_workspace_and_namespace(tmp_path):
    engine = _engine(tmp_path)

    engine.service_health.declare_service(
        service_id="svc.shared",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo-a",
        namespace="ws:demo-a:ops",
    )
    engine.service_health.declare_service(
        service_id="svc.shared",
        service_kind="projection_daemon",
        owner_app="demo-app",
        deterministic=True,
        llm_assisted=False,
        workspace_id="demo-b",
        namespace="ws:demo-b:ops",
    )

    rows = engine.meta_sqlite.list_named_projections(SERVICE_HEALTH_PROJECTION_NAMESPACE)
    assert {row["key"] for row in rows} == {
        "demo-a|ws:demo-a:ops|svc.shared",
        "demo-b|ws:demo-b:ops|svc.shared",
    }
    demo_a = engine.service_health.get_service(
        "svc.shared",
        workspace_id="demo-a",
        namespace="ws:demo-a:ops",
    )
    demo_b = engine.service_health.get_service(
        "svc.shared",
        workspace_id="demo-b",
        namespace="ws:demo-b:ops",
    )
    assert demo_a is not None and demo_a["service_kind"] == "maintenance_daemon"
    assert demo_b is not None and demo_b["service_kind"] == "projection_daemon"
    assert [row["workspace_id"] for row in engine.service_health.list_services(workspace_id="demo-a")] == ["demo-a"]


def test_service_health_repair_rebuilds_missing_projection_from_sparse_truth(tmp_path):
    engine = _engine(tmp_path)
    engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    engine.service_health.start_instance(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        started_at_ms=35,
    )
    engine.service_health.heartbeat(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        status="degraded",
        last_error="boom",
    )
    engine.meta_sqlite.clear_named_projection(
        SERVICE_HEALTH_PROJECTION_NAMESPACE,
        "demo|ws:demo:ops|svc.demo",
    )

    result = engine.service_health.repair_projection(
        workspace_id="demo",
        namespace="ws:demo:ops",
    )

    repaired = engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    assert result.repaired_count == 1
    assert repaired is not None
    assert repaired["status"] == "degraded"
    assert repaired["last_error"] == "boom"
    assert int(repaired["last_seen_ms"]) >= 123


def test_service_health_repair_uses_latest_persisted_lifecycle_timestamp(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    timeline = iter([10, 20, 30, 40, 50, 60, 70])
    monkeypatch.setattr(
        "kogwistar.engine_core.service_health._now_ms",
        lambda: next(timeline),
    )

    engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    engine.service_health.start_instance(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        started_at_ms=35,
    )
    engine.service_health.heartbeat(
        service_id="svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
        instance_id="inst-1",
        status="degraded",
        last_error="boom",
    )
    engine.meta_sqlite.clear_named_projection(
        SERVICE_HEALTH_PROJECTION_NAMESPACE,
        "demo|ws:demo:ops|svc.demo",
    )

    repaired = engine.service_health.repair_projection(
        workspace_id="demo",
        namespace="ws:demo:ops",
    )
    payload = engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    )

    assert repaired.repaired_count == 1
    assert payload is not None
    assert payload["status"] == "degraded"
    assert payload["last_error"] == "boom"
    assert payload["started_at_ms"] == 35
    assert payload["last_seen_ms"] == 70


def test_service_health_repair_keeps_last_seen_ms_monotonic_across_replayed_events(tmp_path):
    engine = _engine(tmp_path)
    engine.service_health.declare_service(
        service_id="svc.demo",
        service_kind="maintenance_daemon",
        owner_app="demo-app",
        deterministic=False,
        llm_assisted=True,
        workspace_id="demo",
        namespace="ws:demo:ops",
    )

    payload: dict[str, object] = {}
    engine.service_health._apply_event_payload(
        payload,
        event_type="service.instance_started",
        payload={"instance_id": "inst-1", "started_at_ms": 500, "status": "starting"},
        event_ts_ms=300,
    )
    engine.service_health._apply_event_payload(
        payload,
        event_type="service.degraded",
        payload={"instance_id": "inst-1", "status": "degraded"},
        event_ts_ms=200,
    )
    engine.service_health._apply_event_payload(
        payload,
        event_type="service.error_changed",
        payload={"instance_id": "inst-1", "last_error": "boom"},
        event_ts_ms=150,
    )

    assert payload["status"] == "degraded"
    assert payload["last_error"] == "boom"
    assert payload["last_seen_ms"] == 500
