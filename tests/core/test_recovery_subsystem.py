from __future__ import annotations

import json

import pytest

from kogwistar.engine_core import (
    GraphKnowledgeEngine,
    OutputReconciliationState,
    RecoverySurface,
    ResumePolicy,
)
from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowCheckpointNode
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.run_registry import RunRegistry


def _engine(tmp_path):
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
        kg_graph_type="conversation",
    )


def _checkpoint(
    *,
    run_id: str,
    step_seq: int,
    status: str | None = None,
    suspended: bool = False,
    restartable: bool = False,
    resume_marker: bool = False,
) -> WorkflowCheckpointNode:
    metadata = {
        "entity_type": "workflow_checkpoint",
        "run_id": run_id,
        "workflow_id": f"wf-{run_id}",
        "conversation_id": f"conv-{run_id}",
        "step_seq": step_seq,
        "checkpoint_schema_version": 1,
        "state_json": json.dumps({"run_id": run_id}),
    }
    if status is not None:
        metadata["status"] = status
    if suspended:
        metadata["suspended"] = True
    if restartable:
        metadata["restartable"] = True
    if resume_marker:
        metadata["resume_marker"] = True
    return WorkflowCheckpointNode(
        id=f"wf_ckpt|{run_id}|{step_seq}",
        label=f"checkpoint {run_id} {step_seq}",
        type="entity",
        doc_id=f"wf_ckpt|{run_id}|{step_seq}",
        summary=f"checkpoint {run_id} {step_seq}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata=metadata,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def test_recovery_repairs_lane_projections_across_namespaces(tmp_path):
    engine = _engine(tmp_path)
    namespaces = ["ws:demo:conv:bg", "ws:demo:conv:fg"]

    token = claims_ctx.set({"storage_ns": namespaces[0]})
    with scoped_namespace(engine, namespaces[0]):
        bg = engine.send_lane_message(
            conversation_id="conv-bg",
            inbox_id="inbox:worker",
            sender_id="lane:fg",
            recipient_id="lane:bg",
            msg_type="request.bg",
            payload={"kind": "bg"},
        )
    claims_ctx.reset(token)
    token = claims_ctx.set({"storage_ns": namespaces[1]})
    with scoped_namespace(engine, namespaces[1]):
        fg = engine.send_lane_message(
            conversation_id="conv-fg",
            inbox_id="inbox:foreground",
            sender_id="lane:bg",
            recipient_id="lane:fg",
            msg_type="reply.fg",
            payload={"kind": "fg"},
        )
    claims_ctx.reset(token)
    assert engine.meta_sqlite.clear_projected_lane_messages(namespaces[0]) == 1
    assert engine.meta_sqlite.clear_projected_lane_messages(namespaces[1]) == 1

    report = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=namespaces,
    )

    assert report.repaired_count == 2
    assert {item.namespace for item in report.repaired_lane_projections} == set(namespaces)
    assert [
        row.message_id
        for row in engine.meta_sqlite.list_projected_lane_messages(namespace=namespaces[0])
    ] == [bg.message_id]
    assert [
        row.message_id
        for row in engine.meta_sqlite.list_projected_lane_messages(namespace=namespaces[1])
    ] == [fg.message_id]


def test_recovery_inspect_reports_queue_lane_dead_letter_and_run_surfaces(tmp_path):
    engine = _engine(tmp_path)
    queue_ns = "ws:demo:maintenance_jobs"
    lane_ns = "ws:demo:conv:bg"
    engine.jobs.enqueue(
        job_id="job-expired",
        namespace=queue_ns,
        entity_kind="maintenance_job",
        entity_id="entity-1",
        job_kind="maintenance_job",
    )
    engine.jobs.enqueue(
        job_id="job-dead",
        namespace=queue_ns,
        entity_kind="maintenance_job",
        entity_id="entity-2",
        job_kind="maintenance_job",
        max_retries=1,
    )
    expired_job = engine.jobs.claim(
        namespace=queue_ns, limit=1, lease_seconds=60
    )[0]
    dead_job = engine.jobs.claim(namespace=queue_ns, limit=1, lease_seconds=1)[0]
    engine.meta_sqlite._state.index_jobs[expired_job.job_id].lease_until = 0
    engine.jobs.retry_or_fail(dead_job, "terminal")

    token = claims_ctx.set({"storage_ns": lane_ns})
    with scoped_namespace(engine, lane_ns):
        msg = engine.send_lane_message(
            conversation_id="conv-lane",
            inbox_id="inbox:worker",
            sender_id="lane:fg",
            recipient_id="lane:bg",
            msg_type="request.work",
            payload={"kind": "work"},
        )
    claims_ctx.reset(token)
    engine.meta_sqlite.claim_projected_lane_messages(
        namespace=lane_ns,
        inbox_id="inbox:worker",
        claimed_by="worker-1",
        limit=1,
        lease_seconds=-1,
    )

    registry = RunRegistry(engine.meta_sqlite)
    registry.create_run(
        run_id="run-1",
        conversation_id=lane_ns,
        workflow_id="wf-run",
        user_id=None,
        user_turn_node_id="turn-1",
        status="running",
    )
    registry.append_event("run-1", "worker.requested", {"message_id": msg.message_id})
    registry.create_run(
        run_id="run-2",
        conversation_id="ws:demo:other",
        workflow_id="wf-run",
        user_id=None,
        user_turn_node_id="turn-2",
        status="running",
    )

    report = engine.recovery.inspect(
        workspace_id="demo",
        namespaces=[queue_ns, lane_ns],
        app_surfaces=[
            OutputReconciliationState(
                surface_id="demo:manifest",
                surface_kind="projection_manifest",
                status="missing",
                drift_detected=True,
            )
        ],
    )

    by_job = {item.job_id: item for item in report.queues}
    assert by_job[expired_job.job_id].status == "DOING"
    assert by_job[expired_job.job_id].expired_lease is True
    assert by_job["job-dead"].status == "FAILED"
    assert report.lane_rows[0].message_id == msg.message_id
    assert report.lane_rows[0].expired_lease is True
    assert {item.surface for item in report.dead_letters} == {"queue"}
    assert [item.run_id for item in report.run_history] == ["run-1"]
    assert report.run_history[0].namespace == lane_ns
    assert report.run_history[0].last_event_type == "worker.requested"
    assert any(f.surface == "projection_manifest" for f in report.findings)


def test_recovery_run_history_scopes_by_namespace_and_dedupes_across_namespaces(tmp_path):
    engine = _engine(tmp_path)
    ns_a = "ws:demo:conv:bg"
    ns_b = "ws:demo:conv:fg"
    registry = RunRegistry(engine.meta_sqlite)
    registry.create_run(
        run_id="run-a",
        conversation_id=ns_a,
        workflow_id="wf-a",
        user_id=None,
        user_turn_node_id="turn-a",
        status="running",
    )
    registry.append_event("run-a", "worker.requested", {"source": "a"})
    registry.create_run(
        run_id="run-b",
        conversation_id=ns_b,
        workflow_id="wf-b",
        user_id=None,
        user_turn_node_id="turn-b",
        status="running",
    )
    registry.append_event("run-b", "worker.requested", {"source": "b"})
    registry.create_run(
        run_id="run-x",
        conversation_id="ws:demo:other",
        workflow_id="wf-x",
        user_id=None,
        user_turn_node_id="turn-x",
        status="running",
    )

    scoped = engine.recovery.inspect(workspace_id="demo", namespaces=[ns_a])
    assert [item.run_id for item in scoped.run_history] == ["run-a"]
    assert scoped.run_history[0].namespace == ns_a

    combined = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=[ns_a, ns_b],
    )
    assert [item.run_id for item in combined.run_history] == ["run-a", "run-b"]
    assert {item.namespace for item in combined.run_history} == {ns_a, ns_b}


def test_recovery_checkpoint_fallback_is_narrow_and_surfaces_unrelated_errors(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    namespace = "ws:demo:conv:bg"

    def _recoverable(**kwargs):
        raise Exception("Missing Embeddings")

    def _unrelated(**kwargs):
        raise Exception("acl boom")

    monkeypatch.setattr(engine.read, "get_nodes", _recoverable)
    assert engine.recovery.inspect_checkpoints(namespace=namespace, workspace_id="demo") == []

    monkeypatch.setattr(engine.read, "get_nodes", _unrelated)
    with pytest.raises(Exception, match="acl boom"):
        engine.recovery.inspect_checkpoints(namespace=namespace, workspace_id="demo")

    report = engine.recovery.inspect(workspace_id="demo", namespaces=[namespace])
    assert any(
        finding.surface == "checkpoints" and "acl boom" in str(finding.details.get("error", ""))
        for finding in report.findings
    )


def test_recovery_classifies_checkpoints_and_auto_resume_is_policy_gated(tmp_path):
    engine = _engine(tmp_path)
    namespace = "ws:demo:conv:bg"
    with scoped_namespace(engine, namespace):
        engine.write.add_node(_checkpoint(run_id="terminal", step_seq=1, status="completed"))
        engine.write.add_node(
            _checkpoint(run_id="manual", step_seq=1, suspended=True)
        )
        engine.write.add_node(
            _checkpoint(
                run_id="restartable",
                step_seq=1,
                restartable=True,
                resume_marker=True,
            )
        )
        engine.write.add_node(_checkpoint(run_id="unknown", step_seq=1))

    inspected = engine.recovery.inspect_checkpoints(
        namespace=namespace,
        workspace_id="demo",
    )
    classifications = {item.run_id: item.classification for item in inspected}
    assert classifications == {
        "terminal": "terminal",
        "manual": "suspended_manual",
        "restartable": "interrupted_restartable",
        "unknown": "interrupted_unknown",
    }

    report = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=[namespace],
    )
    assert not [action for action in report.actions if action.action_kind == "resume_run"]

    resumed: list[str] = []
    resumed_report = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=[namespace],
        resume_policy=ResumePolicy(
            auto_resume=True,
            resume_runner=lambda checkpoint: resumed.append(checkpoint.run_id) or "ok",
        ),
    )
    assert resumed == ["restartable"]
    assert [
        action.details["run_id"]
        for action in resumed_report.actions
        if action.action_kind == "resume_run"
    ] == ["restartable"]


def test_recovery_report_combines_operator_surfaces(tmp_path):
    engine = _engine(tmp_path)

    report = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=["ws:demo:conv:bg", "ws:demo:maintenance_jobs"],
        app_surfaces=[
            RecoverySurface(
                surface_id="maintenance-daemon",
                surface_kind="daemon_health",
                status="starting",
            )
        ],
    )

    assert report.workspace_id == "demo"
    assert report.namespaces == ("ws:demo:conv:bg", "ws:demo:maintenance_jobs")
    assert [surface.surface_kind for surface in report.app_surfaces] == [
        "daemon_health"
    ]
    assert report.daemon_health[0].daemon_id == "maintenance-daemon"
    assert report.daemon_health[0].observed_state == "starting"
    assert {action.action_kind for action in report.actions} == {
        "repair_lane_projection"
    }


def test_recovery_startup_repairs_missing_service_health_projection_but_inspect_does_not(tmp_path):
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
    engine.meta_sqlite.clear_named_projection(
        "service_health",
        "demo|ws:demo:ops|svc.demo",
    )

    inspected = engine.recovery.inspect(
        workspace_id="demo",
        namespaces=["ws:demo:ops"],
    )
    assert not inspected.daemon_health
    assert not [action for action in inspected.actions if action.surface == "service_health"]
    assert engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    ) is None

    recovered = engine.recovery.recover_startup(
        workspace_id="demo",
        namespaces=["ws:demo:ops"],
    )
    assert any(item.daemon_id == "svc.demo" for item in recovered.daemon_health)
    service_health_actions = [
        action
        for action in recovered.actions
        if action.surface == "service_health"
    ]
    assert service_health_actions
    assert service_health_actions[0].details["service_id"] == "svc.demo"
    assert service_health_actions[0].details["status"] == "starting"
    assert engine.service_health.get_service(
        "svc.demo",
        workspace_id="demo",
        namespace="ws:demo:ops",
    ) is not None
