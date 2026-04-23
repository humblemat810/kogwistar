from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.chat_service import ChatRunService
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.server_fixtures import build_engine_triplet


pytestmark = pytest.mark.server

@pytest.fixture()
def service_triplet():
    root = Path(".tmp_repair_tests") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        knowledge, conversation, workflow = build_engine_triplet(
            root=root,
            embedding_function=FakeEmbeddingFunction(dim=3),
            backend_factory=build_fake_backend,
        )
        service = ChatRunService(
            get_knowledge_engine=lambda: knowledge,
            get_conversation_engine=lambda: conversation,
            get_workflow_engine=lambda: workflow,
            run_registry=RunRegistry(workflow.meta_sqlite),
        )
        yield service, conversation, workflow
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _claims():
    return claims_ctx.set(
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
            "sub": "repair-test",
        }
    )


def _workflow(service, workflow_id: str):
    service.workflow_design_upsert_node(
        workflow_id=workflow_id,
        designer_id="repair-test",
        node_id="start",
        label="Start",
        op="start",
        start=True,
    )
    service.workflow_design_upsert_node(
        workflow_id=workflow_id,
        designer_id="repair-test",
        node_id="end",
        label="End",
        op="end",
        terminal=True,
    )
    service.workflow_design_upsert_edge(
        workflow_id=workflow_id,
        designer_id="repair-test",
        edge_id="edge-start-end",
        src="start",
        dst="end",
        relation="wf_next",
        is_default=True,
    )


def test_slice12_repair_service_projection_and_dead_letters(service_triplet):
    service, conversation, workflow = service_triplet
    token = _claims()
    try:
        conv = service.create_conversation(user_id="repair-user")
        _workflow(service, "wf.repair")
        service.declare_service(
            service_id="svc.repair.demo",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.repair",
            target_config={"conversation_id": conv["conversation_id"]},
            enabled=True,
            autostart=True,
        )
        service.service_supervisor.bootstrap()
        projection = service.repair_service_projection("svc.repair.demo")
        assert projection["projection"]["service_id"] == "svc.repair.demo"
        failing = service.scheduler.submit(
            run_id="dead-demo",
            priority_class="background",
            start_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            max_retries=0,
        )
        assert failing["admission"] == "accepted"
        import time

        time.sleep(0.05)
        dead = service.dead_letter_snapshot(limit=10)
        assert any(item["run_id"] == "dead-demo" for item in dead["runs"])
    finally:
        claims_ctx.reset(token)


def test_slice12_repair_orphaned_claimed_messages_and_replay_history(service_triplet):
    service, conversation, workflow = service_triplet
    token = _claims()
    try:
        conv = service.create_conversation(user_id="repair-user")
        _workflow(service, "wf.repair.history")
        service.declare_service(
            service_id="svc.repair.history",
            service_kind="daemon",
            target_kind="workflow",
            target_ref="wf.repair.history",
            target_config={"conversation_id": conv["conversation_id"]},
            enabled=True,
        )
        run = service.trigger_service(
            "svc.repair.history",
            trigger_type="external event",
            payload={},
        )
        run_id = str(run["current_child_run_id"] or "")
        history = service.replay_run_history(run_id)
        assert history["run_id"] == run_id
        assert history["event_count"] >= 0
        repaired = service.repair_orphaned_claimed_messages()
        assert "repaired_message_ids" in repaired
    finally:
        claims_ctx.reset(token)
