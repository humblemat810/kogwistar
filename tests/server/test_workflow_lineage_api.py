import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowRunNode
from kogwistar.server.auth_middleware import reset_claims_ctx, set_claims_ctx
from kogwistar.server.chat_service import ChatRunService
from kogwistar.server.run_registry import RunRegistry
from kogwistar.server.runtime_api import create_runtime_router
from tests._helpers.embeddings import ConstantEmbeddingFunction

pytestmark = [pytest.mark.integration]


class _Service:
    def __init__(self, lineage=None, missing=False):
        self.lineage = lineage or []
        self.missing = missing

    def get_run(self, run_id):
        if self.missing:
            raise KeyError(f"Workflow run not found: {run_id}")
        return {"run_id": run_id, "workflow_id": "wf.parent", "user_id": "run-owner"}

    def workflow_run_lineage(self, run_id):
        return {"run_id": run_id, "lineage": self.lineage}


def _client(service, allow):
    app = FastAPI()

    def require_access(workflow_id, role):
        if workflow_id not in allow:
            raise PermissionError(f"workflow denied: {workflow_id}")

    app.include_router(
        create_runtime_router(
            get_service=lambda: service,
            require_role=lambda _role: None,
            require_namespace=lambda _namespaces: None,
            runtime_namespaces={"workflow"},
            require_workflow_access=require_access,
        )
    )
    return TestClient(app)


def test_lineage_checks_parent_and_every_child_acl():
    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/golden/workflow-run-lineage-v1.json").read_text(
            encoding="utf-8"
        )
    )
    service = _Service(
        lineage=[
            {
                "run_id": "lineage-child",
                "workflow_id": "wf-child",
                "parent_run_id": "lineage-parent",
                "status": "completed",
                "wf_invoked": [],
            },
            {
                "run_id": "lineage-parent",
                "workflow_id": "wf-parent",
                "parent_run_id": None,
                "status": "completed",
                "wf_invoked": [],
            },
        ]
    )
    with _client(service, {"wf.parent", "wf-parent", "wf-child"}) as client:
        response = client.get("/api/workflow/runs/lineage-child/lineage")
    assert response.status_code == contract["success_status"]
    assert response.json() == contract["response"]


def test_lineage_denies_unauthorized_parent_or_child():
    service = _Service(
        lineage=[
            {"run_id": "child", "workflow_id": "wf.child", "parent_run_id": "parent"},
            {"run_id": "parent", "workflow_id": "wf.parent", "parent_run_id": None},
        ]
    )
    with _client(service, {"wf.parent"}) as client:
        assert client.get("/api/workflow/runs/child/lineage").status_code == 403
    with _client(service, {"wf.parent", "wf.child"}) as client:
        assert client.get("/api/workflow/runs/child/lineage").status_code == 200


def test_lineage_returns_visible_prefix_when_graph_read_omits_parent():
    visible_child = {
        "run_id": "lineage-child",
        "workflow_id": "wf-child",
        "parent_run_id": "lineage-parent",
        "status": "completed",
        "wf_invoked": [],
    }
    service = _Service(lineage=[visible_child])
    with _client(service, {"wf.parent", "wf-child"}) as client:
        response = client.get("/api/workflow/runs/lineage-child/lineage")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "lineage-child",
        "lineage": [visible_child],
    }


def _seed_workflow_lineage_acl(engine):
    mention = Grounding(spans=[Span.from_dummy_for_conversation()])
    for run_id, workflow_id, parent_run_id in (
        ("lineage-child", "wf.child", "lineage-parent"),
        ("lineage-parent", "wf.parent", None),
    ):
        engine.write.add_node(
            WorkflowRunNode(
                id=f"wf_run|{run_id}",
                label=run_id,
                type="entity",
                doc_id=f"doc:{run_id}",
                summary=run_id,
                mentions=[mention],
                metadata={
                    "entity_type": "workflow_run",
                    "workflow_id": workflow_id,
                    "workflow_version": "v1",
                    "run_id": run_id,
                    "conversation_id": "conversation-1",
                    "turn_node_id": "turn-1",
                    "status": "completed",
                    "parent_run_id": parent_run_id,
                },
            )
        )
    assert engine.raw_read.get_nodes(
        ids=["wf_run|lineage-child"], limit=1
    )
    engine.acl.record_acl(
        truth_graph="conversation",
        entity_id="wf_run|lineage-child",
        version=1,
        mode="public",
    )
    child_node = engine.raw_read.get_nodes(ids=["wf_run|lineage-child"], limit=1)[0]
    grounding_ids, span_ids = engine.acl.usage_ids_for_item(child_node)
    for grain, target_ids in (("grounding", grounding_ids), ("span", span_ids)):
        for target_item_id in target_ids:
            engine.acl.record_acl(
                grain=grain,
                truth_graph="conversation",
                entity_id="wf_run|lineage-child",
                target_item_id=target_item_id,
                version=1,
                mode="public",
            )
    engine.acl.record_acl(
        truth_graph="conversation",
        entity_id="wf_run|lineage-parent",
        version=1,
        mode="private",
        owner_id="bob",
    )


def _assert_python_lineage_visible_prefix(engine):
    registry = RunRegistry(engine.meta_sqlite)
    registry.create_run(
        run_id="lineage-child",
        conversation_id="conversation-1",
        workflow_id="wf.child",
        user_id="alice",
        user_turn_node_id="turn-1",
        status="completed",
    )
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: engine,
        get_workflow_engine=lambda: engine,
        run_registry=registry,
    )
    token = set_claims_ctx(
        {"sub": "alice", "user_id": "alice", "ns": "workflow"}
    )
    try:
        assert {
            node.id
            for node in engine.read.get_nodes(
                ids=["wf_run|lineage-child", "wf_run|lineage-parent"], limit=2
            )
        } == {"wf_run|lineage-child"}
        with _client(service, {"wf.child"}) as client:
            response = client.get(
                "/api/workflow/runs/lineage-child/lineage"
            )
    finally:
        reset_claims_ctx(token)

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "lineage-child",
        "lineage": [
            {
                "run_id": "lineage-child",
                "workflow_id": "wf.child",
                "parent_run_id": "lineage-parent",
                "status": "completed",
                "wf_invoked": None,
            }
        ],
    }


def test_python_lineage_stops_at_parent_hidden_by_graph_acl(tmp_path):
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "lineage-acl"),
        kg_graph_type="conversation",
        embedding_function=ConstantEmbeddingFunction(dim=3),
        backend_factory=build_in_memory_backend,
        acl_enabled=True,
    )
    try:
        _seed_workflow_lineage_acl(engine)
        _assert_python_lineage_visible_prefix(engine)
    finally:
        engine.close()


def test_python_postgres_lineage_stops_at_parent_hidden_by_graph_acl(
    tmp_path, pg_dsn, pg_schema
):
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture is unavailable")

    import sqlalchemy as sa

    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    postgres_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "lineage-postgres-acl"),
        backend=PgVectorBackend(
            engine=postgres_engine, embedding_dim=3, schema=pg_schema
        ),
        kg_graph_type="conversation",
        embedding_function=ConstantEmbeddingFunction(dim=3),
        acl_enabled=True,
    )
    try:
        _seed_workflow_lineage_acl(engine)
        _assert_python_lineage_visible_prefix(engine)
    finally:
        engine.close()
        postgres_engine.dispose()


def test_run_status_denies_unauthorized_workflow():
    with _client(_Service(), set()) as client:
        response = client.get("/api/workflow/runs/private-run")
    assert response.status_code == 403


def test_run_status_allows_acl_reader_who_does_not_own_run():
    with _client(_Service(), {"wf.parent"}) as client:
        response = client.get("/api/workflow/runs/shared-run")
    assert response.status_code == 200
    assert response.json()["user_id"] == "run-owner"


def test_lineage_returns_404_for_missing_run():
    with _client(_Service(missing=True), set()) as client:
        response = client.get("/api/workflow/runs/missing/lineage")
    assert response.status_code == 404
