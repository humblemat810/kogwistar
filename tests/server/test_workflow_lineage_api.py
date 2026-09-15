from fastapi import FastAPI
from fastapi.testclient import TestClient

from kogwistar.server.runtime_api import create_runtime_router


class _Service:
    def __init__(self, lineage=None, missing=False):
        self.lineage = lineage or []
        self.missing = missing

    def get_run(self, run_id):
        if self.missing:
            raise KeyError(f"Workflow run not found: {run_id}")
        return {"run_id": run_id, "workflow_id": "wf.parent"}

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
    service = _Service(
        lineage=[
            {"run_id": "child", "workflow_id": "wf.child", "parent_run_id": "parent"},
            {"run_id": "parent", "workflow_id": "wf.parent", "parent_run_id": None},
        ]
    )
    with _client(service, {"wf.parent", "wf.child"}) as client:
        response = client.get("/api/workflow/runs/child/lineage")
    assert response.status_code == 200
    assert response.json()["lineage"][0]["workflow_id"] == "wf.child"


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


def test_lineage_returns_404_for_missing_run():
    with _client(_Service(missing=True), set()) as client:
        response = client.get("/api/workflow/runs/missing/lineage")
    assert response.status_code == 404
