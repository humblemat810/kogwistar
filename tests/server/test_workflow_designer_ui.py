from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kogwistar.server.runtime_api import create_runtime_router


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_runtime_router(
            get_service=lambda: object(),
            require_role=lambda _role: None,
            require_namespace=lambda _namespaces: None,
            runtime_namespaces=object(),
        )
    )
    return app


def test_workflow_designer_route_renders_without_external_assets() -> None:
    with TestClient(_app()) as client:
        response = client.get("/api/workflow/designer")

    assert response.status_code == 200
    assert "Workflow Studio" in response.text
    assert "/api/workflow/design/" in response.text
    assert "https://fonts.googleapis.com" not in response.text


def test_workflow_designer_read_routes_apply_acl() -> None:
    checks = []

    class Service:
        def workflow_design_graph(self, *, workflow_id: str, refresh: bool = False):
            return {"workflow_id": workflow_id, "nodes": [], "edges": []}

        def workflow_catalog_ops(self):
            return []

    def require_role(role: str) -> None:
        checks.append(("role", role))

    def require_namespace(namespaces) -> None:
        checks.append(("namespace", namespaces))

    def require_access(workflow_id: str, role: str) -> None:
        checks.append(("access", workflow_id, role))

    app = FastAPI()
    app.include_router(
        create_runtime_router(
            get_service=Service,
            require_role=require_role,
            require_namespace=require_namespace,
            runtime_namespaces={"workflow"},
            require_workflow_access=require_access,
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/workflow/designer").status_code == 200
        assert client.get("/api/workflow/design/wf-1/graph").status_code == 200
        assert client.get("/api/workflow/catalog/ops").status_code == 200

    assert ("role", "ro") in checks
    assert ("access", "wf-1", "ro") in checks


def test_workflow_designer_template_contains_graph_semantics() -> None:
    template = Path("kogwistar/templates/workflow_designer.html").read_text(
        encoding="utf-8"
    )

    for semantic in ("wf_op", "wf_predicate", "wf_start", "wf_terminal"):
        assert semantic in template
    for action in ("/undo", "/redo", "/graph"):
        assert action in template
    for interaction in (
        "state.connect",
        "onwheel",
        "onpointermove",
        "zoomIn",
        "zoomOut",
        "fit",
        "onkeydown",
        "Escape",
        "ctrlKey",
    ):
        assert interaction in template
    assert "sessionStorage" in template
    assert "Authorization" in template
    assert "/api/auth/me" in template
    assert "/api/auth/login?return_to=" in template
