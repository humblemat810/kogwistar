from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kogwistar.server.runtime_api import create_runtime_router

pytestmark = [pytest.mark.integration]


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


def test_workflow_design_history_denies_without_acl_before_read() -> None:
    from fastapi import HTTPException

    reads = []

    class Service:
        def workflow_design_history(self, *, workflow_id: str):
            reads.append(workflow_id)
            return {"workflow_id": workflow_id, "versions": []}

    def require_access(workflow_id: str, role: str) -> None:
        assert role == "ro"
        if workflow_id == "private-wf":
            raise HTTPException(status_code=403, detail="workflow ACL denied")

    app = FastAPI()
    app.include_router(
        create_runtime_router(
            get_service=Service,
            require_role=lambda _role: None,
            require_namespace=lambda _namespaces: None,
            runtime_namespaces={"workflow"},
            require_workflow_access=require_access,
        )
    )
    with TestClient(app) as client:
        denied = client.get("/api/workflow/design/private-wf/history")
        allowed = client.get("/api/workflow/design/visible-wf/history")

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert reads == ["visible-wf"]


def test_workflow_design_node_upsert_requires_rw_acl_before_dispatch() -> None:
    from fastapi import HTTPException

    calls = []

    class Service:
        def workflow_design_upsert_node(self, **kwargs):
            calls.append(kwargs)
            return {"workflow_id": kwargs["workflow_id"], "node_id": kwargs["node_id"]}

    def require_access(workflow_id: str, role: str) -> None:
        allowed = {"read-only-wf": "ro", "writable-wf": "rw"}.get(workflow_id)
        if allowed != role and not (allowed == "rw" and role == "ro"):
            raise HTTPException(status_code=403, detail="workflow ACL denied")

    app = FastAPI()
    app.include_router(
        create_runtime_router(
            get_service=Service,
            require_role=lambda _role: None,
            require_namespace=lambda _namespaces: None,
            runtime_namespaces={"workflow"},
            require_workflow_access=require_access,
        )
    )
    node = {
        "designer_id": "alice",
        "node_id": "start",
        "label": "Start",
        "op": "start",
        "start": True,
    }
    with TestClient(app) as client:
        missing_acl = client.post("/api/workflow/design/no-acl-wf/nodes", json=node)
        read_only = client.post("/api/workflow/design/read-only-wf/nodes", json=node)
        writable = client.post("/api/workflow/design/writable-wf/nodes", json=node)

    assert missing_acl.status_code == 403
    assert read_only.status_code == 403
    assert writable.status_code == 200
    assert calls == [
        {
            **node,
            "workflow_id": "writable-wf",
            "actor_sub": None,
            "source": "rest",
            "terminal": False,
            "fanout": False,
            "metadata": {},
        }
    ]


def test_workflow_design_remaining_routes_enforce_acl_before_effect() -> None:
    from fastapi import HTTPException

    effects = []
    acl_checks = []

    class Service:
        def workflow_design_graph(self, *, workflow_id: str, refresh: bool = False):
            effects.append(("graph", workflow_id, refresh))
            return {"workflow_id": workflow_id, "nodes": [], "edges": []}

        def workflow_design_upsert_edge(self, **kwargs):
            effects.append(("edge_upsert", kwargs))
            return {}

        def workflow_design_delete_node(self, **kwargs):
            effects.append(("node_delete", kwargs))
            return {}

        def workflow_design_delete_edge(self, **kwargs):
            effects.append(("edge_delete", kwargs))
            return {}

        def workflow_design_undo(self, **kwargs):
            effects.append(("undo", kwargs))
            return {}

        def workflow_design_redo(self, **kwargs):
            effects.append(("redo", kwargs))
            return {}

    def require_access(workflow_id: str, role: str) -> None:
        acl_checks.append((workflow_id, role))
        acl_role = {"read-only-wf": "ro", "visible-wf": "ro"}.get(workflow_id)
        if acl_role is None or (role == "rw" and acl_role != "rw"):
            raise HTTPException(status_code=403, detail="workflow ACL denied")

    app = FastAPI()
    app.include_router(
        create_runtime_router(
            get_service=Service,
            require_role=lambda _role: None,
            require_namespace=lambda _namespaces: None,
            runtime_namespaces={"workflow"},
            require_workflow_access=require_access,
        )
    )
    write_body = {"designer_id": "alice"}
    cases = [
        ("GET", "/api/workflow/design/no-acl-wf/graph", None, 403),
        ("GET", "/api/workflow/design/visible-wf/graph", None, 200),
        (
            "POST",
            "/api/workflow/design/no-acl-wf/edges",
            {
                "designer_id": "alice",
                "edge_id": "e1",
                "src": "n1",
                "dst": "n2",
                "relation": "next",
            },
            403,
        ),
        ("DELETE", "/api/workflow/design/no-acl-wf/nodes/n1", write_body, 403),
        ("DELETE", "/api/workflow/design/no-acl-wf/edges/e1", write_body, 403),
        ("POST", "/api/workflow/design/no-acl-wf/undo", write_body, 403),
        ("POST", "/api/workflow/design/no-acl-wf/redo", write_body, 403),
        (
            "POST",
            "/api/workflow/design/read-only-wf/edges",
            {
                "designer_id": "alice",
                "edge_id": "e1",
                "src": "n1",
                "dst": "n2",
                "relation": "next",
            },
            403,
        ),
        ("DELETE", "/api/workflow/design/read-only-wf/nodes/n1", write_body, 403),
        ("DELETE", "/api/workflow/design/read-only-wf/edges/e1", write_body, 403),
        ("POST", "/api/workflow/design/read-only-wf/undo", write_body, 403),
        ("POST", "/api/workflow/design/read-only-wf/redo", write_body, 403),
    ]
    with TestClient(app) as client:
        for method, path, body, expected_status in cases:
            response = client.request(method, path, json=body)
            assert response.status_code == expected_status, path

    assert effects == [("graph", "visible-wf", False)]
    assert acl_checks == [
        ("no-acl-wf", "ro"),
        ("visible-wf", "ro"),
        ("no-acl-wf", "rw"),
        ("no-acl-wf", "rw"),
        ("no-acl-wf", "rw"),
        ("no-acl-wf", "rw"),
        ("no-acl-wf", "rw"),
        ("read-only-wf", "rw"),
        ("read-only-wf", "rw"),
        ("read-only-wf", "rw"),
        ("read-only-wf", "rw"),
        ("read-only-wf", "rw"),
    ]


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
    for feature in (
        'id="runId"',
        "edge_selected",
        "wf_priority",
        "wf_multiplicity",
        "selectedEdges",
        "data-edge",
    ):
        assert feature in template
