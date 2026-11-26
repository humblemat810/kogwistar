"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_designer_capabilities_endpoint(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-designer-capabilities-schema-v1.json"
        ).read_text(encoding="utf-8")
    )
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    class _StubResolver:
        def __init__(self):
            self.handlers = {"start": lambda ctx: None, "answer": lambda ctx: None}
            self.nested_ops = {"answer"}
            self.sandboxed_ops = {"python_exec"}
            self._sandbox = object()

        @property
        def ops(self):
            return set(self.handlers)

        def describe_state(self):
            return {"messages": "a", "selected_node_id": "u"}

    service.resolver = _StubResolver()

    with TestClient(server.app) as client:
        docs_ro = _token_header(client, role="ro", ns="docs")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        wf_cap = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities="workflow.design.inspect",
        )

        forbidden = client.get("/designer/capabilities", headers=docs_ro)
        assert forbidden.status_code == 403

        resp = client.get("/designer/capabilities", headers=wf_cap)
        resp.raise_for_status()
        payload = resp.json()

        assert payload["schema_version"] == "workflow-designer-capabilities/v1"
        assert payload["projection_schema"] == "workflow_design_v1"
        assert payload["design_features"]["undo_redo"] is True
        assert payload["custom_ops"]["allow_unregistered_ops_in_design"] is True
        assert payload["custom_ops"]["allow_execution_of_unregistered_ops"] is False

        runtime = payload["runtime"]
        assert runtime["resolver_found"] is True
        assert sorted(runtime["builtin_ops"]) == sorted(["answer", "llm_call", "start"])
        assert runtime["nested_ops"] == ["answer"]
        assert runtime["sandboxed_ops"] == ["python_exec"]
        assert runtime["sandbox"]["supports_sandboxed_ops"] is True
        assert runtime["sandbox"]["runtime_configured"] is True
        assert runtime["state_schema"] == {"messages": "a", "selected_node_id": "u"}

        node_types = payload["node_types"]
        edge_types = payload["edge_types"]
        assert payload["schema_version"] == contract["schema_version"]
        assert node_types[0]["metadata_schema"] == contract["node_metadata_schema"]
        assert edge_types[0]["metadata_schema"] == contract["edge_metadata_schema"]
        assert node_types[0]["type"] == "workflow_node"
        assert edge_types[0]["type"] == "workflow_edge"
        node_props = node_types[0]["metadata_schema"]["properties"]
        edge_props = edge_types[0]["metadata_schema"]["properties"]
        assert "wf_op" in node_props
        assert "wf_join" in node_props
        assert "wf_predicate" in edge_props
        assert "wf_multiplicity" in edge_props


def test_designer_capabilities_requires_explicit_capability(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with TestClient(server.app) as client:
        wf_ro = _token_header(client, role="ro", ns="workflow")
        with_cap = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities="workflow.design.inspect",
        )

        forbidden = client.get("/designer/capabilities", headers=wf_ro)
        assert forbidden.status_code == 403

        resp = client.get("/designer/capabilities", headers=with_cap)
        resp.raise_for_status()
        assert resp.json()["schema_version"] == "workflow-designer-capabilities/v1"


def test_designer_capabilities_falls_back_to_default_resolver(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with TestClient(server.app) as client:
        wf_ro = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities="workflow.design.inspect",
        )

        resp = client.get("/designer/capabilities", headers=wf_ro)
        resp.raise_for_status()
        payload = resp.json()

        runtime = payload["runtime"]
        assert runtime["resolver_found"] is True
        assert "start" in runtime["builtin_ops"]


def test_admin_delete_doc_uses_engine_rollback_lifecycle(monkeypatch):
    doc_id = "doc-1"
    calls: list[str] = []

    def rollback_document(value):
        calls.append(value)
        return {
            "tombstoned_node_ids": ["node-1", "node-2"],
            "tombstoned_edge_ids": ["edge-1"],
        }

    monkeypatch.setattr(
        server,
        "engine",
        _FixedResource(
            SimpleNamespace(
                rollback=SimpleNamespace(rollback_document=rollback_document)
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(server, "require_role", lambda role: None, raising=False)

    payload = server.admin_delete_doc(doc_id)

    assert payload == {
        "ok": True,
        "doc_id": doc_id,
        "deleted": {"nodes": 2, "edges": 1},
    }
    assert calls == [doc_id]


def test_admin_delete_doc_maps_rollback_failure(monkeypatch):
    doc_id = "doc-2"

    def rollback_document(_value):
        raise RuntimeError("rollback failed")

    monkeypatch.setattr(
        server,
        "engine",
        _FixedResource(
            SimpleNamespace(
                rollback=SimpleNamespace(rollback_document=rollback_document)
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(server, "require_role", lambda role: None, raising=False)

    with pytest.raises(HTTPException) as exc_info:
        server.admin_delete_doc(doc_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == f"Failed to roll back document {doc_id!r}"


def test_workflow_run_submit_goal_payload_survives_failing_otel_sink(
    monkeypatch, engine_triplet
):
    """Goal is ordinary workflow state; OTel failure cannot fail its endpoint run."""
    engine, conversation_engine, workflow_engine = engine_triplet
    captured: dict[str, object] = {}

    class _FailingOtelTracer:
        def start_span(self, *_args, **_kwargs):
            raise RuntimeError("simulated OTel exporter failure")

    def _runtime_runner(req: RuntimeRunRequest) -> dict[str, object]:
        captured["claims"] = req.auth_claims
        captured["initial_state"] = dict(req.initial_state)
        sink = OpenTelemetrySink(_FailingOtelTracer(), queue_max=1)
        try:
            emitter = EventEmitter(sink=sink)
            emitter.emit(
                type="workflow_run_started",
                ctx=TraceContext.new_root(
                    run_id=req.run_id,
                    token_id="goal-token",
                    step_seq=0,
                    node_id="goal-start",
                ),
            )
        finally:
            sink.close(timeout=1.0)
        return {
            "workflow_status": "succeeded",
            "final_state": {
                **req.initial_state,
                "goal_status": "satisfied",
            },
        }

    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _debug_rag_runner,
        runtime_runner=_runtime_runner,
    )

    with TestClient(server.app) as client:
        headers = _token_header(client, role="rw", ns="workflow,conversation")
        conv = client.post(
            "/api/conversations",
            json={"user_id": "goal-api-user"},
            headers=headers,
        )
        assert conv.status_code == 200, conv.text
        response = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "goal.control.v1",
                "conversation_id": conv.json()["conversation_id"],
                "initial_state": {
                    "goal_id": "goal-api-1",
                    "goal_objective": "complete smoke test",
                    "goal_iteration": 0,
                },
            },
            headers=headers,
        )
        assert response.status_code == 202, response.text
        result = _wait_for_status(
            client,
            response.json()["run_id"],
            headers,
            {"succeeded"},
            path_template="/api/workflow/runs/{run_id}",
        )

    assert result["status"] == "succeeded"
    assert captured["claims"]
    assert captured["initial_state"] == {
        "goal_id": "goal-api-1",
        "goal_objective": "complete smoke test",
        "goal_iteration": 0,
    }
