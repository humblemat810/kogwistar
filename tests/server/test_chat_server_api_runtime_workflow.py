"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_runtime_rest_submit_and_sse(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-status-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_success_runner,
    )
    with TestClient(server.app) as client:
        conv_rw = _token_header(client, role="rw", ns="conversation")
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")

        created = client.post(
            "/api/conversations", json={"user_id": "u-runtime"}, headers=conv_rw
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "wf.runtime.simple",
                "conversation_id": conversation_id,
                "initial_state": {"counter": 1},
                "priority_class": "batch",
            },
            headers=wf_rw,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]
        assert submit.json()["priority_class"] == "batch"

        final_run = _wait_for_status(
            client,
            run_id,
            wf_ro,
            {"succeeded"},
            path_template="/api/workflow/runs/{run_id}",
        )
        assert final_run["workflow_id"] == "wf.runtime.simple"
        assert final_run["result"]["workflow_status"] == "succeeded"
        assert "last_step_seq" not in final_run
        assert "step_count" not in final_run
        assert contract["success_status"] == 200
        assert set(contract["required_fields"]).issubset(final_run)
        assert final_run["terminal"] is True

        events = _collect_sse_events(
            client,
            run_id,
            wf_ro,
            path_template="/api/workflow/runs/{run_id}/events",
        )
        names = [name for name, _payload in events]
        assert "run.created" in names
        assert "run.started" in names
        assert "run.stage" in names
        assert "reasoning.summary" in names
        assert names[-1] == "run.completed"


def test_workflow_run_lane_message_lifecycle_streams_through_registry_sse(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_lane_message_runner,
    )
    with TestClient(server.app) as client:
        conv_rw = _token_header(client, role="rw", ns="conversation")
        conv_ro = _token_header(client, role="ro", ns="conversation")
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")

        created = client.post(
            "/api/conversations",
            json={"user_id": "u-runtime-lane"},
            headers=conv_rw,
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "wf.runtime.lane",
                "conversation_id": conversation_id,
                "initial_state": {},
            },
            headers=wf_rw,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]
        _wait_for_status(
            client,
            run_id,
            wf_ro,
            {"succeeded"},
            path_template="/api/workflow/runs/{run_id}",
        )

        polled = client.get(
            f"/api/runs/{run_id}/events/poll?after_seq=0&limit=50",
            headers=conv_ro,
        )
        polled.raise_for_status()
        polled_names = [event["event_type"] for event in polled.json()["events"]]
        assert "worker.requested" in polled_names

        sse_events = _collect_sse_events(
            client,
            run_id,
            wf_ro,
            path_template="/api/workflow/runs/{run_id}/events",
        )
        worker_payloads = [
            payload for name, payload in sse_events if name == "worker.requested"
        ]
        assert worker_payloads
        assert worker_payloads[0]["message_id"] == "msg:server-runtime-lane"
        assert worker_payloads[0]["inbox_id"] == "inbox:worker:runtime"
        assert worker_payloads[0]["status"] == "pending"


def test_runtime_resource_snapshot_reports_pressure(monkeypatch, engine_triplet):
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-resources-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_success_runner,
    )
    with TestClient(server.app) as client:
        partial = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities="project_view",
        )
        denied = client.get("/api/workflow/resources", headers=partial)
        assert denied.status_code == 403

        wf_ro = _token_header(
            client,
            role="ro",
            ns="workflow",
            capabilities=",".join(contract["required_capabilities"]),
        )
        snapshot = client.get("/api/workflow/resources", headers=wf_ro)
    assert snapshot.status_code == 200
    payload = snapshot.json()
    assert all(field in payload for field in contract["required_fields"])
    for key, required_fields in contract["required_object_fields"].items():
        assert all(field in payload[key] for field in required_fields), key
    assert not set(contract["rust_additive_fields"]).intersection(payload)
    assert isinstance(payload["storage_usage_bytes"], int)
    assert payload["scheduler"]["max_active"] >= 1


def test_runtime_rest_cancel(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_cancel_runner,
    )
    with TestClient(server.app) as client:
        conv_rw = _token_header(client, role="rw", ns="conversation")
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")

        created = client.post(
            "/api/conversations", json={"user_id": "u-runtime-cancel"}, headers=conv_rw
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "wf.runtime.cancel",
                "conversation_id": conversation_id,
                "initial_state": {"counter": 1},
            },
            headers=wf_rw,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

        cancel_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts"
                / "golden"
                / "run-cancel-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert cancel_contract["workflow_route"] == "POST /api/workflow/runs/{run_id}/cancel"
        denied = client.post(
            f"/api/workflow/runs/{run_id}/cancel", headers=wf_ro
        )
        assert denied.status_code == 403
        assert conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        ) == []

        cancel = client.post(f"/api/workflow/runs/{run_id}/cancel", headers=wf_rw)
        assert cancel.status_code == cancel_contract["accepted_status"]
        cancel_body = cancel.json()
        assert all(field in cancel_body for field in cancel_contract["required_fields"])
        assert cancel_body["cancel_requested"] is True
        assert cancel_body["status"] in cancel_contract["accepted_run_statuses"]

        final_run = _wait_for_status(
            client,
            run_id,
            wf_ro,
            {"cancelled"},
            path_template="/api/workflow/runs/{run_id}",
        )
        assert final_run["status"] == "cancelled"
        assert "last_step_seq" not in final_run
        assert "step_count" not in final_run

        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1

        events = _collect_sse_events(
            client,
            run_id,
            wf_ro,
            path_template="/api/workflow/runs/{run_id}/events",
        )
        names = [name for name, _payload in events]
        assert "run.cancelling" in names
        assert names[-1] == "run.cancelled"


def test_view_layer_hardening_endpoints(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_success_runner,
    )
    service.cost_ledger.add_event(kind="token", amount=7, source="test")
    service.cost_ledger.add_event(kind="time", amount=11, source="test")

    conversation_engine.write.add_node(
        ConversationNode(
            user_id="tester",
            id="tool-call:1",
            label="tool_call:kg_neighbors",
            type="entity",
            doc_id="tool-call:1",
            summary="Read adjacent graph nodes",
            role="assistant",
            turn_index=2,
            conversation_id="conversation:demo",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"status": "requested"},
            metadata={
                "entity_type": "tool_call",
                "tool_name": "kg_neighbors",
                "level_from_root": 0,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
    )
    conversation_engine.write.add_node(
        ConversationNode(
            user_id="tester",
            id="tool-result:1",
            label="tool_result:kg_neighbors",
            type="entity",
            doc_id="tool-result:1",
            summary="Returned two visible neighbors",
            role="tool",
            turn_index=2,
            conversation_id="conversation:demo",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"status": "completed", "count": 2},
            metadata={
                "entity_type": "tool_result",
                "tool_name": "kg_neighbors",
                "level_from_root": 0,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
    )

    with TestClient(server.app) as client:
        wf_ro = _token_header(client, role="ro", ns="workflow")
        vis = client.get("/api/workflow/visibility", headers=wf_ro)
        vis.raise_for_status()
        assert vis.json()["namespaces"]["execution_namespace"] == "workflow"

        visibility_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-visibility-http-v1.json"
            ).read_text(encoding="utf-8")
        )
        scoped_token = jwt.encode(
            {
                "sub": "reader-1",
                "role": "ro",
                "ns": ["workflow"],
                "capabilities": ["project_view", "read_security_scope"],
                "tenant": "tenant-a",
                "workspace": "workspace-b",
                "project": "project-c",
                "iss": "local",
                "exp": int(time.time()) + 300,
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        scoped_visibility = client.get(
            "/api/workflow/visibility",
            headers={"Authorization": f"Bearer {scoped_token}"},
        )
        assert scoped_visibility.status_code == visibility_contract["success_status"]
        assert scoped_visibility.json() == visibility_contract["response"]

        budget = client.get("/api/workflow/budget/history", headers=wf_ro)
        budget.raise_for_status()
        assert budget.json()["cost_ledger"]["event_count"] == 2
        assert len(budget.json()["events"]) == 2

        timeline_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-scheduler-timeline-empty-v1.json"
            ).read_text(encoding="utf-8")
        )
        sched = client.get(
            "/api/workflow/scheduler/timeline",
            params={"run_id": timeline_contract["run_id"]},
            headers=wf_ro,
        )
        assert sched.status_code == timeline_contract["success_status"]
        assert sched.json() == timeline_contract["response"]

        expected_audit = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/tool-audit-v1.json"
            ).read_text(encoding="utf-8")
        )
        expected_audit.pop("schema_version")
        for item in expected_audit["items"]:
            item.pop("schema_version")
        audit = client.get(
            "/api/workflow/tools/audit",
            params={"conversation_id": "conversation:demo"},
            headers=wf_ro,
        )
        audit.raise_for_status()
        assert audit.json() == expected_audit


def test_get_run_keeps_registry_status_even_if_workflow_completed_node_exists(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "run-terminal-completed"
    registry.create_run(
        run_id=run_id,
        conversation_id="conv-terminal",
        workflow_id="wf.completed",
        user_id="u1",
        user_turn_node_id="turn-1",
        status="running",
    )
    monkeypatch.setattr(
        service,
        "list_steps",
        lambda _run_id: (_ for _ in ()).throw(
            AssertionError("get_run() must not inspect workflow steps")
        ),
        raising=False,
    )
    _seed_terminal_node(
        conversation_engine,
        node_cls=WorkflowCompletedNode,
        entity_type="workflow_completed",
        run_id=run_id,
        workflow_id="wf.completed",
        conversation_id="conv-terminal",
    )

    run = service.get_run(run_id)
    assert run["status"] == "running"
    assert run["terminal"] is False
    assert "last_step_seq" not in run
    assert "step_count" not in run


def test_run_status_route_matches_shared_v1_contract(monkeypatch, engine_triplet):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "run-status-v1.json"
    )
    contract = json.loads(fixture_path.read_text(encoding="utf-8"))
    workflow_contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "workflow-run-status-http-v1.json"
    )
    workflow_contract = json.loads(
        workflow_contract_path.read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="conversation")
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        for status in ("queued", "running", "succeeded", "failed", "cancelled"):
            expected_ids = {
                "run_id": f"run-status-{status}",
                "conversation_id": "conversation-status-contract",
                "workflow_id": "workflow-status-contract",
                "user_id": "user-status-contract",
                "user_turn_node_id": "turn-status-contract",
            }
            registry.create_run(**expected_ids, status=status)
            expected_assistant_turn_node_id = (
                "assistant-turn-status-contract" if status == "succeeded" else None
            )
            expected_result = {"answer": "done"} if status == "succeeded" else None
            expected_error = (
                {"code": "E_TEST", "message": "failed"}
                if status == "failed"
                else None
            )
            if status != "queued":
                registry.update_status(
                    expected_ids["run_id"],
                    status=status,
                    assistant_turn_node_id=expected_assistant_turn_node_id,
                    result=expected_result,
                    error=expected_error,
                    started=True,
                    finished=status in contract["terminal_statuses"],
                )
            stored = registry.get_run(expected_ids["run_id"])
            assert stored is not None
            response = client.get(f"/api/runs/{expected_ids['run_id']}", headers=headers)

            assert response.status_code == contract["success_status"]
            payload = response.json()
            assert set(contract["required_fields"]).issubset(payload)
            assert {key: payload[key] for key in expected_ids} == expected_ids
            assert payload["status"] == status
            assert payload["terminal"] is (status in contract["terminal_statuses"])
            assert payload["cancel_requested"] is False
            assert payload["assistant_turn_node_id"] == expected_assistant_turn_node_id
            assert payload["result"] == expected_result
            assert payload["error"] == expected_error
            assert isinstance(payload["created_at_ms"], int)
            assert isinstance(payload["updated_at_ms"], int)
            assert payload["started_at_ms"] == stored["started_at_ms"]
            assert payload["finished_at_ms"] == stored["finished_at_ms"]

            workflow_response = client.get(
                f"/api/workflow/runs/{expected_ids['run_id']}",
                headers=workflow_headers,
            )
            assert workflow_response.status_code == workflow_contract["success_status"]
            workflow_payload = workflow_response.json()
            assert set(workflow_contract["required_fields"]).issubset(workflow_payload)
            assert {key: workflow_payload[key] for key in expected_ids} == expected_ids
            assert workflow_payload["status"] == status
            assert workflow_payload["terminal"] is (
                status in contract["terminal_statuses"]
            )
            assert workflow_payload["cancel_requested"] is False
            assert workflow_payload["assistant_turn_node_id"] == (
                expected_assistant_turn_node_id
            )
            assert workflow_payload["result"] == expected_result
            assert workflow_payload["error"] == expected_error
            assert workflow_payload["started_at_ms"] == stored["started_at_ms"]
            assert workflow_payload["finished_at_ms"] == stored["finished_at_ms"]
            assert workflow_payload["created_at_ms"] == payload["created_at_ms"]
            assert workflow_payload["updated_at_ms"] == payload["updated_at_ms"]

        missing = client.get("/api/runs/missing-run-status", headers=headers)
        assert missing.status_code == 404

        registry_reads = 0

        def unexpected_registry_read(_run_id: str):
            nonlocal registry_reads
            registry_reads += 1
            raise AssertionError("denied run-status request reached registry")

        monkeypatch.setattr(
            registry.meta_store, "get_server_run", unexpected_registry_read
        )
        denied_headers = (
            _token_header(
                client,
                role="ro",
                ns="conversation",
                capabilities="workflow.run.read",
            ),
            _token_header(
                client,
                role="ro",
                ns="workflow",
                capabilities="project_view",
            ),
        )
        for headers in denied_headers:
            denied = client.get("/api/workflow/runs/private-run", headers=headers)
            assert denied.status_code == 403
        assert registry_reads == 0


def test_get_run_keeps_registry_status_even_if_workflow_cancelled_node_exists(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "run-terminal-cancelled"
    registry.create_run(
        run_id=run_id,
        conversation_id="conv-terminal",
        workflow_id="wf.cancelled",
        user_id="u1",
        user_turn_node_id="turn-1",
        status="running",
    )
    monkeypatch.setattr(
        service,
        "list_steps",
        lambda _run_id: (_ for _ in ()).throw(
            AssertionError("get_run() must not inspect workflow steps")
        ),
        raising=False,
    )
    _seed_terminal_node(
        conversation_engine,
        node_cls=WorkflowCancelledNode,
        entity_type="workflow_cancelled",
        run_id=run_id,
        workflow_id="wf.cancelled",
        conversation_id="conv-terminal",
    )

    run = service.get_run(run_id)
    assert run["status"] == "running"
    assert run["terminal"] is False
    assert "last_step_seq" not in run
    assert "step_count" not in run


def test_cancel_run_is_noop_when_run_already_terminal(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "run-terminal-noop"
    registry.create_run(
        run_id=run_id,
        conversation_id="conv-terminal",
        workflow_id="wf.completed",
        user_id="u1",
        user_turn_node_id="turn-1",
        status="succeeded",
    )

    result = service.cancel_run(run_id)
    assert result["status"] == "succeeded"
    assert result["terminal"] is True
    cancel_requests = conversation_engine.get_nodes(
        where={
            "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
        },
        limit=10,
    )
    assert cancel_requests == []


def test_runtime_design_rest_undo_redo(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        workflow_id = "wf.design.rest.undo_redo"
        designer_id = "tester"

        n1 = client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": f"wf|{workflow_id}|start",
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        )
        n1.raise_for_status()
        n2 = client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": f"wf|{workflow_id}|end",
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        )
        n2.raise_for_status()
        e1 = client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": f"wf|{workflow_id}|e|start_end",
                "src": f"wf|{workflow_id}|start",
                "dst": f"wf|{workflow_id}|end",
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        )
        e1.raise_for_status()

        hist = client.get(f"/api/workflow/design/{workflow_id}/history", headers=wf_ro)
        hist.raise_for_status()
        h = hist.json()
        assert h["current_version"] >= 3
        assert h["can_undo"] is True

        undone = client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        undone.raise_for_status()
        assert undone.json()["status"] in {"ok", "noop"}

        deleted_after_undo = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/wf|{workflow_id}|e|start_end",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert deleted_after_undo.status_code == 404

        redone = client.post(
            f"/api/workflow/design/{workflow_id}/redo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        redone.raise_for_status()
        assert redone.json()["status"] in {"ok", "noop"}

        deleted_after_redo = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/wf|{workflow_id}|e|start_end",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert deleted_after_redo.status_code == 200
        hist_after = client.get(
            f"/api/workflow/design/{workflow_id}/history", headers=wf_ro
        )
        hist_after.raise_for_status()
        timeline_ops = [
            str(item.get("op") or "") for item in hist_after.json().get("timeline", [])
        ]
        assert "UNDO_APPLIED" in timeline_ops
        assert "REDO_APPLIED" in timeline_ops


def test_runtime_design_delete_node_undo_redo_uses_delta_and_preserves_ids(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.delete_node_delta"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        deleted = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/nodes/{end_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        deleted.raise_for_status()
        assert deleted.json()["deleted"] is True
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (start_id, {start_id}, set())

        delta = workflow_engine.meta_sqlite.get_workflow_design_delta(
            workflow_id=workflow_id,
            version=4,
            schema_version=service._DELTA_SCHEMA_VERSION,
        )
        assert delta is not None
        forward = json.loads(str(delta["forward_json"]))
        inverse = json.loads(str(delta["inverse_json"]))
        assert forward["delete_node_ids"] == [end_id]
        assert forward["delete_edge_ids"] == [edge_id]
        assert [str(item["id"]) for item in inverse["upsert_nodes"]] == [end_id]
        assert [str(item["id"]) for item in inverse["upsert_edges"]] == [edge_id]

        def _unexpected_rebuild(*, workflow_id: str, state: dict[str, object]) -> None:
            raise AssertionError(
                f"unexpected rebuild for {workflow_id}: {state.get('current_version')}"
            )

        monkeypatch.setattr(
            service._workflow_design,
            "_workflow_rebuild_namespace_for_state",
            _unexpected_rebuild,
        )

        undone = client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        undone.raise_for_status()
        assert undone.json()["status"] == "ok"
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (
            start_id,
            {start_id, end_id},
            {edge_id},
        )
        _assert_workflow_design_entities_use_scoped_namespace(
            service,
            workflow_engine,
            workflow_id=workflow_id,
            node_ids={start_id, end_id},
            edge_ids={edge_id},
        )

        hist = client.get(f"/api/workflow/design/{workflow_id}/history", headers=wf_ro)
        hist.raise_for_status()
        assert hist.json()["current_version"] == 3
        assert hist.json()["can_redo"] is True

        redone = client.post(
            f"/api/workflow/design/{workflow_id}/redo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        redone.raise_for_status()
        assert redone.json()["status"] == "ok"
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (start_id, {start_id}, set())

        deleted_again = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/nodes/{end_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert deleted_again.status_code == 404


def test_runtime_design_delete_edge_undo_redo_uses_delta_and_preserves_ids(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.delete_edge_delta"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        deleted = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/{edge_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        deleted.raise_for_status()
        assert deleted.json()["deleted"] is True
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (
            start_id,
            {start_id, end_id},
            set(),
        )

        delta = workflow_engine.meta_sqlite.get_workflow_design_delta(
            workflow_id=workflow_id,
            version=4,
            schema_version=service._DELTA_SCHEMA_VERSION,
        )
        assert delta is not None
        forward = json.loads(str(delta["forward_json"]))
        inverse = json.loads(str(delta["inverse_json"]))
        assert forward["delete_node_ids"] == []
        assert forward["delete_edge_ids"] == [edge_id]
        assert inverse["upsert_nodes"] == []
        assert [str(item["id"]) for item in inverse["upsert_edges"]] == [edge_id]

        def _unexpected_rebuild(*, workflow_id: str, state: dict[str, object]) -> None:
            raise AssertionError(
                f"unexpected rebuild for {workflow_id}: {state.get('current_version')}"
            )

        monkeypatch.setattr(
            service._workflow_design,
            "_workflow_rebuild_namespace_for_state",
            _unexpected_rebuild,
        )

        undone = client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        undone.raise_for_status()
        assert undone.json()["status"] == "ok"
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (
            start_id,
            {start_id, end_id},
            {edge_id},
        )
        _assert_workflow_design_entities_use_scoped_namespace(
            service,
            workflow_engine,
            workflow_id=workflow_id,
            node_ids={start_id, end_id},
            edge_ids={edge_id},
        )

        hist = client.get(f"/api/workflow/design/{workflow_id}/history", headers=wf_ro)
        hist.raise_for_status()
        assert hist.json()["current_version"] == 3
        assert hist.json()["can_redo"] is True

        redone = client.post(
            f"/api/workflow/design/{workflow_id}/redo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        redone.raise_for_status()
        assert redone.json()["status"] == "ok"
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (
            start_id,
            {start_id, end_id},
            set(),
        )

        deleted_again = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/{edge_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert deleted_again.status_code == 404


def test_runtime_design_missing_delta_falls_back_to_rebuild(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.delta_fallback"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/{edge_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()

        workflow_engine.meta_sqlite.clear_workflow_design_deltas(
            workflow_id=workflow_id
        )
        rebuild_calls: list[int] = []
        original_rebuild = (
            service._workflow_design._workflow_rebuild_namespace_for_state
        )

        def _wrapped_rebuild(*, workflow_id: str, state: dict[str, object]) -> None:
            rebuild_calls.append(int(state.get("current_version") or -1))
            original_rebuild(workflow_id=workflow_id, state=state)

        monkeypatch.setattr(
            service._workflow_design,
            "_workflow_rebuild_namespace_for_state",
            _wrapped_rebuild,
        )

        undone = client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        undone.raise_for_status()
        assert undone.json()["status"] == "ok"
        assert rebuild_calls == [3]
        assert _visible_workflow_design_ids(
            workflow_engine, workflow_id=workflow_id
        ) == (
            start_id,
            {start_id, end_id},
            {edge_id},
        )


def test_runtime_design_undo_then_append_does_not_restore_discarded_branch(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        workflow_id = "wf.design.rest.undo_append_branch"
        designer_id = "tester"
        start_id = f"wf|{workflow_id}|start"
        end_id = f"wf|{workflow_id}|end"
        alt_id = f"wf|{workflow_id}|alt"
        edge_start_end = f"wf|{workflow_id}|e|start_end"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_start_end,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()

        # New edit after undo must truncate redo branch.
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": alt_id,
                "label": "Alt",
                "op": "noop",
            },
            headers=wf_rw,
        ).raise_for_status()

        hist = client.get(f"/api/workflow/design/{workflow_id}/history", headers=wf_ro)
        hist.raise_for_status()
        h = hist.json()
        assert h["can_redo"] is False
        timeline_ops = [str(item.get("op") or "") for item in h.get("timeline", [])]
        assert "BRANCH_DROPPED" in timeline_ops

        # Move back then forward on the surviving branch.
        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/redo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()

        # Discarded-branch edge must not reappear after redo on the new branch.
        delete_discarded = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/edges/{edge_start_end}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert delete_discarded.status_code == 404

        # New-branch node is present and deletable.
        delete_alt = client.request(
            "DELETE",
            f"/api/workflow/design/{workflow_id}/nodes/{alt_id}",
            json={"designer_id": designer_id},
            headers=wf_rw,
        )
        assert delete_alt.status_code == 200


def test_runtime_design_requires_designer_id_and_enforces_subject(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        workflow_id = "wf.design.rest.actor_validation"

        missing = client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={"node_id": "n1", "label": "Node", "op": "noop"},
            headers=wf_rw,
        )
        assert missing.status_code == 422

        wf_rw_alice = _token_header(client, role="rw", ns="workflow", username="alice")
        mismatch = client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={"designer_id": "bob", "node_id": "n1", "label": "Node", "op": "noop"},
            headers=wf_rw_alice,
        )
        assert mismatch.status_code == 403


def test_runtime_design_refresh_rebuilds_event_projection_and_uses_no_sidecars(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.refresh"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        designer_id = "tester"
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": f"wf|{workflow_id}|start",
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": f"wf|{workflow_id}|mid",
                "label": "Mid",
                "op": "noop",
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": f"wf|{workflow_id}|alt",
                "label": "Alt",
                "op": "noop",
            },
            headers=wf_rw,
        ).raise_for_status()

        refreshed = service.refresh_workflow_design_projection(workflow_id=workflow_id)
        assert refreshed["status"] == "ok"
        hist = client.get(f"/api/workflow/design/{workflow_id}/history", headers=wf_ro)
        hist.raise_for_status()
        payload = hist.json()
        assert [int(item["version"]) for item in payload["versions"]] == [0, 1, 3]
        assert "BRANCH_DROPPED" in [
            str(item.get("op") or "") for item in payload.get("timeline", [])
        ]
        assert not Path(
            workflow_engine.persist_directory, "workflow_design_history.sqlite"
        ).exists()
        assert not Path(
            workflow_engine.persist_directory, "server_runs.sqlite"
        ).exists()


def test_runtime_design_history_recreates_projection_and_tracks_field_semantics(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.history_fields"
    start_id = f"wf|{workflow_id}|start"
    old_id = f"wf|{workflow_id}|old"
    alt_id = f"wf|{workflow_id}|alt"
    edge_old_id = f"wf|{workflow_id}|e|start_old"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": old_id,
                "label": "Old",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_old_id,
                "src": start_id,
                "dst": old_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        initial = client.get(
            f"/api/workflow/design/{workflow_id}/history", headers=wf_ro
        )
        initial.raise_for_status()
        payload = initial.json()
        assert payload["current_version"] == 3
        assert payload["active_tip_version"] == 3
        assert payload["max_version"] == 3
        assert [int(item["version"]) for item in payload["versions"]] == [0, 1, 2, 3]

        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()

        undone = client.get(
            f"/api/workflow/design/{workflow_id}/history", headers=wf_ro
        )
        undone.raise_for_status()
        undone_payload = undone.json()
        assert undone_payload["current_version"] == 2
        assert undone_payload["active_tip_version"] == 3
        assert undone_payload["max_version"] == 3
        assert undone_payload["can_undo"] is True
        assert undone_payload["can_redo"] is True
        assert [int(item["version"]) for item in undone_payload["versions"]] == [
            0,
            1,
            2,
            3,
        ]
        assert [
            int(item["version"]) for item in undone_payload["selected_versions"]
        ] == [0, 1, 2]

        meta = workflow_engine.meta_sqlite
        meta.clear_workflow_design_projection(workflow_id=workflow_id)
        assert meta.get_workflow_design_projection(workflow_id=workflow_id) is None

        recreated = service.workflow_design_history(workflow_id=workflow_id)
        recreated_projection = meta.get_workflow_design_projection(
            workflow_id=workflow_id
        )
        assert recreated["current_version"] == 2
        assert recreated["active_tip_version"] == 3
        assert recreated_projection is not None
        assert recreated_projection["current_version"] == 2
        assert recreated_projection["active_tip_version"] == 3

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": alt_id,
                "label": "Alt",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        branched = client.get(
            f"/api/workflow/design/{workflow_id}/history", headers=wf_ro
        )
        branched.raise_for_status()
        branched_payload = branched.json()
        assert branched_payload["current_version"] == 4
        assert branched_payload["active_tip_version"] == 4
        assert branched_payload["max_version"] == 4
        assert branched_payload["can_redo"] is False
        assert [int(item["version"]) for item in branched_payload["versions"]] == [
            0,
            1,
            2,
            4,
        ]
        version_four = next(
            item
            for item in branched_payload["selected_versions"]
            if int(item["version"]) == 4
        )
        assert int(version_four["prev_version"]) == 2


def test_runtime_design_graph_reports_ready_materialization_status(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.graph_status"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        wf_ro = _token_header(client, role="ro", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        graph = client.get(
            f"/api/workflow/design/{workflow_id}/graph",
            params={"refresh": "true"},
            headers=wf_ro,
        )
        graph.raise_for_status()
        payload = graph.json()
        assert payload["materialization_status"] == "ready"
        assert payload["current_version"] == 3
        assert [
            str(node.get("metadata", {}).get("wf_op")) for node in payload["nodes"]
        ] == [
            "start",
            "end",
        ]


def test_runtime_design_projection_divergence_rows_are_replaced_from_history(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.projection_divergence"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

    state = service.workflow_design_history(workflow_id=workflow_id)
    meta = workflow_engine.meta_sqlite
    meta.replace_workflow_design_projection(
        workflow_id=workflow_id,
        head={
            "current_version": 999,
            "active_tip_version": 999,
            "last_authoritative_seq": int(state["latest_seq"]),
            "last_materialized_seq": 999,
            "projection_schema_version": service._PROJECTION_SCHEMA_VERSION,
            "snapshot_schema_version": service._SNAPSHOT_SCHEMA_VERSION,
            "materialization_status": "ready",
            "updated_at_ms": int(time.time() * 1000),
        },
        versions=[
            {"version": 999, "prev_version": 998, "target_seq": 999, "created_at_ms": 1}
        ],
        dropped_ranges=[
            {"start_seq": 700, "end_seq": 701, "start_version": 998, "end_version": 999}
        ],
    )

    repaired = service.workflow_design_history(workflow_id=workflow_id)
    projection = meta.get_workflow_design_projection(workflow_id=workflow_id)
    assert repaired["current_version"] == state["current_version"]
    assert repaired["active_tip_version"] == state["active_tip_version"]
    assert projection is not None
    assert projection["current_version"] == state["current_version"]
    assert projection["active_tip_version"] == state["active_tip_version"]
    assert 999 not in [int(item["version"]) for item in projection["versions"]]
    assert projection["dropped_ranges"] == []


def test_runtime_design_rebuilding_projection_blocks_mutations_and_runtime_submit(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_success_runner,
    )
    workflow_id = "wf.design.rest.rebuilding"
    workflow_engine.meta_sqlite.replace_workflow_design_projection(
        workflow_id=workflow_id,
        head={
            "current_version": 0,
            "active_tip_version": 0,
            "last_authoritative_seq": 0,
            "last_materialized_seq": 0,
            "projection_schema_version": 1,
            "snapshot_schema_version": 1,
            "materialization_status": "rebuilding",
            "updated_at_ms": int(time.time() * 1000),
        },
        versions=[],
        dropped_ranges=[],
    )

    with TestClient(server.app) as client:
        conv_rw = _token_header(client, role="rw", ns="conversation")
        wf_rw = _token_header(client, role="rw", ns="workflow")

        blocked_node = client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": "tester",
                "node_id": "n1",
                "label": "Node",
                "op": "noop",
            },
            headers=wf_rw,
        )
        assert blocked_node.status_code == 409
        assert "rebuilding" in blocked_node.json()["detail"]

        created = client.post(
            "/api/conversations", json={"user_id": "u-runtime-lock"}, headers=conv_rw
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        blocked_run = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "initial_state": {},
            },
            headers=wf_rw,
        )
        assert blocked_run.status_code == 409
        assert "rebuilding" in blocked_run.json()["detail"]


def test_runtime_loader_uses_only_active_branch_nodes_and_edges(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    workflow_id = "wf.design.rest.active_projection"
    start_id = f"wf|{workflow_id}|start"
    old_id = f"wf|{workflow_id}|old"
    alt_id = f"wf|{workflow_id}|alt"
    edge_old_id = f"wf|{workflow_id}|e|start_old"
    edge_alt_id = f"wf|{workflow_id}|e|start_alt"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": old_id,
                "label": "Old",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_old_id,
                "src": start_id,
                "dst": old_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": alt_id,
                "label": "Alt",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_alt_id,
                "src": start_id,
                "dst": alt_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()

    service.refresh_workflow_design_projection(workflow_id=workflow_id)
    start_seen, node_ids, edge_ids = _visible_workflow_design_ids(
        workflow_engine, workflow_id=workflow_id
    )
    assert start_seen == start_id
    assert node_ids == {start_id, alt_id}
    assert old_id not in node_ids
    assert edge_ids == {edge_alt_id}
    assert edge_old_id not in edge_ids


@pytest.mark.ci
def test_runtime_design_snapshot_restore_matches_full_replay(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    monkeypatch.setattr(service, "_SNAPSHOT_INTERVAL", 2, raising=False)
    workflow_id = "wf.design.rest.snapshot_restore"
    start_id = f"wf|{workflow_id}|start"
    end_id = f"wf|{workflow_id}|end"
    edge_id = f"wf|{workflow_id}|e|start_end"
    with TestClient(server.app) as client:
        wf_rw = _token_header(client, role="rw", ns="workflow")
        designer_id = "tester"

        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": start_id,
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        client.post(
            f"/api/workflow/design/{workflow_id}/nodes",
            json={
                "designer_id": designer_id,
                "node_id": end_id,
                "label": "End",
                "op": "end",
                "terminal": True,
            },
            headers=wf_rw,
        ).raise_for_status()

        snapshot = workflow_engine.meta_sqlite.get_workflow_design_snapshot(
            workflow_id=workflow_id,
            max_version=2,
            schema_version=service._SNAPSHOT_SCHEMA_VERSION,
        )
        assert snapshot is not None
        assert int(snapshot["version"]) == 2

        client.post(
            f"/api/workflow/design/{workflow_id}/edges",
            json={
                "designer_id": designer_id,
                "edge_id": edge_id,
                "src": start_id,
                "dst": end_id,
                "relation": "wf_next",
                "is_default": True,
            },
            headers=wf_rw,
        ).raise_for_status()
        workflow_engine.meta_sqlite.clear_workflow_design_deltas(
            workflow_id=workflow_id
        )

        snapshot_requested = {"called": False}
        original_get_snapshot = workflow_engine.meta_sqlite.get_workflow_design_snapshot

        def _wrapped_get_snapshot(*, workflow_id, max_version, schema_version):
            snapshot_requested["called"] = True
            return original_get_snapshot(
                workflow_id=workflow_id,
                max_version=max_version,
                schema_version=schema_version,
            )

        monkeypatch.setattr(
            workflow_engine.meta_sqlite,
            "get_workflow_design_snapshot",
            _wrapped_get_snapshot,
            raising=False,
        )

        client.post(
            f"/api/workflow/design/{workflow_id}/undo",
            json={"designer_id": designer_id},
            headers=wf_rw,
        ).raise_for_status()

    assert snapshot_requested["called"] is True
    snapshot_shape = _visible_workflow_design_ids(
        workflow_engine, workflow_id=workflow_id
    )
    assert snapshot_shape[1] == {start_id, end_id}
    assert snapshot_shape[2] == set()

    refreshed = service.refresh_workflow_design_projection(workflow_id=workflow_id)
    assert refreshed["status"] == "ok"
    assert (
        _visible_workflow_design_ids(workflow_engine, workflow_id=workflow_id)
        == snapshot_shape
    )
    _assert_workflow_design_entities_use_scoped_namespace(
        service,
        workflow_engine,
        workflow_id=workflow_id,
        node_ids={start_id, end_id},
        edge_ids=set(),
    )
