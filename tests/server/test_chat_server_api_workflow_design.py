"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_workflow_catalog_ops_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-catalog-ops-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        response = client.get("/api/workflow/catalog/ops", headers=headers)

    assert response.status_code == contract["success_status"], response.text
    assert response.json() == contract["response_body"]


def test_workflow_dead_letters_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letters-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        response = client.get(
            "/api/workflow/dead-letters",
            params={"limit": contract["query_limit"]},
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert response.json() == contract["response_body"]


def test_workflow_dead_letter_replay_missing_run_matches_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-dead-letter-replay-missing-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client, role="rw", ns="workflow", capabilities="service.manage"
        )
        response = client.post(
            f"/api/workflow/dead-letters/{contract['run_id']}/replay",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert response.json() == contract["response_body"]


def test_workflow_design_edge_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-upsert-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        for node in (
            {
                "node_id": "start",
                "label": "Start",
                "op": "start",
                "start": True,
            },
            {
                "node_id": "end",
                "label": "End",
                "op": "end",
                "terminal": True,
            },
        ):
            node_response = client.post(
                f"/api/workflow/design/{contract['workflow_id']}/nodes",
                json={"designer_id": contract["designer_id"], **node},
                headers=headers,
            )
            assert node_response.status_code == 200, node_response.text
        edge_path = f"/api/workflow/design/{contract['workflow_id']}/edges"
        edge_body = {
            "designer_id": contract["designer_id"],
            "edge_id": contract["edge_id"],
            "src": "start",
            "dst": "end",
            "is_default": True,
        }
        response = client.post(edge_path, json=edge_body, headers=headers)
        repeated_response = client.post(edge_path, json=edge_body, headers=headers)
        updated_response = client.post(
            edge_path,
            json={**edge_body, "relation": "wf_custom"},
            headers=headers,
        )
        graph = client.get(
            f"/api/workflow/design/{contract['workflow_id']}/graph",
            params={"refresh": "true"},
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    repeated_body = repeated_response.json()
    updated_body = updated_response.json()
    assert body["version"] == contract["version"]
    assert repeated_body["version"] == contract["version"] + 1
    assert updated_response.status_code == contract["success_status"]
    assert updated_body["version"] == contract["version"] + 2
    assert graph.status_code == 200, graph.text
    edges = graph.json()["edges"]
    assert len(edges) == 1
    assert edges[0]["id"] == contract["edge_id"]
    assert edges[0]["relation"] == "wf_custom"
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_design_edge_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-edge-delete-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        for node in (
            {
                "node_id": "start",
                "label": "Start",
                "op": "start",
                "start": True,
            },
            {
                "node_id": "end",
                "label": "End",
                "op": "end",
                "terminal": True,
            },
        ):
            node_response = client.post(
                f"/api/workflow/design/{contract['workflow_id']}/nodes",
                json={"designer_id": contract["designer_id"], **node},
                headers=headers,
            )
            assert node_response.status_code == 200, node_response.text
        edge_response = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/edges",
            json={
                "designer_id": contract["designer_id"],
                "edge_id": contract["edge_id"],
                "src": "start",
                "dst": "end",
                "is_default": True,
            },
            headers=headers,
        )
        assert edge_response.status_code == 200, edge_response.text
        response = client.request(
            "DELETE",
            f"/api/workflow/design/{contract['workflow_id']}/edges/"
            f"{contract['edge_id']}",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        repeated_delete = client.request(
            "DELETE",
            f"/api/workflow/design/{contract['workflow_id']}/edges/"
            f"{contract['edge_id']}",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        history = client.get(
            f"/api/workflow/design/{contract['workflow_id']}/history",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert repeated_delete.status_code == 404, repeated_delete.text
    body = response.json()
    mutation_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-mutation-events-v1.json"
        ).read_text(encoding="utf-8")
    )
    edge_tombstone = next(
        event
        for event in workflow_engine.meta_sqlite.iter_entity_events(
            namespace=f"wf_design:{contract['workflow_id']}", from_seq=1
        )
        if event[1] == "edge"
        and event[2] == contract["edge_id"]
        and event[3] == "TOMBSTONE"
    )
    assert json.loads(edge_tombstone[4]) == mutation_contract[
        "expected_edge_delete_events"
    ][0]["payload"]
    assert history.status_code == 200, history.text
    assert history.json()["current_version"] == body["version"]
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_design_empty_graph_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-empty-graph-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        response = client.get(
            f"/api/workflow/design/{contract['workflow_id']}/graph",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert response.json() == contract["response_body"]


def test_workflow_design_nonempty_graph_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-graph-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="rw", ns="workflow", username="alice")
        path = f"/api/workflow/design/{contract['workflow_id']}"
        for node in contract["node_requests"]:
            response = client.post(
                f"{path}/nodes",
                json={"designer_id": "alice", **node},
                headers=headers,
            )
            assert response.status_code == 200, response.text
        response = client.post(
            f"{path}/edges",
            json={"designer_id": "alice", **contract["edge_request"]},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        graph = client.get(
            f"{path}/graph",
            params={"refresh": str(contract["refresh"]).lower()},
            headers=headers,
        )
        deleted_edge = client.request(
            "DELETE",
            f"{path}/edges/{contract['delete_edge_request']['edge_id']}",
            json={"designer_id": "alice"},
            headers=headers,
        )
        graph_after_edge_delete = client.get(
            f"{path}/graph?refresh=true",
            headers=headers,
        )
        restored_edge = client.post(
            f"{path}/edges",
            json={"designer_id": "alice", **contract["edge_request"]},
            headers=headers,
        )
        deleted = client.request(
            "DELETE",
            f"{path}/nodes/{contract['delete_request']['node_id']}",
            json={"designer_id": "alice"},
            headers=headers,
        )
        graph_after_delete = client.get(
            f"{path}/graph?refresh=true",
            headers=headers,
        )

    assert graph.status_code == contract["success_status"], graph.text
    body = graph.json()
    expected = contract["expected"]
    for field in (
        "current_version",
        "active_tip_version",
        "can_undo",
        "can_redo",
    ):
        assert body[field] == expected[field]

    nodes = {node["id"]: node for node in body["nodes"]}
    assert len(nodes) == len(expected["nodes"])
    for item in expected["nodes"]:
        node = nodes[item["id"]]
        assert node["label"] == item["label"]
        for field in (
            "entity_type",
            "workflow_id",
            "wf_op",
            "wf_start",
            "wf_terminal",
            "wf_fanout",
        ):
            assert node["metadata"][field] == item[field]

    edges = {edge["id"]: edge for edge in body["edges"]}
    assert len(edges) == len(expected["edges"])
    for item in expected["edges"]:
        edge = edges[item["id"]]
        for field in ("source_ids", "target_ids", "relation"):
            assert edge[field] == item[field]
        for field in (
            "entity_type",
            "workflow_id",
            "wf_predicate",
            "wf_priority",
            "wf_is_default",
            "wf_multiplicity",
        ):
            assert edge["metadata"][field] == item[field]

    assert deleted_edge.status_code == contract["success_status"], deleted_edge.text
    assert graph_after_edge_delete.status_code == contract["success_status"]
    edge_deleted_body = graph_after_edge_delete.json()
    expected_edge_deleted = contract["expected_after_edge_delete"]
    for field in (
        "current_version",
        "active_tip_version",
        "can_undo",
        "can_redo",
    ):
        assert edge_deleted_body[field] == expected_edge_deleted[field]
    assert {node["id"] for node in edge_deleted_body["nodes"]} == set(
        expected_edge_deleted["node_ids"]
    )
    assert edge_deleted_body["edges"] == expected_edge_deleted["edges"]
    assert restored_edge.status_code == contract["success_status"], restored_edge.text
    assert deleted.status_code == contract["success_status"], deleted.text
    assert graph_after_delete.status_code == contract["success_status"]
    deleted_body = graph_after_delete.json()
    expected_deleted = contract["expected_after_delete"]
    for field in (
        "current_version",
        "active_tip_version",
        "can_undo",
        "can_redo",
    ):
        assert deleted_body[field] == expected_deleted[field]
    actual_deleted_nodes = {node["id"]: node for node in deleted_body["nodes"]}
    assert len(actual_deleted_nodes) == len(expected_deleted["nodes"])
    for item in expected_deleted["nodes"]:
        node = actual_deleted_nodes[item["id"]]
        assert node["label"] == item["label"]
        for field in (
            "entity_type",
            "workflow_id",
            "wf_op",
            "wf_start",
            "wf_terminal",
            "wf_fanout",
        ):
            assert node["metadata"][field] == item[field]
    assert deleted_body["edges"] == expected_deleted["edges"]


def test_workflow_design_empty_history_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    authority = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-rest-authority-v1.json"
        ).read_text(encoding="utf-8")
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-empty-history-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        for case in authority["cases"]:
            principal = case["principal"]
            claims = {**principal, "sub": "tester", "username": "tester"}
            token = jwt.encode(claims, TEST_JWT_SECRET, algorithm=TEST_JWT_ALG)
            response = client.get(
                authority["path"],
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == case["expected_status"], case["name"]
            if case["expected_status"] == contract["success_status"]:
                assert response.json() == contract["response_body"]


def test_workflow_design_node_upsert_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        response = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/nodes",
            json={
                "designer_id": contract["designer_id"],
                "node_id": contract["node_id"],
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_design_node_upsert_repeated_request_creates_next_python_version(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    body = {
        "designer_id": contract["designer_id"],
        "node_id": contract["node_id"],
        "label": "Start",
        "op": "start",
        "start": True,
    }
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        path = f"/api/workflow/design/{contract['workflow_id']}/nodes"
        first = client.post(path, json=body, headers=headers)
        second = client.post(path, json=body, headers=headers)
        updated = client.post(
            path, json={**body, "label": "Updated start"}, headers=headers
        )
        graph = client.get(
            f"/api/workflow/design/{contract['workflow_id']}/graph",
            params={"refresh": "true"},
            headers=headers,
        )

    assert first.status_code == second.status_code == contract["success_status"]
    assert first.json()["version"] == 1
    assert second.json()["version"] == 2
    assert updated.status_code == contract["success_status"], updated.text
    assert updated.json()["version"] == 3
    assert graph.status_code == 200, graph.text
    nodes = graph.json()["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["id"] == contract["node_id"]
    assert nodes[0]["label"] == "Updated start"


def test_workflow_design_node_upsert_rolls_back_graph_when_control_event_fails(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    design = service._workflow_design

    def fail_control_event(**_kwargs):
        raise RuntimeError("injected design-control failure")

    monkeypatch.setattr(design, "_append_design_control_event", fail_control_event)

    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        with pytest.raises(RuntimeError, match="injected design-control failure"):
            service.workflow_design_upsert_node(
                workflow_id="atomic-node-upsert",
                designer_id="alice",
                node_id="node-1",
                label="Must roll back",
            )
    finally:
        claims_ctx.reset(identity)

    graph = design.workflow_design_graph("atomic-node-upsert", refresh=True)
    history = design.workflow_design_history(workflow_id="atomic-node-upsert")
    assert graph["nodes"] == []
    assert graph["edges"] == []
    assert history["current_version"] == 0
    assert history["current_seq"] == 0


def test_workflow_design_edge_upsert_rolls_back_when_control_event_fails(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    design = service._workflow_design
    workflow_id = "atomic-edge-upsert"
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        for node_id, label in (("start", "Start"), ("end", "End")):
            service.workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id="alice",
                node_id=node_id,
                label=label,
            )
    finally:
        claims_ctx.reset(identity)

    history_before = design.workflow_design_history(workflow_id=workflow_id)
    original_append = design._append_design_control_event

    def fail_edge_control_event(**kwargs):
        if kwargs["payload"]["action"] == "edge_upsert":
            raise RuntimeError("injected edge design-control failure")
        return original_append(**kwargs)

    monkeypatch.setattr(design, "_append_design_control_event", fail_edge_control_event)
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        with pytest.raises(RuntimeError, match="injected edge design-control failure"):
            service.workflow_design_upsert_edge(
                workflow_id=workflow_id,
                designer_id="alice",
                edge_id="edge-1",
                src="start",
                dst="end",
            )
    finally:
        claims_ctx.reset(identity)

    graph = design.workflow_design_graph(workflow_id, refresh=True)
    history = design.workflow_design_history(workflow_id=workflow_id)
    assert {node["id"] for node in graph["nodes"]} == {"start", "end"}
    assert graph["edges"] == []
    assert history["current_version"] == history_before["current_version"]
    assert history["current_seq"] == history_before["current_seq"]


def test_workflow_design_node_delete_rolls_back_when_control_event_fails(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    design = service._workflow_design
    workflow_id = "atomic-node-delete"
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        for node_id, label in (("start", "Start"), ("end", "End")):
            service.workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id="alice",
                node_id=node_id,
                label=label,
            )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id="alice",
            edge_id="edge-1",
            src="start",
            dst="end",
        )
    finally:
        claims_ctx.reset(identity)

    graph_before = design.workflow_design_graph(workflow_id, refresh=True)
    history_before = design.workflow_design_history(workflow_id=workflow_id)
    original_append = design._append_design_control_event

    def fail_node_delete_control_event(**kwargs):
        if kwargs["payload"]["action"] == "node_delete":
            raise RuntimeError("injected node-delete design-control failure")
        return original_append(**kwargs)

    monkeypatch.setattr(
        design, "_append_design_control_event", fail_node_delete_control_event
    )
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        with pytest.raises(
            RuntimeError, match="injected node-delete design-control failure"
        ):
            service.workflow_design_delete_node(
                workflow_id=workflow_id,
                node_id="start",
                designer_id="alice",
            )
    finally:
        claims_ctx.reset(identity)

    graph_after = design.workflow_design_graph(workflow_id, refresh=True)
    history_after = design.workflow_design_history(workflow_id=workflow_id)
    assert {node["id"] for node in graph_after["nodes"]} == {
        node["id"] for node in graph_before["nodes"]
    }
    assert {edge["id"] for edge in graph_after["edges"]} == {
        edge["id"] for edge in graph_before["edges"]
    }
    assert history_after["current_version"] == history_before["current_version"]
    assert history_after["current_seq"] == history_before["current_seq"]


def test_workflow_design_edge_delete_rolls_back_when_control_event_fails(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    design = service._workflow_design
    workflow_id = "atomic-edge-delete"
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        for node_id, label in (("start", "Start"), ("end", "End")):
            service.workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id="alice",
                node_id=node_id,
                label=label,
            )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id="alice",
            edge_id="edge-1",
            src="start",
            dst="end",
        )
    finally:
        claims_ctx.reset(identity)

    graph_before = design.workflow_design_graph(workflow_id, refresh=True)
    history_before = design.workflow_design_history(workflow_id=workflow_id)
    original_append = design._append_design_control_event

    def fail_edge_delete_control_event(**kwargs):
        if kwargs["payload"]["action"] == "edge_delete":
            raise RuntimeError("injected edge-delete design-control failure")
        return original_append(**kwargs)

    monkeypatch.setattr(
        design, "_append_design_control_event", fail_edge_delete_control_event
    )
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        with pytest.raises(
            RuntimeError, match="injected edge-delete design-control failure"
        ):
            service.workflow_design_delete_edge(
                workflow_id=workflow_id,
                edge_id="edge-1",
                designer_id="alice",
            )
    finally:
        claims_ctx.reset(identity)

    graph_after = design.workflow_design_graph(workflow_id, refresh=True)
    history_after = design.workflow_design_history(workflow_id=workflow_id)
    assert {node["id"] for node in graph_after["nodes"]} == {
        node["id"] for node in graph_before["nodes"]
    }
    assert {edge["id"] for edge in graph_after["edges"]} == {
        edge["id"] for edge in graph_before["edges"]
    }
    assert history_after["current_version"] == history_before["current_version"]
    assert history_after["current_seq"] == history_before["current_seq"]


def test_workflow_design_mutations_match_shared_event_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-mutation-events-v1.json"
        ).read_text(encoding="utf-8")
    )
    workflow_id = fixture["workflow_id"]
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": "alice",
        }
    )
    try:
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id="alice",
            node_id="start",
            label="Start",
            op="start",
            start=True,
        )
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id="alice",
            node_id="end",
            label="End",
            op="emit-result",
            terminal=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id="alice",
            edge_id="edge",
            src="start",
            dst="end",
            is_default=True,
        )
        service.workflow_design_delete_edge(
            workflow_id=workflow_id,
            designer_id="alice",
            edge_id="edge",
        )
    finally:
        claims_ctx.reset(identity)

    events = list(
        workflow_engine.meta_sqlite.iter_entity_events(
            namespace=f"wf_design:{workflow_id}", from_seq=1
        )
    )
    actual = []
    edge_tombstone_seq = None
    for event in events:
        item = {
            "entity_kind": event[1],
            "entity_id": event[2],
            "op": event[3],
        }
        if event[1] == "design_control":
            payload = json.loads(event[4])
            if payload.get("action") == "edge_delete":
                assert edge_tombstone_seq is not None
                assert payload["target_seq"] == edge_tombstone_seq
            item["payload"] = {
                key: payload[key]
                for key in (
                    fixture["expected_events"]
                    + fixture["expected_edge_delete_events"]
                )[len(actual)]["payload"]
            }
        elif event[1] == "edge" and event[3] == "TOMBSTONE":
            edge_tombstone_seq = event[0]
            item["payload"] = json.loads(event[4])
        actual.append(item)

    assert actual == (
        fixture["expected_events"] + fixture["expected_edge_delete_events"]
    )


def test_workflow_design_node_upsert_requires_write_graph_before_effect(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-upsert-denied-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    calls = []
    monkeypatch.setattr(
        service._workflow_design,
        "workflow_design_upsert_node",
        lambda **kwargs: calls.append(kwargs) or {},
    )
    with TestClient(server.app) as client:
        for case in contract["denial_cases"]:
            claims = case["claims"]
            headers = _token_header(
                client,
                role=claims["role"],
                ns=claims["ns"],
                username=claims["sub"],
                capabilities=",".join(claims["capabilities"]),
            )
            response = client.post(case["path"], json=case["request"], headers=headers)
            assert response.status_code == case["status"], case["case"]

    assert calls == []


def test_workflow_design_node_delete_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-node-delete-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        created = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/nodes",
            json={
                "designer_id": contract["designer_id"],
                "node_id": contract["node_id"],
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=headers,
        )
        assert created.status_code == 200, created.text
        response = client.request(
            "DELETE",
            f"/api/workflow/design/{contract['workflow_id']}/nodes/"
            f"{contract['node_id']}",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        repeated_delete = client.request(
            "DELETE",
            f"/api/workflow/design/{contract['workflow_id']}/nodes/"
            f"{contract['node_id']}",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        history = client.get(
            f"/api/workflow/design/{contract['workflow_id']}/history",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert repeated_delete.status_code == 404, repeated_delete.text
    body = response.json()
    assert history.status_code == 200, history.text
    assert history.json()["current_version"] == body["version"]
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_design_redo_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-redo-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        created = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/nodes",
            json={
                "designer_id": contract["designer_id"],
                "node_id": contract["node_id"],
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=headers,
        )
        assert created.status_code == 200, created.text
        undone = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/undo",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        assert undone.status_code == 200, undone.text
        response = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/redo",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        repeated = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/redo",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    repeated_body = repeated.json()
    assert repeated.status_code == contract["success_status"]
    assert repeated_body["status"] == "noop"
    assert repeated_body["current_version"] == body["current_version"]
    assert repeated_body["current_seq"] == body["current_seq"]
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field, expected_length in contract["array_lengths"].items():
        assert len(body[field]) == expected_length
    assert body["commits"]["1"]["action"] == contract["version_one_action"]
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_design_undo_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-design-undo-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=contract["designer_id"],
        )
        created = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/nodes",
            json={
                "designer_id": contract["designer_id"],
                "node_id": contract["node_id"],
                "label": "Start",
                "op": "start",
                "start": True,
            },
            headers=headers,
        )
        assert created.status_code == 200, created.text
        response = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/undo",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )
        repeated = client.post(
            f"/api/workflow/design/{contract['workflow_id']}/undo",
            json={"designer_id": contract["designer_id"]},
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    repeated_body = repeated.json()
    assert repeated.status_code == contract["success_status"]
    assert repeated_body["status"] == "noop"
    assert repeated_body["current_version"] == body["current_version"]
    assert repeated_body["current_seq"] == body["current_seq"]
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field, expected_length in contract["array_lengths"].items():
        assert len(body[field]) == expected_length
    assert body["commits"]["1"]["action"] == contract["version_one_action"]
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_workflow_lane_progress_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-lane-progress-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    registry.create_run(
        run_id=contract["run_id"],
        conversation_id="lane-progress-contract-conversation",
        workflow_id="lane-progress-contract-workflow",
        user_id="alice",
        user_turn_node_id="lane-progress-contract-turn",
        status="running",
    )
    registry.append_event(
        contract["run_id"], contract["event_type"], contract["payload"]
    )
    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        response = client.get(
            f"/api/workflow/lane/progress?run_id={contract['run_id']}",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    assert body["total"] == contract["expected_total"]
    assert len(body["items"]) == contract["expected_item_count"]
    item = body["items"][0]
    assert set(item) == set(contract["expected_item_keys"])
    for field, expected in contract["expected_item_fields"].items():
        assert item[field] == expected
    assert isinstance(item["created_at_ms"], int)


def test_workflow_repair_orphans_matches_shared_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-repair-orphans-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    meta_store = conversation_engine.meta_sqlite
    claims_token = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "storage_ns": "workflow",
            "security_scope": "tenant/demo",
        }
    )
    try:
        for message_id, inbox_id in (
            ("expired-orphan", contract["inbox_id"]),
            (contract["active_message_id"], "repair-active"),
        ):
            meta_store.project_lane_message(
                message_id=message_id,
                namespace="workflow",
                purpose="system",
                inbox_id=inbox_id,
                conversation_id="repair-contract-conversation",
                recipient_id="python-worker",
                sender_id="rust-scheduler",
                msg_type="workflow.run.execute",
                status="pending",
                created_at=1,
                available_at=0,
                run_id=f"run-{message_id}",
                step_id="start",
                correlation_id=f"corr-{message_id}",
                payload_json="{}",
            )
            claimed = meta_store.claim_projected_lane_messages(
                namespace="workflow",
                inbox_id=inbox_id,
                claimed_by="repair-contract-worker",
                limit=1,
                lease_seconds=(
                    -1 if message_id == "expired-orphan" else 60
                ),
                message_ids=[message_id],
            )
            assert [row.message_id for row in claimed] == [message_id]
    finally:
        claims_ctx.reset(claims_token)

    with TestClient(server.app) as client:
        headers = _token_header(client, role="rw", ns="workflow")
        response = client.post(
            "/api/workflow/messages/repair-orphans"
            f"?inbox_id={contract['inbox_id']}&limit={contract['limit']}",
            headers=headers,
        )

    assert response.status_code == contract["success_status"], response.text
    assert response.json() == {
        "repaired_message_ids": contract["expected_repaired_message_ids"]
    }
    active_rows = meta_store.list_projected_lane_messages(
        namespace="workflow",
        inbox_id="repair-active",
        status="claimed",
        limit=10,
    )
    assert [row.message_id for row in active_rows] == [
        contract["active_message_id"]
    ]
