"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_server_health_matches_shared_contract() -> None:
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2] / "contracts/golden/health-v1.json"
        ).read_text(encoding="utf-8")
    )
    with TestClient(server.app) as client:
        response = client.get("/health")
    assert response.status_code == contract["success_status"]
    payload = response.json()
    assert payload["ok"] is contract["ok"]
    assert all(field in payload for field in contract["required_fields"])
    assert isinstance(payload["backend"], str)
    assert isinstance(payload["persist_directory"], str)
    assert isinstance(payload["conversation_persist_directory"], str)
    assert isinstance(payload["workflow_persist_directory"], str)
    assert isinstance(payload["wisdom_persist_directory"], str)
    assert payload["pg_schema_base"] is None or isinstance(
        payload["pg_schema_base"], str
    )


def test_document_upsert_tree_uses_document_graph_extraction_entrypoint(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-upsert-tree-success-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    request = fixture["request"]
    expected = fixture["expected"]
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )

    captured: dict[str, object] = {}

    def _fake_persist_document_graph_extraction(*, doc_id, parsed, mode="append"):
        captured["doc_id"] = doc_id
        captured["parsed_type"] = type(parsed).__name__
        captured["mode"] = mode
        captured["node_count"] = len(parsed.nodes)
        captured["edge_count"] = len(parsed.edges)
        return {
            "document_id": doc_id,
            "node_ids": [],
            "edge_ids": [],
            "nodes_added": 0,
            "edges_added": 0,
        }

    monkeypatch.setattr(
        engine,
        "persist_document_graph_extraction",
        _fake_persist_document_graph_extraction,
        raising=True,
    )

    with TestClient(server.app) as client:
        headers = token_header(
            client,
            role="rw",
            ns="docs,conversation,workflow",
            capabilities="",
        )
        resp = client.post(
            "/api/document.upsert_tree",
            json=request,
            headers=headers,
        )

    assert resp.status_code == expected["status_code"]
    assert resp.json() == expected["body"]
    assert captured == {
        "doc_id": request["doc_id"],
        "parsed_type": "GraphExtractionWithIDs",
        "mode": "append",
        "node_count": len(request["nodes"]),
        "edge_count": len(request["edges"]),
    }


def test_document_upsert_tree_requires_rw_even_with_write_graph_capability(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-upsert-tree-success-v1.json"
    )
    request = json.loads(fixture_path.read_text(encoding="utf-8"))["request"]
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    persistence_calls = 0

    def unexpected_persist(**_kwargs):
        nonlocal persistence_calls
        persistence_calls += 1
        raise AssertionError("read-only graph upsert reached persistence")

    monkeypatch.setattr(engine, "persist_document_graph_extraction", unexpected_persist)

    with TestClient(server.app) as client:
        headers = token_header(
            client, role="ro", ns="docs", capabilities="write_graph"
        )
        response = client.post(
            "/api/document.upsert_tree", json=request, headers=headers
        )

    assert response.status_code == 403
    assert persistence_calls == 0


def test_document_upsert_tree_accepts_nonempty_shared_graph_extraction(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "graph-extraction-v1.json"
    )
    extraction = json.loads(fixture_path.read_text(encoding="utf-8"))
    request = {
        key: extraction[key]
        for key in ("doc_id", "insertion_method", "nodes", "edges")
    }
    for entity in (*request["nodes"], *request["edges"]):
        local_id = entity.pop("local_id", None)
        if "id" not in entity and local_id is not None:
            entity["id"] = local_id
        if "mentions" not in entity and "references" in entity:
            spans = entity.pop("references")
            entity["mentions"] = [{"spans": spans}]
        for mention in entity.get("mentions", []):
            for span in mention.get("spans", []):
                span.setdefault("insertion_method", request["insertion_method"])
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    source_document = Document.from_text(
        extraction["content"],
        id=request["doc_id"],
        metadata={"insertion_method": request["insertion_method"]},
    )
    engine.write.add_document(source_document)

    with TestClient(server.app) as client:
        headers = token_header(client, role="rw", ns="docs,conversation,workflow")
        response = client.post("/api/document.upsert_tree", json=request, headers=headers)

    assert response.status_code == 200, response.text
    result = response.json()
    engine_result = result["engine_result"]
    assert result["status"] == "ok"
    assert result["inserted_nodes"] == 2
    assert result["inserted_edges"] == 1
    assert engine_result["document_id"] == request["doc_id"]
    assert engine_result["nodes_added"] == 2
    assert engine_result["edges_added"] == 1
    assert len(engine_result["node_ids"]) == 2
    assert len(engine_result["edge_ids"]) == 1
    persisted_nodes = engine.backend.node_get(
        ids=engine_result["node_ids"], include=["documents"]
    )
    persisted_edges = engine.backend.edge_get(
        ids=engine_result["edge_ids"], include=["documents"]
    )
    assert persisted_nodes["ids"] == engine_result["node_ids"]
    assert persisted_edges["ids"] == engine_result["edge_ids"]
    stored_nodes = [Node.model_validate_json(item) for item in persisted_nodes["documents"]]
    stored_edges = [Edge.model_validate_json(item) for item in persisted_edges["documents"]]
    stored_node_ids = {node.id for node in stored_nodes}
    assert all(
        span.doc_id == request["doc_id"]
        for node in stored_nodes
        for grounding in node.mentions
        for span in grounding.spans
    )
    assert all(
        endpoint_id in stored_node_ids
        for edge in stored_edges
        for endpoint_id in edge.source_ids + edge.target_ids
    )

    stable_request = json.loads(json.dumps(request))
    stable_nodes = [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ]
    stable_edge = "33333333-3333-4333-8333-333333333333"
    for index, entity_id in enumerate(stable_nodes):
        stable_request["nodes"][index]["id"] = entity_id
    stable_request["edges"][0]["id"] = stable_edge
    stable_request["edges"][0]["source_ids"] = [stable_nodes[0]]
    stable_request["edges"][0]["target_ids"] = [stable_nodes[1]]
    with TestClient(server.app) as client:
        owner_headers = token_header(
            client, role="rw", ns="docs,conversation,workflow", username="bob"
        )
        owner_response = client.post(
            "/api/document.upsert_tree", json=stable_request, headers=owner_headers
        )
        assert owner_response.status_code == 200, owner_response.text
        other_writer_headers = token_header(
            client, role="rw", ns="docs,conversation,workflow", username="alice"
        )
        external_request = json.loads(json.dumps(stable_request))
        external_node = "44444444-4444-4444-8444-444444444444"
        external_request["nodes"] = [external_request["nodes"][1]]
        external_request["nodes"][0]["id"] = external_node
        external_request["edges"][0]["id"] = "55555555-5555-4555-8555-555555555555"
        external_request["edges"][0]["source_ids"] = [stable_nodes[0]]
        external_request["edges"][0]["target_ids"] = [external_node]
        external_endpoint_response = client.post(
            "/api/document.upsert_tree",
            json=external_request,
            headers=other_writer_headers,
        )
        other_writer_response = client.post(
            "/api/document.upsert_tree",
            json=stable_request,
            headers=other_writer_headers,
        )
    assert other_writer_response.status_code == 200, other_writer_response.text
    assert external_endpoint_response.status_code == 200, external_endpoint_response.text


def test_document_validate_graph_matches_shared_success_contract(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-validate-graph-success-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )

    with TestClient(server.app) as client:
        response = client.post("/api/document.validate_graph", json=fixture["request"])

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]


def test_document_upsert_tree_storage_error_matches_shared_fixture(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "persistence-error-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    monkeypatch.setenv("AUTH_MODE", "production")

    def _fail_persistence(**_kwargs):
        raise OSError(fixture["message"])

    monkeypatch.setattr(
        engine, "persist_document_graph_extraction", _fail_persistence, raising=True
    )

    with TestClient(server.app) as client:
        headers = token_header(client, role="rw", ns="docs,conversation,workflow")
        response = client.post(
            "/api/document.upsert_tree",
            json={"doc_id": "doc::storage_error", "nodes": [], "edges": []},
            headers=headers,
        )

    assert response.status_code == fixture["python"]["status"]
    assert response.json()["detail"] == fixture["python"]["detail"]


def test_document_upsert_tree_generic_store_error_matches_shared_fixture(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "persistence-error-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    error_contract = fixture["generic_store_error"]
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    monkeypatch.setenv("AUTH_MODE", "production")

    def _fail_persistence(**_kwargs):
        raise RuntimeError(error_contract["message"])

    monkeypatch.setattr(
        engine, "persist_document_graph_extraction", _fail_persistence, raising=True
    )

    with TestClient(server.app) as client:
        headers = token_header(client, role="rw", ns="docs,conversation,workflow")
        response = client.post(
            "/api/document.upsert_tree",
            json={"doc_id": "doc::generic_storage_error", "nodes": [], "edges": []},
            headers=headers,
        )

    assert response.status_code == error_contract["python"]["status"]
    assert response.json()["detail"] == error_contract["python"]["detail"]


def test_document_upsert_tree_transaction_abort_matches_comparison_fixture(
    monkeypatch, engine_triplet
):
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "transaction-abort-error-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    monkeypatch.setenv("AUTH_MODE", "production")

    def _abort_transaction(**_kwargs):
        raise RuntimeError(fixture["message"])

    monkeypatch.setattr(
        engine, "persist_document_graph_extraction", _abort_transaction, raising=True
    )

    with TestClient(server.app) as client:
        headers = token_header(client, role="rw", ns="docs,conversation,workflow")
        response = client.post(
            "/api/document.upsert_tree",
            json={"doc_id": "doc::transaction_abort", "nodes": [], "edges": []},
            headers=headers,
        )

    assert fixture["parity"] is False
    assert response.status_code == fixture["python"]["status"]
    assert response.json()["detail"] == fixture["python"]["detail"]


def test_document_graph_upsert_result_serializes_persisted_result_shape():
    result = server.DocumentGraphUpsertResult(
        status="ok",
        inserted_nodes=1,
        inserted_edges=1,
        engine_result={
            "document_id": "doc::typed_result",
            "node_ids": ["node-a"],
            "edge_ids": ["edge-a"],
            "nodes_added": 1,
            "edges_added": 1,
        },
    )

    assert result.model_dump()["engine_result"] == {
        "document_id": "doc::typed_result",
        "node_ids": ["node-a"],
        "edge_ids": ["edge-a"],
        "nodes_added": 1,
        "edges_added": 1,
    }
