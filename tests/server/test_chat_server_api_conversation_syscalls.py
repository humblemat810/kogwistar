"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_mount_memory_syscall_uses_python_backend(monkeypatch, tmp_path, backend):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"mount-memory-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        with TestClient(server.app) as client:
            owner_headers = _token_header(
                client, role="rw", ns="conversation", username="memory-owner"
            )
            created = client.post(
                "/api/conversations",
                json={"user_id": "memory-owner"},
                headers=owner_headers,
            )
            created.raise_for_status()
            conversation_id = created.json()["conversation_id"]
            service = ConversationService.from_engine(
                conversation_engine,
                knowledge_engine=engine,
                workflow_engine=workflow_engine,
            )
            snapshot_id = service.persist_context_snapshot(
                conversation_id=conversation_id,
                run_id="memory-run",
                run_step_seq=0,
                stage="draft_answer",
                view=SimpleNamespace(
                    messages=[{"role": "user", "content": "memory snapshot"}],
                    items=[],
                    cost=None,
                    token_budget=32,
                ),
                model_name="fake-model",
                budget_tokens=32,
            )
            conversation_engine.acl.record_acl(
                grain="node",
                truth_graph="conversation",
                entity_id=snapshot_id,
                version=2,
                mode="private",
                owner_id="memory-owner",
                supersedes_version=1,
            )
            request = {
                "version": "v1",
                "op": "mount_memory",
                "args": {
                    "conversation_id": conversation_id,
                    "run_id": "memory-run",
                    "stage": "draft_answer",
                },
            }
            mounted = client.post(
                "/api/syscall/v1/mount_memory",
                json=request,
                headers=owner_headers,
            )
            assert mounted.status_code == 200, mounted.text
            assert mounted.json()["result"]["snapshot_node_id"] == snapshot_id

            missing_snapshot_request = {
                **request,
                "args": {**request["args"], "run_id": "missing-memory-run"},
            }
            missing_snapshot = client.post(
                "/api/syscall/v1/mount_memory",
                json=missing_snapshot_request,
                headers=owner_headers,
            )
            assert missing_snapshot.status_code == 404, missing_snapshot.text
            assert snapshot_id not in missing_snapshot.text

            outsider = client.post(
                "/api/syscall/v1/mount_memory",
                json=request,
                headers=_token_header(
                    client,
                    role="ro",
                    ns="conversation",
                    username="unrelated-reader",
                ),
            )
            assert outsider.status_code == 404
            assert snapshot_id not in outsider.text
            assert "memory snapshot" not in outsider.text
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_project_view_syscall_uses_python_backend(monkeypatch, tmp_path, backend):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"project-view-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-resources-http-v1.json"
            ).read_text(encoding="utf-8")
        )
        with TestClient(server.app) as client:
            headers = _token_header(
                client,
                role="ro",
                ns="workflow",
                capabilities=",".join(contract["required_capabilities"]),
            )
            response = client.post(
                "/api/syscall/v1/project_view",
                json={"version": "v1", "op": "project_view", "args": {}},
                headers=headers,
            )

        assert response.status_code == 200, response.text
        payload = response.json()["result"]
        assert all(field in payload for field in contract["required_fields"])
        for key, required_fields in contract["required_object_fields"].items():
            assert all(field in payload[key] for field in required_fields), key
    finally:
        for engine in engines:
            engine.close()


def test_create_conversation_syscall_uses_python_in_memory_backend(
    monkeypatch, tmp_path
):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / "create-conversation-in-memory",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=build_in_memory_backend,
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        now = int(time.time())
        token = jwt.encode(
            {
                "sub": "memory-owner",
                "user_id": "memory-owner",
                "role": "rw",
                "ns": ["conversation"],
                "capabilities": ["spawn_process"],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        headers = {"Authorization": f"Bearer {token}"}
        expected = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/conversation-create-syscall-result-v1.json"
            ).read_text(encoding="utf-8")
        )
        expected.update(
            conversation_id="memory-conversation",
            user_id="memory-owner",
            start_node_id="memory-conversation-start",
            tail_node_id="memory-conversation-start",
        )
        with TestClient(server.app) as client:
            request = {
                "version": "v1",
                "op": "create_conversation",
                "args": {
                    "user_id": "impostor",
                    "conversation_id": "memory-conversation",
                    "start_node_id": "memory-conversation-start",
                },
            }
            denied = client.post(
                "/api/syscall/v1/create_conversation", json=request, headers=headers
            )
            assert denied.status_code == 403

            request["args"]["user_id"] = "memory-owner"
            created = client.post(
                "/api/syscall/v1/create_conversation", json=request, headers=headers
            )
            assert created.status_code == 200, created.text
            payload = created.json()
            assert payload["version"] == "v1"
            assert payload["op"] == "create_conversation"
            assert payload["status"] == "ok"
            assert payload["result"] == expected
    finally:
        for engine in engines:
            engine.close()


def test_receive_message_syscall_uses_python_in_memory_backend(
    monkeypatch, tmp_path
):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / "receive-message-in-memory",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=build_in_memory_backend,
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/receive-message-result-v1.json"
            ).read_text(encoding="utf-8")
        )
        for message in fixture["messages"]:
            content = message["content"]
            embedding = conversation_engine.iterative_defensive_emb(content)
            conversation_engine.write.add_node(
                ConversationNode(
                    id=message["node_id"],
                    label=f"Turn {message['turn_index']}",
                    type="entity",
                    summary=content,
                    mentions=[
                        Grounding(spans=[Span.from_dummy_for_conversation("c1")])
                    ],
                    doc_id="conv:c1",
                    metadata={
                        "entity_type": message["entity_type"],
                        "in_conversation_chain": True,
                        "level_from_root": 0,
                    },
                    role=message["role"],
                    turn_index=message["turn_index"],
                    conversation_id=fixture["conversation_id"],
                    user_id="reader-1",
                    embedding=embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else embedding,
                )
            )
        empty_embedding = conversation_engine.iterative_defensive_emb("snapshot")
        conversation_engine.write.add_node(
            ConversationNode(
                id="receive-empty-snapshot",
                label="Empty conversation snapshot",
                type="entity",
                summary="snapshot",
                mentions=[
                    Grounding(spans=[Span.from_dummy_for_conversation("c-empty")])
                ],
                doc_id="conv:c-empty",
                metadata={
                    "entity_type": "context_snapshot",
                    "in_conversation_chain": True,
                    "level_from_root": 0,
                },
                conversation_id="c-empty",
                user_id="reader-1",
                embedding=empty_embedding.tolist()
                if hasattr(empty_embedding, "tolist")
                else empty_embedding,
            )
        )

        with TestClient(server.app) as client:
            headers = _token_header(client, role="ro", ns="conversation")
            response = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": fixture["conversation_id"]},
                },
                headers=headers,
            )
            unknown = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "c-unknown"},
                },
                headers=headers,
            )
            empty = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "c-empty"},
                },
                headers=headers,
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["version"] == "v1"
        assert payload["op"] == "receive_message"
        assert payload["status"] == "ok"
        assert payload["result"] == fixture
        assert unknown.status_code == 404, unknown.text
        assert empty.status_code == 200, empty.text
        assert empty.json()["result"]["messages"] == []
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_request_approval_syscall_uses_python_backend(monkeypatch, tmp_path, backend):
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/syscall-request-approval-success-v1.json"
        ).read_text(encoding="utf-8")
    )
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"request-approval-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        with TestClient(server.app) as client:
            headers = _token_header(
                client,
                role="rw",
                ns="workflow",
                capabilities="approve_action,project_view,read_security_scope",
            )
            for case in contract["cases"]:
                response = client.post(
                    "/api/syscall/v1/request_approval",
                    json=case["request"],
                    headers=headers,
                )
                assert response.status_code == case["http_status"], response.text
                payload = response.json()
                assert payload["version"] == case["request"]["version"]
                assert payload["op"] == case["request"]["op"]
                assert payload["status"] == "ok"
                assert payload["error"] is None
                result = payload["result"]
                assert result["status"] == case["result_status"], case["name"]
                entries = result[case["snapshot_field"]][case["entries_field"]]
                expected_entry = case["expected_entry"]
                subject_entries = [
                    entry
                    for entry in entries
                    if entry.get("subject") == expected_entry["subject"]
                ]
                assert subject_entries == [expected_entry], case["name"]

            unrecognized_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-request-approval-unrecognized-v1.json"
                ).read_text(encoding="utf-8")
            )
            unrecognized = client.post(
                "/api/syscall/v1/request_approval",
                json=unrecognized_contract["request"],
                headers=headers,
            )
            assert unrecognized.status_code == unrecognized_contract["http_status"]
            assert unrecognized.json() == unrecognized_contract["response"]

            denied = client.post(
                "/api/syscall/v1/request_approval",
                json=contract["cases"][0]["request"],
                headers=_token_header(
                    client, role="rw", ns="workflow", capabilities="project_view"
                ),
            )
            assert denied.status_code == 403

    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_send_message_syscall_uses_python_backend(monkeypatch, tmp_path, backend):
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/send-message-syscall-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"send-message-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        service, registry = _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        monkeypatch.setattr(service._run_execution, "_run_answer", lambda _request: None)
        with TestClient(server.app) as client:
            headers = _token_header(
                client,
                role="rw",
                ns="conversation",
                username="memory-send-owner",
                capabilities="send_message,spawn_process,read_graph",
            )
            created = client.post(
                "/api/conversations",
                json={"user_id": "memory-send-owner"},
                headers=headers,
            )
            assert created.status_code == 200, created.text
            conversation_id = created.json()["conversation_id"]
            request = {
                "version": "v1",
                "op": "send_message",
                "args": {
                    "conversation_id": conversation_id,
                    "text": "hello from in-memory backend",
                    "idempotency_key": "memory-send-once",
                },
            }
            first = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert first.status_code == contract["http_status"], first.text
            first_payload = first.json()
            for key, value in contract["envelope"].items():
                assert first_payload[key] == value
            first_result = first_payload["result"]
            assert set(contract["common_result_fields"]).issubset(first_result)
            for key, value in contract["common_result_values"].items():
                assert first_result[key] == value
            replay = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert replay.status_code == contract["http_status"], replay.text
            assert replay.json() == first_payload

            request["args"]["text"] = "changed payload"
            conflict = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert conflict.status_code == contract["conflict_status"]

        result = first_payload["result"]
        assert result["status"] == "queued"
        assert registry.get_run(result["run_id"])["user_turn_node_id"] == result[
            "user_turn_node_id"
        ]
        assert len(service.list_transcript(conversation_id)) == 1
    finally:
        for engine in engines:
            engine.close()


def test_conversation_syscalls_use_python_postgres_backend(
    monkeypatch, tmp_path, pg_dsn, pg_schema
):
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture is unavailable")

    import sqlalchemy as sa

    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    schema_names = [f"{pg_schema}_{graph_type}" for graph_type in ("knowledge", "conversation", "workflow")]
    postgres_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)

    def backend_factory(engine):
        return PgVectorBackend(
            engine=postgres_engine,
            embedding_dim=384,
            schema=f"{pg_schema}_{engine.kg_graph_type}",
        )

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / "send-message-postgres",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=backend_factory,
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        service, registry = _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        monkeypatch.setattr(service._run_execution, "_run_answer", lambda _request: None)
        now = int(time.time())
        token = jwt.encode(
            {
                "sub": "alice",
                "user_id": "alice",
                "role": "rw",
                "ns": ["conversation", "workflow"],
                "capabilities": [
                    "send_message",
                    "create_conversation",
                    "spawn_process",
                    "read_graph",
                    "invoke_tool",
                    "project_view",
                    "approve_action",
                    "read_security_scope",
                    "workflow.run.write",
                    "workflow.run.read",
                    "workflow.design.write",
                    "write_graph",
                ],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        headers = {"Authorization": f"Bearer {token}"}
        denied_termination_token = jwt.encode(
            {
                "sub": "alice",
                "user_id": "alice",
                "role": "rw",
                "ns": ["workflow"],
                "capabilities": ["workflow.run.read"],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        denied_termination_headers = {
            "Authorization": f"Bearer {denied_termination_token}"
        }
        workflow_id = "wf.postgres-syscall-lifecycle"
        identity = claims_ctx.set(
            {
                "ns": "workflow",
                "role": "rw",
                "capabilities": [
                    "spawn_process",
                    "workflow.run.write",
                    "workflow.run.read",
                    "workflow.design.write",
                    "write_graph",
                ],
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
                op="end",
                terminal=True,
            )
            service.workflow_design_upsert_edge(
                workflow_id=workflow_id,
                designer_id="alice",
                edge_id="edge-start-end",
                src="start",
                dst="end",
                relation="wf_next",
                is_default=True,
            )
        finally:
            claims_ctx.reset(identity)
        with TestClient(server.app) as client:
            created = client.post(
                "/api/syscall/v1/create_conversation",
                json={
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {
                        "conversation_id": "conv-create-test",
                        "user_id": "alice",
                        "start_node_id": "conv-create-start",
                    },
                },
                headers=headers,
            )
            assert created.status_code == 200, created.text
            create_fixture = json.loads(
                (Path(__file__).resolve().parents[2] / "contracts/golden/conversation-create-syscall-result-v1.json").read_text(encoding="utf-8")
            )
            send_fixture = json.loads(
                (Path(__file__).resolve().parents[2] / "contracts/golden/send-message-syscall-result-v1.json").read_text(encoding="utf-8")
            )
            assert created.json()["result"] == create_fixture
            conversation_id = created.json()["result"]["conversation_id"]
            request = {
                "version": "v1",
                "op": "send_message",
                "args": {
                    "conversation_id": conversation_id,
                    "text": "hello from Python PostgreSQL backend",
                    "idempotency_key": "postgres-send-once",
                },
            }
            first = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert first.status_code == send_fixture["http_status"], first.text
            first_payload = first.json()
            for key, value in send_fixture["envelope"].items():
                assert first_payload[key] == value
            first_result = first_payload["result"]
            assert set(send_fixture["common_result_fields"]).issubset(first_result)
            for key, value in send_fixture["common_result_values"].items():
                assert first_result[key] == value
            replay = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert replay.status_code == send_fixture["http_status"], replay.text
            assert replay.json() == first_payload

            request["args"]["text"] = "changed payload"
            conflict = client.post(
                "/api/syscall/v1/send_message", json=request, headers=headers
            )
            assert conflict.status_code == send_fixture["conflict_status"]

            received = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": conversation_id},
                },
                headers=headers,
            )
            assert received.status_code == 200, received.text
            received_payload = received.json()
            assert received_payload["version"] == "v1"
            assert received_payload["op"] == "receive_message"
            assert received_payload["status"] == "ok"
            received_messages = received_payload["result"]["messages"]
            assert len(received_messages) == 1
            received_turn = received_messages[0]
            assert received_turn["node_id"] == first.json()["result"][
                "user_turn_node_id"
            ]
            assert received_turn["role"] == "user"
            assert received_turn["entity_type"] == "conversation_turn"
            assert received_turn["content"] == "hello from Python PostgreSQL backend"

            empty_created = client.post(
                "/api/syscall/v1/create_conversation",
                json={
                    "version": "v1",
                    "op": "create_conversation",
                    "args": {
                        "conversation_id": "conv-empty-test",
                        "user_id": "alice",
                    },
                },
                headers=headers,
            )
            assert empty_created.status_code == 200, empty_created.text
            empty_received = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "conv-empty-test"},
                },
                headers=headers,
            )
            assert empty_received.status_code == 200, empty_received.text
            assert empty_received.json()["result"]["messages"] == []
            unknown_received = client.post(
                "/api/syscall/v1/receive_message",
                json={
                    "version": "v1",
                    "op": "receive_message",
                    "args": {"conversation_id": "conv-unknown-test"},
                },
                headers=headers,
            )
            assert unknown_received.status_code == 404, unknown_received.text

            conversation_service = ConversationService.from_engine(
                conversation_engine,
                knowledge_engine=engine,
                workflow_engine=workflow_engine,
            )
            snapshot_id = conversation_service.persist_context_snapshot(
                conversation_id=conversation_id,
                run_id="postgres-memory-run",
                run_step_seq=0,
                stage="final",
                view=SimpleNamespace(
                    messages=[{"role": "user", "content": "postgres snapshot"}],
                    items=[],
                    cost=None,
                    token_budget=32,
                ),
                model_name="fake-model",
                budget_tokens=32,
            )
            conversation_engine.acl.record_acl(
                grain="node",
                truth_graph="conversation",
                entity_id=snapshot_id,
                version=2,
                mode="private",
                owner_id="alice",
                supersedes_version=1,
            )
            mounted = client.post(
                "/api/syscall/v1/mount_memory",
                json={
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": conversation_id,
                        "run_id": "postgres-memory-run",
                        "stage": "final",
                    },
                },
                headers=headers,
            )
            assert mounted.status_code == 200, mounted.text
            assert mounted.json()["result"]["snapshot_node_id"] == snapshot_id

            missing_snapshot = client.post(
                "/api/syscall/v1/mount_memory",
                json={
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": conversation_id,
                        "run_id": "missing-postgres-run",
                        "stage": "final",
                    },
                },
                headers=headers,
            )
            assert missing_snapshot.status_code == 404, missing_snapshot.text
            assert snapshot_id not in missing_snapshot.text

            invoked = client.post(
                "/api/syscall/v1/invoke_tool",
                json={
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "latest_snapshot",
                        "conversation_id": conversation_id,
                        "run_id": "postgres-memory-run",
                        "stage": "final",
                    },
                },
                headers=headers,
            )
            assert invoked.status_code == 200, invoked.text
            assert invoked.json()["result"]["snapshot"]["snapshot_node_id"] == snapshot_id

            invoke_error_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-invoke-tool-errors-v1.json"
                ).read_text(encoding="utf-8")
            )
            graph_ids_before_invoke_errors = {
                node.id
                for node in conversation_engine.read.get_nodes(
                    where={"conversation_id": conversation_id},
                    node_type=ConversationNode,
                    limit=1_000,
                )
            }
            for case in invoke_error_contract["cases"]:
                error_response = client.post(
                    "/api/syscall/v1/invoke_tool",
                    json=case["request"],
                    headers=headers,
                )
                assert error_response.status_code == case["http_status"], case["name"]
                assert error_response.json() == case["response"], case["name"]
            graph_ids_after_invoke_errors = {
                node.id
                for node in conversation_engine.read.get_nodes(
                    where={"conversation_id": conversation_id},
                    node_type=ConversationNode,
                    limit=1_000,
                )
            }
            assert graph_ids_after_invoke_errors == graph_ids_before_invoke_errors

            outsider = client.post(
                "/api/syscall/v1/mount_memory",
                json={
                    "version": "v1",
                    "op": "mount_memory",
                    "args": {
                        "conversation_id": conversation_id,
                        "run_id": "postgres-memory-run",
                        "stage": "final",
                    },
                },
                headers=_token_header(
                    client, role="ro", ns="conversation", username="outsider"
                ),
            )
            assert outsider.status_code == 404
            assert snapshot_id not in outsider.text
            assert "postgres snapshot" not in outsider.text

            approval_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-request-approval-success-v1.json"
                ).read_text(encoding="utf-8")
            )
            approval_headers = _token_header(
                client,
                role="rw",
                ns="workflow",
                username="alice",
                capabilities="approve_action,project_view,read_security_scope",
            )
            for case in approval_contract["cases"]:
                response = client.post(
                    "/api/syscall/v1/request_approval",
                    json=case["request"],
                    headers=approval_headers,
                )
                assert response.status_code == case["http_status"], response.text
                result = response.json()["result"]
                assert result["status"] == case["result_status"]
                assert case["expected_entry"] in result[case["snapshot_field"]][
                    case["entries_field"]
                ]

            denied_approval = client.post(
                "/api/syscall/v1/request_approval",
                json={
                    "version": "v1",
                    "op": "request_approval",
                    "args": {"action": "deny", "reason": "postgres deny branch"},
                },
                headers=approval_headers,
            )
            assert denied_approval.status_code == 200, denied_approval.text
            assert denied_approval.json()["status"] == "blocked"
            assert denied_approval.json()["error"]["reason"] == "postgres deny branch"

            invalid_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-request-approval-invalid-v1.json"
                ).read_text(encoding="utf-8")
            )
            for case in invalid_contract["cases"]:
                invalid_response = client.post(
                    "/api/syscall/v1/request_approval",
                    json=case["request"],
                    headers=approval_headers,
                )
                assert invalid_response.status_code == case["http_status"], case["name"]
                assert invalid_response.json() == case["response"], case["name"]

            spawned = client.post(
                "/api/syscall/v1/spawn_process",
                json={
                    "version": "v1",
                    "op": "spawn_process",
                    "args": {
                        "workflow_id": workflow_id,
                        "conversation_id": conversation_id,
                        "user_id": "alice",
                        "initial_state": {"source": "postgres-syscall"},
                    },
                },
                headers=headers,
            )
            assert spawned.status_code == 200, spawned.text
            spawned_result = spawned.json()["result"]
            submit_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/workflow-run-submit-http-v1.json"
                ).read_text(encoding="utf-8")
            )
            assert all(
                field in spawned_result
                for field in submit_contract["required_common_fields"]
            )
            for field, value in submit_contract["expected_common_values"].items():
                assert spawned_result[field] == value
            run_id = spawned_result["run_id"]
            assert registry.get_run(run_id) is not None
            terminate_errors = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-terminate-errors-v1.json"
                ).read_text(encoding="utf-8")
            )
            for args, expected_status in (
                ({}, terminate_errors["missing_run_id_status"]),
                ({"run_id": f"{run_id}-unknown"}, terminate_errors["unknown_run_id_status"]),
            ):
                invalid_termination = client.post(
                    "/api/syscall/v1/terminate_process",
                    json={"version": "v1", "op": "terminate_process", "args": args},
                    headers=headers,
                )
                assert invalid_termination.status_code == expected_status
            assert registry.get_run(run_id)["cancel_requested"] is False
            resume_errors = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-resume-errors-v1.json"
                ).read_text(encoding="utf-8")
            )
            for args, expected_status in (
                ({}, resume_errors["missing_run_id_status"]),
                (
                    {
                        "run_id": f"{run_id}-unknown",
                        "workflow_id": "unknown-workflow",
                        "conversation_id": "unknown-conversation",
                        "turn_node_id": "unknown-turn",
                        "suspended_node_id": "unknown-node",
                        "suspended_token_id": "unknown-token",
                        "client_result": {},
                    },
                    resume_errors["unknown_run_id_status"],
                ),
            ):
                invalid_resume = client.post(
                    "/api/syscall/v1/resume",
                    json={"version": "v1", "op": "resume", "args": args},
                    headers=headers,
                )
                assert invalid_resume.status_code == expected_status
            assert registry.get_run(run_id)["cancel_requested"] is False

            denied_termination = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": run_id},
                },
                headers=denied_termination_headers,
            )
            assert denied_termination.status_code == 403
            assert registry.get_run(run_id)["cancel_requested"] is False

            terminated = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": run_id},
                },
                headers=headers,
            )
            assert terminated.status_code == 200, terminated.text
            cancel_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/run-cancel-v1.json"
                ).read_text(encoding="utf-8")
            )
            terminated_result = terminated.json()["result"]
            assert all(
                field in terminated_result
                for field in cancel_contract["required_fields"]
            )
            assert (
                terminated_result["status"]
                in cancel_contract["accepted_run_statuses"]
            )
            assert terminated_result["cancel_requested"] is True
            assert registry.get_run(run_id)["cancel_requested"] is True
            final_run = _wait_for_status(client, run_id, headers, {"cancelled"})
            assert final_run["status"] == "cancelled"
            events_after_termination = _collect_sse_events(client, run_id, headers)
            event_names = [name for name, _payload in events_after_termination]
            assert cancel_contract["nonterminal_cancel_event"] in event_names
            repeated_termination = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": run_id},
                },
                headers=headers,
            )
            assert repeated_termination.status_code == 200
            assert repeated_termination.json()["result"]["status"] == "cancelled"
            assert repeated_termination.json()["result"]["cancel_requested"] is True
            assert _collect_sse_events(client, run_id, headers) == events_after_termination

        result = first.json()["result"]
        assert result["status"] == "queued"
        assert registry.get_run(result["run_id"])["user_turn_node_id"] == result[
            "user_turn_node_id"
        ]
        assert len(service.list_transcript(conversation_id)) == 1
    finally:
        for engine in engines:
            engine.close()
        postgres_engine.dispose()
        cleanup_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)
        try:
            with cleanup_engine.begin() as connection:
                for schema in schema_names:
                    connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            cleanup_engine.dispose()


def test_checkpoint_syscall_uses_python_postgres_backend(
    monkeypatch, tmp_path, pg_dsn, pg_schema
):
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture is unavailable")

    import sqlalchemy as sa

    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    schema_names = [f"{pg_schema}_{graph_type}" for graph_type in ("knowledge", "conversation", "workflow")]
    postgres_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)

    def backend_factory(engine):
        return PgVectorBackend(
            engine=postgres_engine,
            embedding_dim=384,
            schema=f"{pg_schema}_{engine.kg_graph_type}",
        )

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / "checkpoint-postgres",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=backend_factory,
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        contract = json.loads(
            (Path(__file__).resolve().parents[2] / "contracts/golden/syscall-checkpoint-persisted-v1.json").read_text(encoding="utf-8")
        )
        _seed_checkpoint(
            conversation_engine,
            run_id=contract["run_id"],
            workflow_id=contract["workflow_id"],
            step_seq=contract["step_seq_expected"],
            state=contract["state"],
        )
        with TestClient(server.app) as client:
            response = client.post(
                "/api/syscall/v1/checkpoint",
                json={
                    "version": contract["version"],
                    "op": contract["op"],
                    "args": {
                        "run_id": contract["run_id"],
                        "step_seq": contract["step_seq_input"],
                    },
                },
                headers=_token_header(client, role="ro", ns="workflow"),
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == contract["expected_result"]
    finally:
        for item in engines:
            item.close()
        postgres_engine.dispose()
        cleanup_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)
        try:
            with cleanup_engine.begin() as connection:
                for schema in schema_names:
                    connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            cleanup_engine.dispose()


def test_project_view_syscall_uses_python_postgres_backend(
    monkeypatch, tmp_path, pg_dsn, pg_schema
):
    if not pg_dsn or not pg_schema:
        pytest.skip("PostgreSQL fixture is unavailable")

    import sqlalchemy as sa

    from kogwistar.engine_core.postgres_backend import PgVectorBackend

    schema_names = [f"{pg_schema}_{graph_type}" for graph_type in ("knowledge", "conversation", "workflow")]
    postgres_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)

    def backend_factory(engine):
        return PgVectorBackend(
            engine=postgres_engine,
            embedding_dim=384,
            schema=f"{pg_schema}_{engine.kg_graph_type}",
        )

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / "project-view-postgres",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=backend_factory,
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        contract = json.loads(
            (Path(__file__).resolve().parents[2] / "contracts/golden/workflow-resources-http-v1.json").read_text(encoding="utf-8")
        )
        with TestClient(server.app) as client:
            denied = client.post(
                "/api/syscall/v1/project_view",
                json={"version": "v1", "op": "project_view", "args": {}},
                headers=_token_header(client, role="ro", ns="workflow", capabilities="not-a-required-capability"),
            )
            allowed = client.post(
                "/api/syscall/v1/project_view",
                json={"version": "v1", "op": "project_view", "args": {}},
                headers=_token_header(
                    client,
                    role="ro",
                    ns="workflow",
                    capabilities=",".join(contract["required_capabilities"]),
                ),
            )
        assert denied.status_code == 403
        assert allowed.status_code == 200, allowed.text
        result = allowed.json()["result"]
        assert all(field in result for field in contract["required_fields"])
        for key, fields in contract["required_object_fields"].items():
            assert all(field in result[key] for field in fields), key
    finally:
        for item in engines:
            item.close()
        postgres_engine.dispose()
        cleanup_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)
        try:
            with cleanup_engine.begin() as connection:
                for schema in schema_names:
                    connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        finally:
            cleanup_engine.dispose()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_run_lifecycle_syscalls_use_python_backend(monkeypatch, tmp_path, backend):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"run-lifecycle-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        service, registry = _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        workflow_id = "wf.in-memory-syscall"
        identity = claims_ctx.set(
            {
                "ns": "workflow",
                "role": "rw",
                "capabilities": [
                    "spawn_process",
                    "workflow.run.write",
                    "workflow.run.read",
                    "project_view",
                    "workflow.design.write",
                    "write_graph",
                ],
                "sub": "memory-workflow-owner",
            }
        )
        try:
            service.workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id="memory-workflow-owner",
                node_id="start",
                label="Start",
                op="start",
                start=True,
            )
            service.workflow_design_upsert_node(
                workflow_id=workflow_id,
                designer_id="memory-workflow-owner",
                node_id="end",
                label="End",
                op="end",
                terminal=True,
            )
            service.workflow_design_upsert_edge(
                workflow_id=workflow_id,
                designer_id="memory-workflow-owner",
                edge_id="edge-start-end",
                src="start",
                dst="end",
                relation="wf_next",
                is_default=True,
            )
            conversation = service.create_conversation(user_id="memory-workflow-owner")
        finally:
            claims_ctx.reset(identity)

        with TestClient(server.app) as client:
            headers = _token_header(
                client,
                role="rw",
                ns="workflow",
                username="memory-workflow-owner",
                capabilities=(
                    "spawn_process,workflow.run.write,workflow.run.read,project_view,"
                    "workflow.design.write,write_graph,approve_action"
                ),
            )
            spawned = client.post(
                "/api/syscall/v1/spawn_process",
                json={
                    "version": "v1",
                    "op": "spawn_process",
                    "args": {
                        "workflow_id": workflow_id,
                        "conversation_id": conversation["conversation_id"],
                        "turn_node_id": conversation["start_node_id"],
                        "user_id": "memory-workflow-owner",
                        "initial_state": {
                            "user_id": "memory-workflow-owner",
                            "conversation_id": conversation["conversation_id"],
                        },
                    },
                },
                headers=headers,
            )
            assert spawned.status_code == 200, spawned.text
            spawned_result = spawned.json()["result"]
            assert spawned_result["workflow_id"] == workflow_id
            submit_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/workflow-run-submit-http-v1.json"
                ).read_text(encoding="utf-8")
            )
            assert all(
                field in spawned_result
                for field in submit_contract["required_common_fields"]
            )
            for field, value in submit_contract["expected_common_values"].items():
                assert spawned_result[field] == value

            denied_headers = _token_header(
                client,
                role="rw",
                ns="workflow",
                username="memory-workflow-owner",
                capabilities="workflow.run.read",
            )
            terminate_errors = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-terminate-errors-v1.json"
                ).read_text(encoding="utf-8")
            )
            for args, expected_status in (
                ({}, terminate_errors["missing_run_id_status"]),
                (
                    {"run_id": f"{spawned_result['run_id']}-unknown"},
                    terminate_errors["unknown_run_id_status"],
                ),
            ):
                invalid_termination = client.post(
                    "/api/syscall/v1/terminate_process",
                    json={"version": "v1", "op": "terminate_process", "args": args},
                    headers=headers,
                )
                assert invalid_termination.status_code == expected_status
            assert registry.get_run(spawned_result["run_id"])["cancel_requested"] is False
            resume_errors = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/syscall-resume-errors-v1.json"
                ).read_text(encoding="utf-8")
            )
            for args, expected_status in (
                ({}, resume_errors["missing_run_id_status"]),
                (
                    {
                        "run_id": f"{spawned_result['run_id']}-unknown",
                        "workflow_id": "unknown-workflow",
                        "conversation_id": "unknown-conversation",
                        "turn_node_id": "unknown-turn",
                        "suspended_node_id": "unknown-node",
                        "suspended_token_id": "unknown-token",
                        "client_result": {},
                    },
                    resume_errors["unknown_run_id_status"],
                ),
            ):
                invalid_resume = client.post(
                    "/api/syscall/v1/resume",
                    json={"version": "v1", "op": "resume", "args": args},
                    headers=headers,
                )
                assert invalid_resume.status_code == expected_status
            assert registry.get_run(spawned_result["run_id"])["cancel_requested"] is False

            denied_termination = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": spawned_result["run_id"]},
                },
                headers=denied_headers,
            )
            assert denied_termination.status_code == 403
            assert registry.get_run(spawned_result["run_id"])["cancel_requested"] is False

            events_before_termination = registry.list_events(
                spawned_result["run_id"], limit=100
            )
            terminated = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": spawned_result["run_id"]},
                },
                headers=headers,
            )
            assert terminated.status_code == 200, terminated.text
            cancel_contract = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "contracts/golden/run-cancel-v1.json"
                ).read_text(encoding="utf-8")
            )
            terminated_result = terminated.json()["result"]
            assert all(field in terminated_result for field in cancel_contract["required_fields"])
            assert terminated_result["status"] in cancel_contract["accepted_run_statuses"]
            deadline = time.monotonic() + 10.0
            final_run = registry.get_run(spawned_result["run_id"])
            while time.monotonic() < deadline:
                if final_run and final_run["status"] in {
                    "succeeded",
                    "failed",
                    "cancelled",
                }:
                    break
                time.sleep(0.05)
                final_run = registry.get_run(spawned_result["run_id"])
            assert final_run is not None
            assert final_run["status"] == "cancelled"
            events_after_termination = registry.list_events(
                spawned_result["run_id"], limit=100
            )
            event_names = [event["event_type"] for event in events_after_termination]
            assert cancel_contract["nonterminal_cancel_event"] in event_names
            repeated_termination = client.post(
                "/api/syscall/v1/terminate_process",
                json={
                    "version": "v1",
                    "op": "terminate_process",
                    "args": {"run_id": spawned_result["run_id"]},
                },
                headers=headers,
            )
            assert repeated_termination.status_code == 200
            assert repeated_termination.json()["result"]["status"] == "cancelled"
            assert repeated_termination.json()["result"]["cancel_requested"] is True
            assert registry.list_events(
                spawned_result["run_id"], limit=100
            ) == events_after_termination
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_checkpoint_syscall_reads_python_backend(monkeypatch, tmp_path, backend):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"checkpoint-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/syscall-checkpoint-persisted-v1.json"
            ).read_text(encoding="utf-8")
        )
        _seed_checkpoint(
            conversation_engine,
            run_id=contract["run_id"],
            workflow_id=contract["workflow_id"],
            step_seq=contract["step_seq_expected"],
            state=contract["state"],
        )
        with TestClient(server.app) as client:
            response = client.post(
                "/api/syscall/v1/checkpoint",
                json={
                    "version": contract["version"],
                    "op": contract["op"],
                    "args": {
                        "run_id": contract["run_id"],
                        "step_seq": contract["step_seq_input"],
                    },
                },
                headers=_token_header(client, role="ro", ns="workflow"),
            )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == contract["expected_result"]
    finally:
        for engine in engines:
            engine.close()


@pytest.mark.parametrize("backend", ["in_memory", "chroma"])
def test_invoke_tool_syscall_uses_python_backend(monkeypatch, tmp_path, backend):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)

    engines = build_engine_triplet(
        root=tmp_path / f"invoke-tool-{backend}",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=(
            build_in_memory_backend if backend == "in_memory" else None
        ),
    )
    try:
        engine, conversation_engine, workflow_engine = engines
        service, _registry = _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        identity = claims_ctx.set(
            {
                "ns": "conversation",
                "role": "rw",
                "capabilities": [
                    "spawn_process",
                    "send_message",
                    "invoke_tool",
                    "read_graph",
                    "project_view",
                    "read_security_scope",
                    "workflow.run.write",
                    "workflow.run.read",
                ],
                "sub": "memory-tool-owner",
            }
        )
        try:
            conversation = service.create_conversation(user_id="memory-tool-owner")
        finally:
            claims_ctx.reset(identity)

        with TestClient(server.app) as client:
            headers = _token_header(
                client,
                role="rw",
                ns="conversation",
                username="memory-tool-owner",
                capabilities=(
                    "invoke_tool,read_graph,send_message,project_view,"
                    "read_security_scope,spawn_process,workflow.run.write,workflow.run.read"
                ),
            )
            response = client.post(
                "/api/syscall/v1/invoke_tool",
                json={
                    "version": "v1",
                    "op": "invoke_tool",
                    "args": {
                        "tool_name": "list_transcript",
                        "conversation_id": conversation["conversation_id"],
                    },
                },
                headers=headers,
            )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["tool_name"] == "list_transcript"
        assert result["conversation_id"] == conversation["conversation_id"]
        assert result["messages"] == []

        error_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/syscall-invoke-tool-errors-v1.json"
            ).read_text(encoding="utf-8")
        )
        graph_ids_before_errors = {
            node.id
            for node in conversation_engine.read.get_nodes(
                where={"conversation_id": conversation["conversation_id"]},
                node_type=ConversationNode,
                limit=1_000,
            )
        }
        for case in error_contract["cases"]:
            error_response = client.post(
                "/api/syscall/v1/invoke_tool",
                json=case["request"],
                headers=headers,
            )
            assert error_response.status_code == case["http_status"], case["name"]
            assert error_response.json() == case["response"], case["name"]
        graph_ids_after_errors = {
            node.id
            for node in conversation_engine.read.get_nodes(
                where={"conversation_id": conversation["conversation_id"]},
                node_type=ConversationNode,
                limit=1_000,
            )
        }
        assert graph_ids_after_errors == graph_ids_before_errors
    finally:
        for engine in engines:
            engine.close()
