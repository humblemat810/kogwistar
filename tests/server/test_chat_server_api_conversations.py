"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_chat_rest_submit_and_sse(monkeypatch, engine_triplet):
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only answer submission attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")
        wrong_ns_headers = _token_header(client, role="ro", ns="wisdom")

        created = client.post(
            "/api/conversations", json={"user_id": "u-chat"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        validation_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts"
                / "golden"
                / "conversation-answer-validation-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert validation_contract["schema_version"] == 1
        for case in validation_contract["cases"]:
            invalid = client.post(
                f"/api/conversations/{conversation_id}/turns:answer",
                json={"user_id": "u-chat", "text": case["text"]},
                headers=rw_headers,
            )
            assert invalid.status_code == case["http_status"], case["name"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "u-chat", "text": "hello world"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        submit_body = submit.json()
        assert set(submit_body) == {
            "run_id",
            "conversation_id",
            "workflow_id",
            "status",
            "user_turn_node_id",
        }
        assert submit_body["conversation_id"] == conversation_id
        assert submit_body["workflow_id"] == "debug.rag.v1"
        assert submit_body["status"] == "queued"
        run_id = submit_body["run_id"]
        assert run_id
        assert submit_body["user_turn_node_id"]

        final_run = _wait_for_status(client, run_id, ro_headers, {"succeeded"})
        assert final_run["assistant_turn_node_id"]
        assert "last_step_seq" not in final_run
        assert "step_count" not in final_run
        assert not Path(
            workflow_engine.persist_directory, "server_runs.sqlite"
        ).exists()

        turns = client.get(
            f"/api/conversations/{conversation_id}/turns", headers=ro_headers
        )
        assert turns.status_code == 200
        turns.raise_for_status()
        transcript_response = turns.json()
        assert set(transcript_response) == {"conversation_id", "turns"}
        assert transcript_response["conversation_id"] == conversation_id
        transcript = transcript_response["turns"]
        assert [turn["role"] for turn in transcript] == ["user", "assistant"]
        assert transcript[1]["content"] == "Assistant reply: hello world"

        events = _collect_sse_events(client, run_id, ro_headers)
        names = [name for name, _payload in events]
        assert "run.created" in names
        assert "run.started" in names
        assert "run.stage" in names
        assert "reasoning.summary" in names
        assert "output.delta" in names
        assert "output.completed" in names
        assert names[-1] == "run.completed"

        denied_stream = client.get(
            f"/api/runs/{run_id}/events", headers=wrong_ns_headers
        )
        assert denied_stream.status_code == 403
        assert run_id not in denied_stream.text

        sse_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-events-sse-v1.json"
            ).read_text(encoding="utf-8")
        )
        with client.stream(
            "GET", f"/api/runs/{run_id}/events", headers=ro_headers
        ) as response:
            assert response.status_code == sse_contract["success_status"]
            assert response.headers["content-type"].startswith(
                sse_contract["media_type"]
            )
            sse_body = "".join(response.iter_text())
        lines = sse_body.splitlines()
        event_line = next(
            index
            for index, line in enumerate(lines)
            if line == f"event: {sse_contract['event_name']}"
        )
        frame_start = (
            event_line - 1
            if event_line > 0 and lines[event_line - 1].startswith("id: ")
            else event_line
        )
        frame_end = next(
            (index for index in range(frame_start + 1, len(lines)) if not lines[index]),
            len(lines),
        )
        frame_lines = lines[frame_start:frame_end]
        event_id = next(line[4:] for line in frame_lines if line.startswith("id: "))
        assert event_id.isdigit()
        event_data = json.loads(
            next(line[6:] for line in frame_lines if line.startswith("data: "))
        )
        assert set(sse_contract["required_data_fields"]).issubset(event_data)
        assert event_data["run_id"] == run_id
        assert event_data["event_type"] == sse_contract["event_name"]

        reopened = client.get(
            f"/api/conversations/{conversation_id}", headers=ro_headers
        )
        assert reopened.status_code == 200
        reopened.raise_for_status()
        conversation_summary = reopened.json()
        assert set(conversation_summary) == {
            "conversation_id",
            "user_id",
            "status",
            "start_node_id",
            "tail_node_id",
            "turn_count",
        }
        assert conversation_summary["conversation_id"] == conversation_id
        assert conversation_summary["user_id"] == "u-chat"
        assert conversation_summary["status"] == "active"
        assert conversation_summary["start_node_id"]
        assert conversation_summary["tail_node_id"]
        assert conversation_summary["turn_count"] == 2


def test_conversation_list_is_user_scoped_and_has_stable_response_shape(
    monkeypatch, engine_triplet
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only conversation list attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    now = int(time.time())

    def headers(user_id: str, role: str) -> dict[str, str]:
        token = jwt.encode(
            {
                "sub": user_id,
                "user_id": user_id,
                "role": role,
                "ns": ["conversation"],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        return {"Authorization": f"Bearer {token}"}

    with TestClient(server.app) as client:
        rw_headers = headers("owner-a", "rw")
        ro_headers = headers("owner-a", "ro")

        created = client.post(
            "/api/conversations",
            json={"user_id": "untrusted-body-user", "conversation_id": "owned"},
            headers=rw_headers,
        )
        assert created.status_code == 200, created.text
        assert created.json()["user_id"] == "owner-a"
        service._conversation_queries.create_conversation(
            user_id="different-owner", conversation_id="not-visible"
        )

        listed = client.get("/api/conversations", headers=ro_headers)
        assert listed.status_code == 200, listed.text
        list_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/conversation-list-http-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert set(listed.json()) == set(list_contract["response_fields"])
        assert all(
            set(conversation) == set(list_contract["conversation_fields"])
            for conversation in listed.json()["conversations"]
        )
        assert listed.json() == {
            "conversations": [
                {
                    "id": "owned",
                    "start_node_id": created.json()["start_node_id"],
                    "status": "active",
                    "turn_count": 0,
                }
            ]
        }


def test_conversation_list_http_denials_precede_service_read(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-list-denied-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    reads = []
    monkeypatch.setattr(
        service._conversation_queries,
        "list_conversations_for_user",
        lambda user_id: reads.append(user_id) or [],
    )

    with TestClient(server.app) as client:
        for case in fixture["denial_cases"]:
            claims = case["claims"]
            headers = _token_header(
                client,
                role=claims["role"],
                ns=claims["ns"],
                username=claims["sub"],
                capabilities=",".join(claims["capabilities"]),
            )
            response = client.get(fixture["path"], headers=headers)
            assert response.status_code == case["status"], case["case"]

    assert reads == []


def test_conversation_create_uses_authenticated_owner_and_persists_start_node(
    monkeypatch, engine_triplet
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only conversation creation attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "owner-create",
            "user_id": "owner-create",
            "role": "rw",
            "ns": ["conversation"],
            "iat": now,
            "exp": now + 3600,
            "iss": "local",
        },
        TEST_JWT_SECRET,
        algorithm=TEST_JWT_ALG,
    )
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(server.app) as client:
        created = client.post(
            "/api/conversations",
            json={
                "user_id": "spoofed-owner",
                "conversation_id": "created-by-owner",
                "start_node_id": "created-start-node",
            },
            headers=headers,
        )

    assert created.status_code == 200, created.text
    result = created.json()
    assert result["conversation_id"] == "created-by-owner"
    assert result["start_node_id"] == "created-start-node"
    assert result["user_id"] == "owner-create"
    assert result["turn_count"] == 0

    nodes = conversation_engine.read.get_nodes(
        where={"conversation_id": "created-by-owner"},
        node_type=ConversationNode,
        limit=10,
    )
    starts = [
        node
        for node in nodes
        if node.metadata.get("entity_type") == "conversation_start"
    ]
    assert len(starts) == 1
    assert starts[0].id == "created-start-node"
    assert starts[0].user_id == "owner-create"


def test_conversation_detail_uses_python_defaults_and_checks_namespace(
    monkeypatch, engine_triplet
):
    detail_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-detail-http-v1.json"
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only conversation detail attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service._conversation_queries.create_conversation(
        user_id="alice", conversation_id="conversation-empty", start_node_id="start-empty"
    )

    now = int(time.time())

    def headers(user_id: str, namespace: str) -> dict[str, str]:
        token = jwt.encode(
            {
                "sub": user_id,
                "user_id": user_id,
                "role": "ro",
                "ns": [namespace],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        return {"Authorization": f"Bearer {token}"}

    with TestClient(server.app) as client:
        response = client.get(
            detail_contract["path"],
            headers=headers("alice", "conversation"),
        )
        assert response.status_code == 200, response.text
        assert response.json() == detail_contract["response"]

        denied = client.get(
            detail_contract["path"],
            headers=headers("foreign-user", "wisdom"),
        )
        assert denied.status_code == 403
        assert "alice" not in denied.text
        assert "start_node_id" not in denied.text


def test_conversation_transcript_uses_python_defaults_and_checks_namespace(
    monkeypatch, engine_triplet
):
    transcript_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-transcript-http-v1.json"
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only transcript read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service._conversation_queries.create_conversation(
        user_id="owner-transcript", conversation_id="conversation-empty"
    )

    now = int(time.time())

    def headers(user_id: str, namespace: str) -> dict[str, str]:
        token = jwt.encode(
            {
                "sub": user_id,
                "user_id": user_id,
                "role": "ro",
                "ns": [namespace],
                "iat": now,
                "exp": now + 3600,
                "iss": "local",
            },
            TEST_JWT_SECRET,
            algorithm=TEST_JWT_ALG,
        )
        return {"Authorization": f"Bearer {token}"}

    with TestClient(server.app) as client:
        response = client.get(
            transcript_contract["path"],
            headers=headers("owner-transcript", "conversation"),
        )
        assert response.status_code == 200, response.text
        assert response.json() == transcript_contract["response"]

        denied = client.get(
            transcript_contract["path"],
            headers=headers("foreign-user", "wisdom"),
        )
        assert denied.status_code == 403
        assert "owner-transcript" not in denied.text
        assert "conversation-empty" not in denied.text


def test_latest_snapshot_uses_python_defaults_and_checks_namespace(
    monkeypatch, engine_triplet
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

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only snapshot read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with TestClient(server.app) as client:
        conv_headers = _token_header(client, role="rw", ns="conversation")
        workflow_headers = _token_header(
            client, role="ro", ns="workflow", username="snapshot-owner"
        )
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        created = client.post(
            "/api/conversations", json={"user_id": "snapshot-owner"}, headers=conv_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]
        start_node_id = created.json()["start_node_id"]

        service = ConversationService.from_engine(
            conversation_engine,
            knowledge_engine=engine,
            workflow_engine=workflow_engine,
        )
        snapshot_id = service.persist_context_snapshot(
            conversation_id=conversation_id,
            run_id="python-only-snapshot-run",
            run_step_seq=0,
            stage="draft_answer",
            view=SimpleNamespace(
                messages=[{"role": "user", "content": "snapshot fixture"}],
                items=[SimpleNamespace(node_id=start_node_id)],
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
            owner_id="snapshot-owner",
            supersedes_version=1,
        )

        response = client.get(
            f"/api/conversations/{conversation_id}/snapshots/latest",
            headers=workflow_headers,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload) == {
            "snapshot_node_id",
            "conversation_id",
            "metadata",
            "properties",
        }
        assert payload["snapshot_node_id"] == snapshot_id
        assert payload["conversation_id"] == conversation_id
        assert isinstance(payload["metadata"], dict)
        assert isinstance(payload["properties"], dict)

        denied = client.get(
            f"/api/conversations/{conversation_id}/snapshots/latest",
            headers=denied_headers,
        )
        assert denied.status_code == 403
        assert snapshot_id not in denied.text
        assert conversation_id not in denied.text

        mount_args = {
            "version": "v1",
            "op": "mount_memory",
            "args": {
                "conversation_id": conversation_id,
                "run_id": "python-only-snapshot-run",
                "stage": "draft_answer",
            },
        }
        mounted = client.post(
            "/api/syscall/v1/mount_memory",
            json=mount_args,
            headers=_token_header(
                client, role="ro", ns="conversation", username="snapshot-owner"
            ),
        )
        assert mounted.status_code == 200, mounted.text
        assert mounted.json()["result"]["snapshot_node_id"] == snapshot_id
        assert mounted.json()["result"]["metadata"].get("summary") != "snapshot fixture"

        private_denied = client.post(
            "/api/syscall/v1/mount_memory",
            json=mount_args,
            headers=_token_header(
                client, role="ro", ns="conversation", username="unrelated-reader"
            ),
        )
        assert private_denied.status_code == 404
        assert snapshot_id not in private_denied.text
        assert "snapshot fixture" not in private_denied.text

        namespace_denied = client.post(
            "/api/syscall/v1/mount_memory",
            json=mount_args,
            headers=_token_header(
                client, role="ro", ns="wisdom", username="snapshot-owner"
            ),
        )
        assert namespace_denied.status_code == 403


def test_mount_memory_acl_and_payload_survive_python_engine_reopen(
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

    root = tmp_path / "mount-memory-reopen"

    def open_engines():
        return build_engine_triplet(
            root=root,
            embedding_function=FakeEmbeddingFunction(),
        )

    engines = open_engines()
    try:
        engine, conversation_engine, workflow_engine = engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        with TestClient(server.app) as client:
            owner_headers = _token_header(
                client, role="rw", ns="conversation", username="snapshot-owner"
            )
            created = client.post(
                "/api/conversations",
                json={"user_id": "snapshot-owner"},
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
                run_id="reopen-run",
                run_step_seq=0,
                stage="draft_answer",
                view=SimpleNamespace(
                    messages=[{"role": "user", "content": "persisted private snapshot"}],
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
                owner_id="snapshot-owner",
                supersedes_version=1,
            )
            args = {
                "version": "v1",
                "op": "mount_memory",
                "args": {
                    "conversation_id": conversation_id,
                    "run_id": "reopen-run",
                    "stage": "draft_answer",
                },
            }
            mounted = client.post(
                "/api/syscall/v1/mount_memory", json=args, headers=owner_headers
            )
            assert mounted.status_code == 200, mounted.text
            expected_result = mounted.json()["result"]
            assert expected_result["snapshot_node_id"] == snapshot_id
    finally:
        for engine in engines:
            engine.close()

    reopened_engines = open_engines()
    try:
        engine, conversation_engine, workflow_engine = reopened_engines
        _configure_server(
            monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
        )
        with TestClient(server.app) as client:
            owner_headers = _token_header(
                client, role="ro", ns="conversation", username="snapshot-owner"
            )
            outsider_headers = _token_header(
                client, role="ro", ns="conversation", username="unrelated-reader"
            )
            args = {
                "version": "v1",
                "op": "mount_memory",
                "args": {
                    "conversation_id": conversation_id,
                    "run_id": "reopen-run",
                    "stage": "draft_answer",
                },
            }
            mounted = client.post(
                "/api/syscall/v1/mount_memory", json=args, headers=owner_headers
            )
            assert mounted.status_code == 200, mounted.text
            assert mounted.json()["result"] == expected_result

            denied = client.post(
                "/api/syscall/v1/mount_memory", json=args, headers=outsider_headers
            )
            assert denied.status_code == 404
            assert snapshot_id not in denied.text
            assert "persisted private snapshot" not in denied.text
    finally:
        for engine in reopened_engines:
            engine.close()
