"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_checkpoint_detail_uses_python_defaults_and_checks_namespace(
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
        raise AssertionError("Python-only checkpoint read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-checkpoint-run"
    _seed_checkpoint(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=0,
        state={"counter": 7},
    )

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-checkpoint-detail-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(
            f"/api/runs/{run_id}/checkpoints/0", headers=workflow_headers
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert set(contract["required_fields"]).issubset(payload)
        assert payload["run_id"] == run_id
        assert payload["step_seq"] == 0
        assert payload["state"] == {"counter": 7}

        workflow_response = client.get(
            f"/api/workflow/runs/{run_id}/checkpoints/0", headers=workflow_headers
        )
        assert workflow_response.status_code == contract["success_status"]
        assert workflow_response.json() == payload

        denied = client.get(
            f"/api/runs/{run_id}/checkpoints/0", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_checkpoint_list_uses_python_defaults_and_checks_namespace(
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
        raise AssertionError("Python-only checkpoint-list read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-checkpoint-list-run"
    _seed_checkpoint(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=3,
        state={"cursor": "later"},
    )
    _seed_checkpoint(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=0,
        state={"cursor": "start"},
    )

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-checkpoints-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(
            f"/api/runs/{run_id}/checkpoints", headers=workflow_headers
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert set(contract["required_envelope_fields"]).issubset(payload)
        assert payload["run_id"] == run_id
        checkpoints = payload["checkpoints"]
        assert len(checkpoints) == 2
        assert [item["step_seq"] for item in checkpoints] == [0, 3]
        for checkpoint in checkpoints:
            assert set(contract["required_checkpoint_fields"]).issubset(checkpoint)
            assert checkpoint["workflow_id"] == "python-only-workflow"
        assert checkpoints[0]["state"] == {"cursor": "start"}
        assert checkpoints[1]["state"] == {"cursor": "later"}

        workflow_response = client.get(
            f"/api/workflow/runs/{run_id}/checkpoints", headers=workflow_headers
        )
        assert workflow_response.status_code == contract["success_status"]
        assert workflow_response.json() == payload

        denied = client.get(
            f"/api/runs/{run_id}/checkpoints", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_run_detail_uses_python_defaults_and_checks_namespace(monkeypatch, engine_triplet):
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
        raise AssertionError("Python-only run-detail read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-run-detail"
    expected_ids = {
        "run_id": run_id,
        "conversation_id": "python-only-conversation",
        "workflow_id": "python-only-workflow",
        "user_id": "python-only-user",
        "user_turn_node_id": "python-only-turn",
    }
    registry.create_run(**expected_ids, status="succeeded")

    with TestClient(server.app) as client:
        conversation_headers = _token_header(client, role="ro", ns="conversation")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-status-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(
            f"/api/runs/{run_id}", headers=conversation_headers
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert set(contract["required_fields"]).issubset(payload)
        assert {key: payload[key] for key in expected_ids} == expected_ids
        assert payload["status"] == "succeeded"
        assert payload["terminal"] is True
        assert payload["cancel_requested"] is False
        assert payload["result"] is None
        assert payload["error"] is None

        denied = client.get(f"/api/runs/{run_id}", headers=denied_headers)
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_run_evidence_uses_python_defaults_and_checks_namespace(
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
        raise AssertionError("Python-only run-evidence read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/run-evidence-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = contract["run_id"]
    conversation_id = contract["conversation_id"]

    with TestClient(server.app) as client:
        conversation_headers = _token_header(client, role="rw", ns="conversation")
        workflow_headers = _token_header(
            client, role="ro", ns="workflow", username="alice"
        )
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        created = client.post(
            "/api/conversations",
            json={"user_id": "alice", "conversation_id": conversation_id},
            headers=conversation_headers,
        )
        assert created.status_code == 200, created.text
        start_node_id = created.json()["start_node_id"]
        for node_id in contract["evidence_refs"]:
            conversation_engine.write.add_node(
                Node(
                    id=node_id,
                    label=node_id,
                    type="entity",
                    summary=node_id,
                    doc_id=node_id,
                    mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                    metadata={
                        "entity_type": "evidence",
                        "conversation_id": conversation_id,
                    },
                    properties={},
                    level_from_root=0,
                    domain_id=None,
                    canonical_entity_id=None,
                    embedding=None,
                )
            )
        registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id="evidence-workflow",
            user_id="alice",
            user_turn_node_id=start_node_id,
            status="succeeded",
        )
        conversation_service = ConversationService.from_engine(
            conversation_engine,
            knowledge_engine=engine,
            workflow_engine=workflow_engine,
        )
        snapshot_id = conversation_service.persist_context_snapshot(
            conversation_id=conversation_id,
            run_id=run_id,
            run_step_seq=contract["snapshot_metadata"]["run_step_seq"],
            stage="answer",
            view=SimpleNamespace(
                messages=[{"role": "user", "content": "evidence test"}],
                items=[
                    SimpleNamespace(node_id=node_id)
                    for node_id in contract["evidence_refs"]
                ],
                cost=None,
                token_budget=32,
            ),
            model_name="test-model",
            budget_tokens=32,
            evidence_pack_digest=contract["source_summary"]["evidence_pack_digest"],
        )
        persisted_snapshot_metadata = conversation_engine.read.get_node_metadatas(
            ids=[snapshot_id], limit=1
        )[0]
        assert persisted_snapshot_metadata["used_node_ids"] == contract["evidence_refs"]
        conversation_engine.acl.record_acl(
            grain="node",
            truth_graph="conversation",
            entity_id=snapshot_id,
            version=2,
            mode="private",
            owner_id="alice",
            supersedes_version=1,
        )

        response = client.get(
            f"/api/runs/{run_id}/evidence", headers=workflow_headers
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert snapshot_id == contract["snapshot_node_id"]
        normalized_payload = dict(payload)
        normalized_payload["snapshot_metadata"] = {
            key: payload["snapshot_metadata"][key]
            for key in contract["snapshot_metadata"]
        }
        assert normalized_payload == contract

        denied = client.get(
            f"/api/runs/{run_id}/evidence", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_resume_route_uses_python_defaults_and_checks_namespace(
    monkeypatch, engine_triplet
):
    resume_result_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-resume-result-v1.json"
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
        raise AssertionError("Python-only run resume attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    captured_requests = []

    def capture_resume_request(request):
        captured_requests.append(request)
        return {
            "workflow_status": "succeeded",
            "final_state": {"resumed_value": request.client_result["resumed_value"]},
            "budget": {},
        }

    monkeypatch.setattr(
        service._run_execution, "_default_resume_runner", capture_resume_request
    )
    request_body = {
        "suspended_node_id": "approval-step",
        "suspended_token_id": "approval-token",
        "client_result": {"status": "success", "resumed_value": "approved"},
        "workflow_id": "python-only-workflow",
        "conversation_id": "python-only-conversation",
        "turn_node_id": "python-only-turn",
        "user_id": "python-only-user",
    }

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="rw", ns="workflow")
        denied_headers = _token_header(client, role="rw", ns="wisdom")
        response = client.post(
            "/api/runs/python-only-resume-run/resume",
            json=request_body,
            headers=workflow_headers,
        )
        assert response.status_code == 200, response.text
        assert set(response.json()) == set(resume_result_contract["required_fields"])
        assert response.json() == {
            "workflow_status": "succeeded",
            "final_state": {"resumed_value": "approved"},
            "budget": {},
        }
        assert len(captured_requests) == 1
        request = captured_requests[0]
        assert request.run_id == "python-only-resume-run"
        assert request.suspended_node_id == request_body["suspended_node_id"]
        assert request.suspended_token_id == request_body["suspended_token_id"]
        assert request.client_result == request_body["client_result"]
        assert request.workflow_id == request_body["workflow_id"]
        assert request.conversation_id == request_body["conversation_id"]
        assert request.turn_node_id == request_body["turn_node_id"]
        assert request.user_id == request_body["user_id"]

        denied = client.post(
            "/api/runs/python-only-resume-run/resume",
            json=request_body,
            headers=denied_headers,
        )
        assert denied.status_code == 403
        assert "python-only-resume-run" not in denied.text
        assert len(captured_requests) == 1


@pytest.mark.parametrize("backend", ["in_memory", "chroma", "postgres"])
def test_python_resume_syscall_continues_persisted_workflow(
    monkeypatch, request, tmp_path, backend, pg_dsn, pg_schema
):
    from kogwistar.server import chat_service_run_execution as run_execution_module

    resume_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-resume-result-v1.json"
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
        raise AssertionError("Python-only resume attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    postgres_engine = None
    backend_factory = None
    if backend == "in_memory":
        backend_factory = build_in_memory_backend
    elif backend == "postgres":
        if not pg_dsn or not pg_schema:
            pytest.skip("PostgreSQL fixture is unavailable")
        import sqlalchemy as sa

        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        postgres_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)

        def cleanup_postgres_schemas():
            cleanup_engine = sa.create_engine(pg_dsn, future=True, pool_pre_ping=True)
            with cleanup_engine.begin() as connection:
                for graph_type in ("knowledge", "conversation", "workflow"):
                    connection.execute(
                        sa.text(
                            f'DROP SCHEMA IF EXISTS "{pg_schema}_{graph_type}" CASCADE'
                        )
                    )
            cleanup_engine.dispose()
            postgres_engine.dispose()

        request.addfinalizer(cleanup_postgres_schemas)
        backend_factory = lambda engine: PgVectorBackend(
            engine=postgres_engine,
            embedding_dim=384,
            schema=f"{pg_schema}_{engine.kg_graph_type}",
        )

    engines = build_engine_triplet(
        root=tmp_path / f"python-{backend}-resume",
        embedding_function=FakeEmbeddingFunction(),
        backend_factory=backend_factory,
    )
    knowledge_engine, conversation_engine, workflow_engine = engines
    service, _registry = _configure_server(
        monkeypatch,
        knowledge_engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
    )
    workflow_id = "python-in-memory-resume-workflow"
    run_id = "python-in-memory-resume-run"
    conversation_id = "python-in-memory-resume-conversation"
    turn_node_id = "python-in-memory-resume-turn"
    owner = "python-in-memory-resume-owner"
    identity = claims_ctx.set(
        {
            "ns": "workflow",
            "role": "rw",
            "capabilities": ["workflow.design.write", "write_graph"],
            "sub": owner,
        }
    )
    try:
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="start",
            label="Start",
            op="python-resume-start",
            start=True,
        )
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="gate",
            label="Wait for approval",
            op="python-resume-gate",
        )
        service.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=owner,
            node_id="end",
            label="Finish",
            op="python-resume-end",
            terminal=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=owner,
            edge_id="start-gate",
            src="start",
            dst="gate",
            relation="wf_next",
            is_default=True,
        )
        service.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=owner,
            edge_id="gate-end",
            src="gate",
            dst="end",
            relation="wf_next",
            is_default=True,
        )
    finally:
        claims_ctx.reset(identity)

    resolver = MappingStepResolver()
    end_calls = []
    end_calls_lock = threading.Lock()

    @resolver.register("python-resume-start")
    def _start(_context):
        return RunSuccess(state_update=[("u", {"started": True})])

    @resolver.register("python-resume-gate")
    def _gate(_context):
        return RunSuspended(resume_payload={"question": "approve?"})

    @resolver.register("python-resume-end")
    def _end(context):
        with end_calls_lock:
            end_calls.append(context.state_view.get("approved"))
        return RunSuccess(
            state_update=[
                ("u", {"ended": True, "approved": context.state_view.get("approved")})
            ]
        )

    monkeypatch.setattr(
        "kogwistar.conversation.resolvers.default_resolver", resolver
    )
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )
    suspended = runtime.run(
        workflow_id=workflow_id,
        conversation_id=conversation_id,
        turn_node_id=turn_node_id,
        initial_state={
            "conversation_id": conversation_id,
            "user_id": owner,
            "turn_node_id": turn_node_id,
            "turn_index": 0,
            "role": "user",
            "user_text": "approve",
            "mem_id": "memory",
        },
        run_id=run_id,
    )
    assert suspended.status == "suspended"
    checkpoints = conversation_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    assert checkpoints
    latest_checkpoint = max(
        checkpoints,
        key=lambda node: int(getattr(node, "metadata", {}).get("step_seq", -1)),
    )
    state_json = latest_checkpoint.metadata["state_json"]
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    suspended_tokens = state_json["_rt_join"]["suspended"]
    assert len(suspended_tokens) == 1
    suspended_node_id, _, suspended_token_id, _ = suspended_tokens[0]

    syscall_request = {
        "version": "v1",
        "op": "resume",
        "args": {
            "run_id": run_id,
            "suspended_node_id": suspended_node_id,
            "suspended_token_id": suspended_token_id,
            "client_result": {
                "status": "success",
                "state_update": [["u", {"approved": True}]],
            },
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "user_id": owner,
        },
    }
    with TestClient(server.app) as client:
        resume_headers = _token_header(
            client,
            role="rw",
            ns="workflow",
            username=owner,
            capabilities="workflow.run.write,approve_action",
        )

    original_load_checkpoint = run_execution_module.load_checkpoint
    checkpoint_read_barrier = threading.Barrier(2)

    def synchronized_load_checkpoint(*args, **kwargs):
        checkpoint = original_load_checkpoint(*args, **kwargs)
        checkpoint_read_barrier.wait(timeout=10)
        return checkpoint

    monkeypatch.setattr(
        run_execution_module, "load_checkpoint", synchronized_load_checkpoint
    )

    def resume_over_http():
        with TestClient(server.app) as client:
            return client.post(
                "/api/syscall/v1/resume",
                json=syscall_request,
                headers=resume_headers,
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(resume_over_http) for _ in range(2)]
            responses = [future.result(timeout=30) for future in futures]
    finally:
        monkeypatch.setattr(
            run_execution_module, "load_checkpoint", original_load_checkpoint
        )

    successful_responses = [response for response in responses if response.status_code == 200]
    rejected_responses = [response for response in responses if response.status_code == 500]
    assert len(successful_responses) == 1, [response.text for response in responses]
    assert len(rejected_responses) == 1, [response.text for response in responses]
    assert "owned until" in rejected_responses[0].text
    assert end_calls == [True]

    response = successful_responses[0]
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["op"] == "resume"
    assert payload["status"] == "ok"
    result = payload["result"]
    assert set(resume_contract["required_fields"]).issubset(result)
    assert result["workflow_status"] == "succeeded", json.dumps(result, sort_keys=True)
    assert result["workflow_status"] in resume_contract["completed_statuses"]
    assert result["final_state"]["started"] is True
    assert result["final_state"]["approved"] is True
    assert result["final_state"]["ended"] is True
    for engine in engines:
        engine.close()


def test_resume_route_preserves_response_contract_with_rust_runtime_selection(
    monkeypatch, tmp_path
):
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(selector, raising=False)
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    monkeypatch.setenv("KOGWISTAR_RUST_RUNTIME_URL", "http://rust-runtime")

    engine, conversation_engine, workflow_engine = build_engine_triplet(
        root=tmp_path / "rust-selected-resume",
        embedding_function=FakeEmbeddingFunction(),
    )
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    monkeypatch.setattr(
        "kogwistar.server.chat_service_run_execution.load_checkpoint",
        lambda **_kwargs: {"budget": {}},
    )
    authority_calls = []

    class _FakeRustRuntimeAuthority:
        def __init__(self, *, runtime, base_url, cache_dir):
            authority_calls.append(("init", base_url))

        def resume(self, **kwargs):
            authority_calls.append(("resume", kwargs))
            return RunResult(
                run_id=kwargs["run_id"],
                final_state={"approved": True},
                mq=queue.Queue(),
                status="succeeded",
            )

        def close(self):
            authority_calls.append(("close", None))

    monkeypatch.setattr(
        "kogwistar.runtime.rust_runtime_authority.RustRuntimeAuthority",
        _FakeRustRuntimeAuthority,
    )
    request_body = {
        "suspended_node_id": "approval-step",
        "suspended_token_id": "approval-token",
        "client_result": {"status": "success", "state_update": []},
        "workflow_id": "rust-selected-workflow",
        "conversation_id": "rust-selected-conversation",
        "turn_node_id": "rust-selected-turn",
        "user_id": "not-an-authority-source",
    }

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="rw", ns="workflow")
        denied_headers = _token_header(client, role="rw", ns="wisdom")
        response = client.post(
            "/api/runs/rust-selected-resume-run/resume",
            json=request_body,
            headers=workflow_headers,
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "workflow_status": "succeeded",
            "final_state": {"approved": True},
            "budget": {
                "token_used": 0,
                "time_used_ms": 0,
                "cost_budget": 0.0,
                "cost_used": 0.0,
                "budget_kind": "token",
                "budget_scope": "run",
            },
        }
        assert [name for name, _value in authority_calls] == [
            "init",
            "resume",
            "close",
        ]
        resumed = authority_calls[1][1]
        assert resumed["run_id"] == "rust-selected-resume-run"
        assert resumed["client_result"].model_dump()["status"] == "success"
        assert resumed["workflow_id"] == request_body["workflow_id"]
        assert resumed["conversation_id"] == request_body["conversation_id"]
        assert resumed["turn_node_id"] == request_body["turn_node_id"]

        denied = client.post(
            "/api/runs/rust-selected-resume-run/resume",
            json=request_body,
            headers=denied_headers,
        )
        assert denied.status_code == 403
        assert "rust-selected-resume-run" not in denied.text
        assert [name for name, _value in authority_calls] == [
            "init",
            "resume",
            "close",
        ]


def test_run_replay_uses_python_defaults_and_checks_namespace(monkeypatch, engine_triplet):
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
        raise AssertionError("Python-only run replay attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-run-replay"
    _seed_checkpoint(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=0,
        state={"counter": 1},
    )
    _seed_step(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=1,
        op="increment",
        state_update=[["u", {"counter": 2}]],
    )

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-replay-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(
            f"/api/runs/{run_id}/replay?target_step_seq=1",
            headers=workflow_headers,
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert set(contract["required_fields"]).issubset(payload)
        assert payload["run_id"] == run_id
        assert payload["target_step_seq"] == 1
        assert payload["state"] == {"counter": 2}

        denied = client.get(
            f"/api/runs/{run_id}/replay?target_step_seq=1", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_resume_contract_uses_python_defaults_and_checks_namespace(
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
        raise AssertionError("Python-only resume-contract read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-resume-contract-run"
    _seed_checkpoint(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=0,
        state={"answer": "persisted"},
    )

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-resume-contract-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(
            f"/api/runs/{run_id}/resume-contract", headers=workflow_headers
        )
        assert response.status_code == fixture["success_status"], response.text
        payload = response.json()
        assert set(fixture["required_fields"]).issubset(payload)
        assert payload["run_id"] == run_id
        assert payload["latest_checkpoint_step_seq"] == 0
        assert payload["checkpoint_schema_version"] == 1
        assert "checkpoint_schema_version" in payload["persisted_keys"]
        assert payload["supported_wait_reasons"] == fixture[
            "supported_wait_reasons"
        ]
        workflow_response = client.get(
            f"/api/workflow/runs/{run_id}/resume-contract",
            headers=workflow_headers,
        )
        assert workflow_response.status_code == response.status_code
        assert workflow_response.json() == payload

        denied = client.get(
            f"/api/runs/{run_id}/resume-contract", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_run_steps_uses_python_defaults_and_checks_namespace(monkeypatch, engine_triplet):
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
        raise AssertionError("Python-only workflow-step read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-workflow-steps-run"
    _seed_step(
        conversation_engine,
        run_id=run_id,
        workflow_id="python-only-workflow",
        step_seq=0,
        op="prepare",
        state_update=[["u", {"answer": "persisted"}]],
        result_fields={"answer": "persisted"},
        trace_context={
            "trace_id": "trace-seed",
            "span_id": "span-step-0",
            "parent_span_id": "span-parent",
            "attempt": 2,
        },
    )

    with TestClient(server.app) as client:
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-step-inspection-v1.json"
            ).read_text(encoding="utf-8")
        )
        trace_fixture = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-step-trace-context-v1.json"
            ).read_text(encoding="utf-8")
        )

        response = client.get(f"/api/runs/{run_id}/steps", headers=workflow_headers)
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert payload["run_id"] == run_id
        assert len(payload["steps"]) == 1
        step = payload["steps"][0]
        assert set(contract["required_step_fields"]).issubset(step)
        assert step["step_seq"] == 0
        assert step["workflow_id"] == "python-only-workflow"
        assert step["op"] == "prepare"
        assert step["status"] == contract["step_status"]
        assert step["result"]["answer"] == "persisted"
        assert step["trace_context"] == trace_fixture["trace_context"]
        workflow_response = client.get(
            f"/api/workflow/runs/{run_id}/steps", headers=workflow_headers
        )
        assert workflow_response.status_code == response.status_code
        assert workflow_response.json() == payload

        denied = client.get(f"/api/runs/{run_id}/steps", headers=denied_headers)
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_run_steps_matches_shared_rust_projection_fixture(monkeypatch, engine_triplet):
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

    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-step-inspection-v1.json"
        ).read_text(encoding="utf-8")
    )
    seed = contract["parity_seed"]
    _seed_step(
        conversation_engine,
        run_id=seed["run_id"],
        workflow_id=seed["workflow_id"],
        step_seq=seed["step_seq"],
        op=seed["op"],
        state_update=seed["state_update"],
        result_fields=seed["result_fields"],
        trace_context=seed["trace_context"],
    )

    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="workflow")
        response = client.get(
            f"/api/runs/{seed['run_id']}/steps", headers=headers
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert payload["run_id"] == seed["run_id"]
        assert len(payload["steps"]) == 1
        expected = contract["parity_step"]
        actual = payload["steps"][0]
        assert {field: actual[field] for field in expected} == expected


def test_chat_rest_events_poll_returns_run_events(monkeypatch, engine_triplet):
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
        raise AssertionError("Python-only run-event polling attempted to load Rust")

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

        created = client.post(
            "/api/conversations", json={"user_id": "u-poll"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "u-poll", "text": "poll this run"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

        _wait_for_status(client, run_id, ro_headers, {"succeeded"})

        polled = client.get(
            f"/api/runs/{run_id}/events/poll?after_seq=0&limit=10",
            headers=ro_headers,
        )
        contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-events-poll-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert polled.status_code == contract["success_status"]
        payload = polled.json()
        assert set(contract["required_fields"]).issubset(payload)
        assert payload["run_id"] == run_id
        assert payload["events"]
        assert len(payload["events"]) <= 10
        assert payload["events"][0]["event_type"] == "run.created"
        assert set(contract["required_event_fields"]).issubset(payload["events"][0])


def test_chat_rest_events_poll_matches_rust_seeded_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/run-events-poll-v1.json"
        ).read_text(encoding="utf-8")
    )
    registry.create_run(
        run_id="run-1",
        conversation_id="conversation-1",
        workflow_id="workflow-1",
        user_id="user-1",
        user_turn_node_id="turn-1",
        status="running",
    )
    registry.append_event("run-1", "run.started", {"source": "test"})

    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="conversation", username="user-1")
        response = client.get(
            "/api/runs/run-1/events/poll?after_seq=0&limit=10",
            headers=headers,
        )
        invalid_limit = client.get(
            "/api/runs/run-1/events/poll?limit=invalid", headers=headers
        )
        assert invalid_limit.status_code == 422
        detail = invalid_limit.json()["detail"][0]
        assert {key: detail[key] for key in ("type", "loc", "input")} == {
            "type": "int_parsing",
            "loc": ["query", "limit"],
            "input": "invalid",
        }
        invalid_after_seq = client.get(
            "/api/runs/run-1/events/poll?after_seq=invalid", headers=headers
        )
        assert invalid_after_seq.status_code == 422

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    assert body["run_id"] == "run-1"
    assert len(body["events"]) == 1
    event = body["events"][0]
    assert set(event) == set(contract["required_event_fields"])
    assert event["seq"] == 1
    assert event["event_type"] == "run.started"
    assert event["payload"] == {"source": "test"}
    assert isinstance(event["created_at_ms"], int)


def test_chat_rest_sse_reconnects_cleanly_after_terminal_run(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post(
            "/api/conversations", json={"user_id": "u-terminal-sse"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "u-terminal-sse", "text": "finish this run"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

        final_run = _wait_for_status(client, run_id, ro_headers, {"succeeded"})
        assert final_run["terminal"] is True

        polled = client.get(
            f"/api/runs/{run_id}/events/poll?after_seq=0&limit=100",
            headers=ro_headers,
        )
        polled.raise_for_status()
        events = polled.json()["events"]
        assert events
        last_seq = int(events[-1]["seq"])

        with client.stream(
            "GET",
            f"/api/runs/{run_id}/events?after_seq={last_seq}",
            headers=ro_headers,
            timeout=5,
        ) as resp:
            resp.raise_for_status()
            body = "".join(resp.iter_text())

        assert body == ""


def test_chat_rest_sse_matches_rust_seeded_contract(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/run-events-sse-v1.json"
        ).read_text(encoding="utf-8")
    )
    registry.create_run(
        run_id="run-1",
        conversation_id="conversation-1",
        workflow_id="workflow-1",
        user_id="user-1",
        user_turn_node_id="turn-1",
        status="running",
    )
    registry.append_event("run-1", "run.started", {"source": "test"})
    registry.update_status(run_id="run-1", status="succeeded", finished=True)

    with TestClient(server.app) as client:
        headers = _token_header(client, role="ro", ns="conversation", username="user-1")
        with client.stream(
            "GET", "/api/runs/run-1/events?after_seq=0", headers=headers, timeout=5
        ) as response:
            assert response.status_code == contract["success_status"]
            assert response.headers["content-type"].startswith(contract["media_type"])
            body = "".join(response.iter_text())

    lines = body.splitlines()
    assert f"event: {contract['event_name']}" in lines
    assert "id: 1" in lines
    data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
    assert set(contract["required_data_fields"]).issubset(data)
    assert data["run_id"] == "run-1"
    assert data["event_type"] == contract["event_name"]
    assert data["source"] == "test"


def test_chat_rest_debug_rag_workflow_returns_seeded_knowledge(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _seed_knowledge_node(
        engine,
        node_id="kg-alpha",
        label="Alpha node",
        summary="Alpha summary for debug retrieval.",
    )
    _seed_knowledge_node(
        engine,
        node_id="kg-beta",
        label="Beta node",
        summary="Beta summary for debug retrieval.",
    )
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _debug_rag_runner
    )
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post(
            "/api/conversations", json={"user_id": "u-debug-rag"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={
                "user_id": "u-debug-rag",
                "text": "show me debug rag results",
                "workflow_id": DEBUG_RAG_WORKFLOW_ID,
            },
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

        final_run = _wait_for_status(client, run_id, ro_headers, {"succeeded"})
        assert final_run["assistant_turn_node_id"]

        turns = client.get(
            f"/api/conversations/{conversation_id}/turns", headers=ro_headers
        )
        turns.raise_for_status()
        transcript = turns.json()["turns"]
        assistant_text = transcript[-1]["content"]
        assert "Debug RAG response" in assistant_text
        assert "kg-alpha" in assistant_text

        events = client.get(
            f"/api/runs/{run_id}/events/poll?after_seq=0&limit=50",
            headers=ro_headers,
        )
        events.raise_for_status()
        event_names = [evt["event_type"] for evt in events.json()["events"]]
        assert "run.stage" in event_names
        assert "reasoning.summary" in event_names

        projected = conversation_engine.get_nodes(
            where={
                "$and": [
                    {"conversation_id": conversation_id},
                    {"entity_type": "knowledge_reference"},
                ]
            },
            limit=20,
        )
        assert projected


def test_chat_rest_cancelled_run_persists_no_assistant(monkeypatch, engine_triplet):
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
        raise AssertionError("Python-only run cancellation attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    cancel_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "run-cancel-v1.json"
        ).read_text(encoding="utf-8")
    )
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post(
            "/api/conversations", json={"user_id": "u-cancel"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "u-cancel", "text": "stop this answer"},
            headers=rw_headers,
        )
        run_id = submit.json()["run_id"]

        cancel = client.post(f"/api/runs/{run_id}/cancel", headers=rw_headers)
        assert cancel.status_code == cancel_contract["accepted_status"]
        cancel_payload = cancel.json()
        assert set(cancel_contract["required_fields"]).issubset(cancel_payload)
        assert cancel_payload["run_id"] == run_id
        assert cancel_payload["cancel_requested"] is True
        assert cancel_payload["status"] in cancel_contract["accepted_run_statuses"]
        assert cancel_payload["terminal"] is (
            cancel_payload["status"] in cancel_contract["terminal_statuses"]
        )
        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1

        final_run = _wait_for_status(client, run_id, ro_headers, {"cancelled"})
        assert final_run["assistant_turn_node_id"] in (None, "")
        assert "last_step_seq" not in final_run
        assert "step_count" not in final_run

        turns = client.get(
            f"/api/conversations/{conversation_id}/turns", headers=ro_headers
        )
        turns.raise_for_status()
        transcript = turns.json()["turns"]
        assert [turn["role"] for turn in transcript] == ["user"]

        events = _collect_sse_events(client, run_id, ro_headers)
        names = [name for name, _payload in events]
        assert cancel_contract["nonterminal_cancel_event"] in names
        assert names[-1] == "run.cancelled"


def test_chat_debug_endpoints_namespace_and_workflow_viz(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        docs_headers = _token_header(client, role="rw", ns="docs")
        conv_headers = _token_header(client, role="rw", ns="conversation")
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        snapshot_owner_headers = _token_header(
            client, role="ro", ns="workflow", username="u-snap"
        )

        forbidden = client.post(
            "/api/conversations", json={"user_id": "u-docs"}, headers=docs_headers
        )
        assert forbidden.status_code == 403

        created = client.post(
            "/api/conversations", json={"user_id": "u-snap"}, headers=conv_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]
        start_node_id = created.json()["start_node_id"]

        svc = ConversationService.from_engine(
            conversation_engine,
            knowledge_engine=engine,
            workflow_engine=workflow_engine,
        )
        snapshot_id = svc.persist_context_snapshot(
            conversation_id=conversation_id,
            run_id="snap-run",
            run_step_seq=0,
            stage="draft_answer",
            view=SimpleNamespace(
                messages=[{"role": "user", "content": "hello snapshot"}],
                items=[SimpleNamespace(node_id=start_node_id)],
                cost=None,
                token_budget=128,
            ),
            model_name="fake-model",
            budget_tokens=128,
        )
        conversation_engine.acl.record_acl(
            grain="node",
            truth_graph="conversation",
            entity_id=snapshot_id,
            version=2,
            mode="private",
            owner_id="u-snap",
            supersedes_version=1,
        )

        latest = client.get(
            f"/api/conversations/{conversation_id}/snapshots/latest",
            headers=snapshot_owner_headers,
        )
        assert latest.status_code == 200
        latest.raise_for_status()
        snapshot = latest.json()
        assert set(snapshot) == {
            "snapshot_node_id",
            "conversation_id",
            "metadata",
            "properties",
        }
        assert snapshot["snapshot_node_id"] == snapshot_id
        assert snapshot["conversation_id"] == conversation_id
        assert isinstance(snapshot["metadata"], dict)
        assert isinstance(snapshot["properties"], dict)

        run_id = "seed-run"
        workflow_id = "wf.test"
        _seed_checkpoint(
            conversation_engine,
            run_id=run_id,
            workflow_id=workflow_id,
            step_seq=0,
            state={"counter": 1},
        )
        _seed_step(
            conversation_engine,
            run_id=run_id,
            workflow_id=workflow_id,
            step_seq=0,
            op="prepare",
            state_update=[["u", {"answer": "worker"}]],
            result_fields={"answer": "worker"},
            trace_context={
                "trace_id": "trace-seed",
                "span_id": "span-step-0",
                "parent_span_id": "span-parent",
                "attempt": 2,
            },
        )
        _seed_step(
            conversation_engine,
            run_id=run_id,
            workflow_id=workflow_id,
            step_seq=1,
            op="persist",
            state_update=[["u", {"counter": 2}]],
        )

        steps = client.get(f"/api/runs/{run_id}/steps", headers=workflow_headers)
        steps.raise_for_status()
        steps_payload = steps.json()
        steps_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-step-inspection-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert steps.status_code == steps_contract["success_status"]
        assert steps_payload["run_id"] == run_id
        assert [step["step_seq"] for step in steps_payload["steps"]] == [0, 1]
        for field in steps_contract["required_step_fields"]:
            assert field in steps_payload["steps"][0]
        assert steps_payload["steps"][0]["op"] == "prepare"
        assert steps_payload["steps"][0]["status"] == steps_contract["step_status"]
        assert (
            steps_payload["steps"][0]["result"] == steps_contract["successful_result"]
        )
        assert isinstance(steps_payload["steps"][0]["duration_ms"], int)
        trace_fixture = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/workflow-step-trace-context-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert (
            steps_payload["steps"][0]["trace_context"] == trace_fixture["trace_context"]
        )
        assert ("trace_context" not in steps_payload["steps"][1]) is steps_contract[
            "legacy_step_omits_trace_context"
        ]

        checkpoints = client.get(
            f"/api/runs/{run_id}/checkpoints", headers=workflow_headers
        )
        checkpoint_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-checkpoints-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert checkpoints.status_code == checkpoint_contract["success_status"]
        checkpoints_payload = checkpoints.json()
        assert set(checkpoint_contract["required_envelope_fields"]).issubset(
            checkpoints_payload
        )
        checkpoint_items = checkpoints_payload["checkpoints"]
        assert checkpoint_items
        for checkpoint_item in checkpoint_items:
            assert set(checkpoint_contract["required_checkpoint_fields"]).issubset(
                checkpoint_item
            )
            assert isinstance(checkpoint_item["state"], dict)
        assert checkpoint_items[0]["state"]["counter"] == 1

        checkpoint = client.get(
            f"/api/runs/{run_id}/checkpoints/0", headers=workflow_headers
        )
        checkpoint_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-checkpoint-detail-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert checkpoint.status_code == checkpoint_contract["success_status"]
        checkpoint_payload = checkpoint.json()
        assert set(checkpoint_contract["required_fields"]).issubset(checkpoint_payload)
        assert checkpoint_payload["run_id"] == run_id
        assert checkpoint_payload["step_seq"] == 0
        assert checkpoint_payload["state"]["counter"] == 1

        replay = client.get(
            f"/api/runs/{run_id}/replay?target_step_seq=1", headers=workflow_headers
        )
        replay_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-replay-v1.json"
            ).read_text(encoding="utf-8")
        )
        assert replay.status_code == replay_contract["success_status"]
        replay_payload = replay.json()
        assert set(replay_contract["required_fields"]).issubset(replay_payload)
        assert replay_payload["run_id"] == run_id
        assert replay_payload["target_step_seq"] == 1
        assert replay_payload["state"]["counter"] == 2

        resume_contract_fixture = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/run-resume-contract-v1.json"
            ).read_text(encoding="utf-8")
        )
        empty_resume = client.get(
            "/api/runs/no-checkpoint-run/resume-contract",
            headers=workflow_headers,
        )
        assert empty_resume.status_code == resume_contract_fixture["success_status"]
        empty_resume_payload = empty_resume.json()
        assert set(resume_contract_fixture["required_fields"]).issubset(
            empty_resume_payload
        )
        for key, expected in resume_contract_fixture["no_checkpoint"].items():
            assert empty_resume_payload[key] == expected
        assert (
            empty_resume_payload["supported_wait_reasons"]
            == resume_contract_fixture["supported_wait_reasons"]
        )

        resume_contract = client.get(
            f"/api/runs/{run_id}/resume-contract", headers=workflow_headers
        )
        assert resume_contract.status_code == resume_contract_fixture["success_status"]
        contract = resume_contract.json()
        assert set(resume_contract_fixture["required_fields"]).issubset(contract)
        assert contract["run_id"] == run_id
        assert contract["latest_checkpoint_step_seq"] == 0
        assert contract["checkpoint_schema_version"] == 1
        assert "checkpoint_schema_version" in contract["persisted_keys"]
        assert "_rt_join" in contract["ephemeral_keys"]
        assert "approval" in contract["supported_wait_reasons"]
        assert "rate_window" in contract["supported_wait_reasons"]

        viz_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/graph-viz-d3-http-success-v1.json"
            ).read_text(encoding="utf-8")
        )
        viz = client.get("/api/viz/d3.json?mode=reify")
        assert viz.status_code == viz_contract["success_status"]
        payload = viz.json()
        assert set(viz_contract["required_fields"]).issubset(payload)
        for field in viz_contract["array_fields"]:
            assert isinstance(payload[field], list)
        assert payload["mode"] == viz_contract["mode"]
        assert payload["doc_id"] is viz_contract["doc_id"]

        cytoscape_contract = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "contracts/golden/graph-viz-cytoscape-http-success-v1.json"
            ).read_text(encoding="utf-8")
        )
        cytoscape = client.get(
            "/api/viz/cytoscape.json",
            params={"mode": cytoscape_contract["mode"]},
            headers=workflow_headers,
        )
        assert cytoscape.status_code == cytoscape_contract["success_status"]
        cytoscape_payload = cytoscape.json()
        assert set(cytoscape_contract["required_fields"]).issubset(cytoscape_payload)
        for field in cytoscape_contract["array_fields"]:
            assert isinstance(cytoscape_payload[field], list)
        assert cytoscape_payload["mode"] == cytoscape_contract["mode"]
        if cytoscape_contract["doc_id_query_echo"]:
            assert cytoscape_payload["doc_id"] is None
