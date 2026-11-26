"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_server_api_support import *  # noqa: F403

def test_workflow_namespace_scope_isolated_across_concurrent_threads(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    original_namespace = workflow_engine.namespace
    first_entered = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()

    def first_scope() -> tuple[bool, str]:
        with service._workflow_namespace_scope("scope-first") as scoped:
            first_namespace = service._workflow_namespace("scope-first")
            assert scoped.write._e.namespace == first_namespace
            assert workflow_engine.namespace == original_namespace
            first_entered.set()
            assert second_attempted.wait(timeout=2)
            entered_concurrently = second_entered.wait(timeout=0.1)
            observed_namespace = scoped.write._e.namespace
            return entered_concurrently, observed_namespace

    def second_scope() -> str:
        assert first_entered.wait(timeout=2)
        second_attempted.set()
        with service._workflow_namespace_scope("scope-second") as scoped:
            second_entered.set()
            assert workflow_engine.namespace == original_namespace
            return scoped.write._e.namespace

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(first_scope)
        second = executor.submit(second_scope)
        first_result = first.result(timeout=3)
        second_namespace = second.result(timeout=3)

    assert first_result == (False, service._workflow_namespace("scope-first"))
    assert second_namespace == service._workflow_namespace("scope-second")
    assert workflow_engine.namespace == original_namespace


def test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-submit-http-v1.json"
        ).read_text(encoding="utf-8")
    )

    def assert_submit_contract(response, expected_conversation_id, expected_workflow_id, turn_node_id):
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert all(field in payload for field in contract["required_common_fields"])
        assert isinstance(payload["run_id"], str) and payload["run_id"]
        assert payload["conversation_id"] == expected_conversation_id
        assert payload["workflow_id"] == expected_workflow_id
        assert payload["turn_node_id"] == turn_node_id
        for field, value in contract["expected_common_values"].items():
            assert payload[field] == value

    captured: list[str] = []

    def _runtime_runner(req: RuntimeRunRequest) -> dict[str, object]:
        captured.append(str(getattr(req, "runtime_kind", "")))
        return {
            "workflow_status": "succeeded",
            "final_state": {"runtime_kind": getattr(req, "runtime_kind", "")},
        }

    service, _registry = _configure_server(
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
            json={"user_id": "runtime-kind-user"},
            headers=headers,
        )
        assert conv.status_code == 200, conv.text
        conversation_id = str(conv.json()["conversation_id"])

        async_resp = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "wf.runtime.kind.async",
                "conversation_id": conversation_id,
                "turn_node_id": "turn-contract-async",
                "initial_state": {},
                "runtime_kind": "async",
            },
            headers=headers,
        )
        assert_submit_contract(
            async_resp,
            conversation_id,
            "wf.runtime.kind.async",
            "turn-contract-async",
        )
        async_run_id = str(async_resp.json()["run_id"])
        _wait_for_status(
            client,
            async_run_id,
            headers,
            {"succeeded"},
            path_template="/api/workflow/runs/{run_id}",
        )

        default_resp = client.post(
            "/api/workflow/runs",
            json={
                "workflow_id": "wf.runtime.kind.sync",
                "conversation_id": conversation_id,
                "turn_node_id": "turn-contract-default",
                "initial_state": {},
            },
            headers=headers,
        )
        assert_submit_contract(
            default_resp,
            conversation_id,
            "wf.runtime.kind.sync",
            "turn-contract-default",
        )
        default_run_id = str(default_resp.json()["run_id"])
        _wait_for_status(
            client,
            default_run_id,
            headers,
            {"succeeded"},
            path_template="/api/workflow/runs/{run_id}",
        )

    assert captured == ["async", "sync"]
    assert service is not None


def test_workflow_run_submit_requires_both_admission_capabilities(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/workflow-run-submit-auth-v1.json"
        ).read_text(encoding="utf-8")
    )

    with TestClient(server.app) as client:
        full_capability_headers = _token_header(
            client,
            role="rw",
            ns="workflow,conversation",
            username="run-auth-owner",
            capabilities=",".join(contract["required_capabilities"]),
        )
        conversation = client.post(
            "/api/conversations",
            json={"user_id": "run-auth-owner"},
            headers=full_capability_headers,
        )
        assert conversation.status_code == 200, conversation.text
        conversation_id = conversation.json()["conversation_id"]

        for missing_capability in contract["required_capabilities"]:
            granted = [
                capability
                for capability in contract["required_capabilities"]
                if capability != missing_capability
            ]
            headers = _token_header(
                client,
                role="rw",
                ns="workflow",
                username="run-auth-owner",
                capabilities=",".join(granted),
            )
            response = client.post(
                "/api/workflow/runs",
                json={
                    "workflow_id": "wf.run.auth.contract",
                    "conversation_id": conversation_id,
                    "initial_state": {},
                },
                headers=headers,
            )
            assert response.status_code == contract["denied_status"], response.text


def test_default_runtime_runner_uses_async_or_sync_runtime_by_kind(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    from kogwistar.server.run_registry import RunRegistry

    registry = RunRegistry(conversation_engine.meta_sqlite)
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=_debug_rag_runner,
    )

    calls: list[str] = []

    class _FakeSyncRuntime:
        def __init__(self, **kwargs):
            calls.append("sync-init")
            self.kwargs = kwargs

        def run(self, **kwargs):
            calls.append("sync-run")
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-sync"),
                final_state={"selected_runtime": "sync"},
                mq=queue.Queue(),
                status="succeeded",
            )

    class _FakeAsyncRuntime:
        def __init__(self, **kwargs):
            calls.append("async-init")
            self.kwargs = kwargs

        async def run(self, **kwargs):
            calls.append("async-run")
            return RunResult(
                run_id=str(kwargs.get("run_id") or "run-async"),
                final_state={"selected_runtime": "async"},
                mq=queue.Queue(),
                status="succeeded",
            )

    monkeypatch.setattr("kogwistar.runtime.runtime.WorkflowRuntime", _FakeSyncRuntime)
    monkeypatch.setattr(
        "kogwistar.runtime.async_runtime.AsyncWorkflowRuntime", _FakeAsyncRuntime
    )

    async_req = RuntimeRunRequest(
        run_id="run-async",
        workflow_id="wf.runtime.kind.async",
        conversation_id="conv-async",
        turn_node_id="turn-async",
        user_id="user-1",
        initial_state={},
        knowledge_engine=engine,
        conversation_engine=conversation_engine,
        workflow_engine=workflow_engine,
        registry=registry,
        publish=lambda *args, **kwargs: {"ok": True},
        is_cancel_requested=lambda: False,
        runtime_kind="async",
    )
    sync_req = RuntimeRunRequest(
        run_id="run-sync",
        workflow_id="wf.runtime.kind.sync",
        conversation_id="conv-sync",
        turn_node_id="turn-sync",
        user_id="user-1",
        initial_state={},
        knowledge_engine=engine,
        conversation_engine=conversation_engine,
        workflow_engine=workflow_engine,
        registry=registry,
        publish=lambda *args, **kwargs: {"ok": True},
        is_cancel_requested=lambda: False,
        runtime_kind="",
    )

    async_out = service._run_execution._default_runtime_runner(async_req)
    sync_out = service._run_execution._default_runtime_runner(sync_req)
    async_default_service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=_debug_rag_runner,
        default_runtime_kind="async",
    )
    blank_default_req = RuntimeRunRequest(
        run_id="run-blank",
        workflow_id="wf.runtime.kind.blank",
        conversation_id="conv-blank",
        turn_node_id="turn-blank",
        user_id="user-1",
        initial_state={},
        knowledge_engine=engine,
        conversation_engine=conversation_engine,
        workflow_engine=workflow_engine,
        registry=registry,
        publish=lambda *args, **kwargs: {"ok": True},
        is_cancel_requested=lambda: False,
        runtime_kind="",
    )
    blank_out = async_default_service._run_execution._default_runtime_runner(
        blank_default_req
    )

    assert async_out["workflow_status"] == "succeeded"
    assert async_out["final_state"]["selected_runtime"] == "async"
    assert sync_out["workflow_status"] == "succeeded"
    assert sync_out["final_state"]["selected_runtime"] == "sync"
    assert blank_out["workflow_status"] == "succeeded"
    assert blank_out["final_state"]["selected_runtime"] == "async"
    assert calls[:6] == [
        "async-init",
        "async-run",
        "sync-init",
        "sync-run",
        "async-init",
        "async-run",
    ]
