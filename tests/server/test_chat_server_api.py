from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from jose import jwt

pytest.importorskip("fastapi")
pytest.importorskip("mcp")
pytest.importorskip("sqlalchemy")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from .auth_env import TEST_JWT_ALG, TEST_JWT_SECRET

os.environ.setdefault("JWT_SECRET", TEST_JWT_SECRET)
os.environ.setdefault("JWT_ALG", TEST_JWT_ALG)

import kogwistar.server_mcp_with_admin as server
from kogwistar.conversation.agentic_answering_design import DEBUG_RAG_WORKFLOW_ID
from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import (
    WorkflowCancelledNode,
    WorkflowCheckpointNode,
    WorkflowCompletedNode,
    WorkflowStepExecNode,
)
from kogwistar.runtime.runtime import RunResult
from kogwistar.runtime.telemetry import EventEmitter, TraceContext
from kogwistar.runtime.telemetry_otel import OpenTelemetrySink
from kogwistar.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from kogwistar.server.run_registry import RunRegistry, RunRegistryLaneMessageEventSink
from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.server_fixtures import build_engine_triplet
from tests._helpers.server_http_helpers import token_header

pytestmark = pytest.mark.ci_full


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


_token_header = token_header


class _FixedResource:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def __getattr__(self, name):
        return getattr(self._value, name)


@pytest.fixture()
def engine_triplet():
    root = Path(".tmp_chat_server_tests") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        yield build_engine_triplet(root=root, embedding_function=ef)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _configure_server(
    monkeypatch,
    engine,
    conversation_engine,
    workflow_engine,
    answer_runner,
    runtime_runner=None,
):
    registry = RunRegistry(conversation_engine.meta_sqlite)
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=answer_runner,
        runtime_runner=runtime_runner,
    )
    monkeypatch.setattr(server, "engine", _FixedResource(engine), raising=False)
    monkeypatch.setattr(
        server,
        "conversation_engine",
        _FixedResource(conversation_engine),
        raising=False,
    )
    monkeypatch.setattr(
        server, "workflow_engine", _FixedResource(workflow_engine), raising=False
    )
    monkeypatch.setattr(server, "run_registry", _FixedResource(registry), raising=False)
    monkeypatch.setattr(server, "chat_service", _FixedResource(service), raising=False)

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    monkeypatch.setattr(
        server.app.router, "lifespan_context", _noop_lifespan, raising=False
    )
    return service, registry


def _wait_for_status(
    client: TestClient,
    run_id: str,
    headers: dict[str, str],
    expected: set[str],
    *,
    path_template: str = "/api/runs/{run_id}",
) -> dict:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        resp = client.get(path_template.format(run_id=run_id), headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        if payload["status"] in expected:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not reach one of {sorted(expected)}")


def _collect_sse_events(
    client: TestClient,
    run_id: str,
    headers: dict[str, str],
    *,
    path_template: str = "/api/runs/{run_id}/events",
) -> list[tuple[str, dict]]:
    with client.stream(
        "GET", path_template.format(run_id=run_id), headers=headers
    ) as resp:
        resp.raise_for_status()
        body = "".join(resp.iter_text())
    events: list[tuple[str, dict]] = []
    current_event: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line.split(": ", 1)[1]
        elif line.startswith("data: ") and current_event is not None:
            events.append((current_event, json.loads(line.split(": ", 1)[1])))
    return events


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
        headers = token_header(client, role="rw", ns="docs,conversation,workflow")
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


def _visible_workflow_design_ids(
    workflow_engine, *, workflow_id: str
) -> tuple[str, set[str], set[str]]:
    start, nodes, adj, _rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )
    edge_ids = {str(edge.id) for edges in adj.values() for edge in edges}
    return str(start.id), {str(node_id) for node_id in nodes.keys()}, edge_ids


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


def _assert_workflow_design_entities_use_scoped_namespace(
    service,
    workflow_engine,
    *,
    workflow_id: str,
    node_ids: set[str],
    edge_ids: set[str],
) -> None:
    expected_namespace = service._workflow_namespace(workflow_id)
    with service._workflow_namespace_scope(workflow_id) as scoped_engine:
        nodes = scoped_engine.read.get_nodes(ids=sorted(node_ids)) if node_ids else []
        edges = scoped_engine.read.get_edges(ids=sorted(edge_ids)) if edge_ids else []
        assert {str(node.id) for node in nodes} == node_ids
        assert {str(edge.id) for edge in edges} == edge_ids
        assert all(
            node.metadata.get("namespace") == expected_namespace for node in nodes
        )
        assert all(
            edge.metadata.get("namespace") == expected_namespace for edge in edges
        )


def _success_runner(req: AnswerRunRequest) -> dict:
    req.publish("run.stage", {"stage": "retrieve"})
    req.publish(
        "reasoning.summary",
        {"stage": "retrieve", "summary": "Retrieving candidate evidence."},
    )
    req.publish("run.stage", {"stage": "draft_answer"})
    req.publish(
        "reasoning.summary",
        {"stage": "draft_answer", "summary": "Drafting the answer."},
    )
    assistant_text = f"Assistant reply: {req.user_text}"
    assistant_turn_node_id = f"assistant|{uuid.uuid4().hex}"
    embedding = req.conversation_engine.iterative_defensive_emb(assistant_text)
    req.conversation_engine.write.add_node(
        ConversationNode(
            id=assistant_turn_node_id,
            label="Assistant turn",
            type="entity",
            summary=assistant_text,
            conversation_id=req.conversation_id,
            role="assistant",  # type: ignore[arg-type]
            turn_index=req.prev_turn_meta_summary.tail_turn_index + 1,
            properties={"content": assistant_text, "entity_type": "assistant_turn"},
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "level_from_root": 0,
                "entity_type": "assistant_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            ),
        )
    )
    req.prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
        assistant_text
    )
    req.prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
    req.prev_turn_meta_summary.tail_turn_index += 1
    return {
        "assistant_turn_node_id": assistant_turn_node_id,
        "assistant_text": assistant_text,
        "workflow_status": "succeeded",
    }


def _debug_rag_runner(req: AnswerRunRequest) -> dict:
    req.publish("run.stage", {"stage": "retrieve"})
    req.publish(
        "reasoning.summary",
        {"stage": "retrieve", "summary": "Retrieving candidate evidence."},
    )
    req.publish("run.stage", {"stage": "draft_answer"})
    req.publish(
        "reasoning.summary",
        {"stage": "draft_answer", "summary": "Drafting the answer."},
    )
    assistant_text = "\n".join(
        [
            "Debug RAG response",
            "",
            f"Question: {req.user_text}",
            "",
            "Retrieved nodes:",
            "1. Alpha node [kg-alpha]",
            "   Alpha summary for debug retrieval.",
            "2. Beta node [kg-beta]",
            "   Beta summary for debug retrieval.",
        ]
    )
    assistant_turn_node_id = f"assistant|{uuid.uuid4().hex}"
    embedding = req.conversation_engine.iterative_defensive_emb(assistant_text)
    req.conversation_engine.write.add_node(
        ConversationNode(
            id=assistant_turn_node_id,
            label="Assistant turn",
            type="entity",
            summary=assistant_text,
            conversation_id=req.conversation_id,
            role="assistant",  # type: ignore[arg-type]
            turn_index=req.prev_turn_meta_summary.tail_turn_index + 1,
            properties={"content": assistant_text, "entity_type": "assistant_turn"},
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "level_from_root": 0,
                "entity_type": "assistant_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=(
                embedding.tolist() if hasattr(embedding, "tolist") else embedding
            ),
        )
    )
    req.conversation_engine.write.add_node(
        ConversationNode(
            id=f"knowledge_reference|{uuid.uuid4().hex}",
            label="Knowledge reference",
            type="reference_pointer",
            summary="Alpha summary for debug retrieval.",
            conversation_id=req.conversation_id,
            role="system",  # type: ignore[arg-type]
            turn_index=None,
            properties={
                "target_namespace": "kg",
                "refers_to_collection": "nodes",
                "target_id": "kg-alpha",
                "entity_type": "knowledge_reference",
            },
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            metadata={
                "level_from_root": 0,
                "entity_type": "knowledge_reference",
                "in_conversation_chain": False,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )
    req.prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
        assistant_text
    )
    req.prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
    req.prev_turn_meta_summary.tail_turn_index += 1
    return {
        "assistant_turn_node_id": assistant_turn_node_id,
        "assistant_text": assistant_text,
        "workflow_status": "succeeded",
    }


def _cancel_runner(req: AnswerRunRequest) -> dict:
    req.publish("run.stage", {"stage": "retrieve"})
    req.publish(
        "reasoning.summary",
        {"stage": "retrieve", "summary": "Retrieving candidate evidence."},
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected the run to be cancelled")


def _runtime_success_runner(req: RuntimeRunRequest) -> dict:
    req.publish("run.stage", {"stage": "execute"})
    req.publish(
        "reasoning.summary",
        {"stage": "execute", "summary": "Executing workflow runtime."},
    )
    if req.is_cancel_requested():
        raise RunCancelledError()
    return {
        "workflow_status": "succeeded",
        "final_state": {
            "workflow_id": req.workflow_id,
            "echo_state": dict(req.initial_state),
        },
    }


def _runtime_lane_message_runner(req: RuntimeRunRequest) -> dict:
    sink = RunRegistryLaneMessageEventSink(registry=req.registry, run_id=req.run_id)
    sink(
        {
            "event_type": "worker.requested",
            "run_id": req.run_id,
            "message_id": "msg:server-runtime-lane",
            "conversation_id": req.conversation_id,
            "inbox_id": "inbox:worker:runtime",
            "sender_id": "lane:foreground",
            "recipient_id": "lane:worker:runtime",
            "msg_type": "request.runtime",
            "status": "pending",
            "correlation_id": "corr:server-runtime-lane",
            "step_id": "0",
            "workflow_node_id": "wf.runtime.lane:send",
            "step_seq": 0,
        }
    )
    return {
        "workflow_status": "succeeded",
        "final_state": {"workflow_id": req.workflow_id},
    }


def _runtime_cancel_runner(req: RuntimeRunRequest) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected runtime workflow to be cancelled")


def test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
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
                "initial_state": {},
                "runtime_kind": "async",
            },
            headers=headers,
        )
        assert async_resp.status_code == 202, async_resp.text
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
                "initial_state": {},
            },
            headers=headers,
        )
        assert default_resp.status_code == 202, default_resp.text
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


def _seed_step(
    conversation_engine,
    *,
    run_id: str,
    workflow_id: str,
    step_seq: int,
    op: str,
    state_update: list,
    result_fields: dict | None = None,
    trace_context: dict | None = None,
):
    result = {
        "conversation_node_id": None,
        "state_update": state_update,
        "update": None,
        "next_step_names": [],
        "workflow_invocations": [],
        "status": "success",
    }
    result.update(result_fields or {})
    node = WorkflowStepExecNode(
        id=f"wf_step|{run_id}|{step_seq}",
        label=f"Step {step_seq}",
        type="entity",
        doc_id=f"wf_step|{run_id}|{step_seq}",
        summary=f"workflow_step_exec {run_id} {step_seq}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_step_exec",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "workflow_node_id": f"wf:{workflow_id}:{op}",
            "step_seq": step_seq,
            "op": op,
            "status": "ok",
            "duration_ms": 1,
            "result_json": json.dumps(result),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    if trace_context is not None:
        node.metadata.update(
            {
                "trace_id": trace_context["trace_id"],
                "span_id": trace_context["span_id"],
                "parent_span_id": trace_context["parent_span_id"],
                "trace_attempt": trace_context["attempt"],
            }
        )
    conversation_engine.write.add_node(node)


def _seed_checkpoint(
    conversation_engine, *, run_id: str, workflow_id: str, step_seq: int, state: dict
):
    node = WorkflowCheckpointNode(
        id=f"wf_ckpt|{run_id}|{step_seq}",
        label=f"Checkpoint {step_seq}",
        type="entity",
        doc_id=f"wf_ckpt|{run_id}|{step_seq}",
        summary=f"workflow_checkpoint {run_id} {step_seq}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_checkpoint",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "step_seq": step_seq,
            "state_json": json.dumps(state),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.write.add_node(node)


def _seed_terminal_node(
    conversation_engine,
    *,
    node_cls,
    entity_type: str,
    run_id: str,
    workflow_id: str,
    conversation_id: str,
):
    node = node_cls(
        id=f"{entity_type}|{run_id}",
        label=entity_type,
        type="entity",
        doc_id=f"{entity_type}|{run_id}",
        summary=f"{entity_type} {run_id}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={"entity_type": entity_type},
        metadata={
            "entity_type": entity_type,
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "accepted_step_seq": 1,
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.write.add_node(node)


def _seed_knowledge_node(
    knowledge_engine,
    *,
    node_id: str,
    label: str,
    summary: str,
):
    node = Node(
        id=node_id,
        label=label,
        type="entity",
        summary=summary,
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=f"doc:{node_id}",
        level_from_root=0,
    )
    knowledge_engine.write.add_node(node)


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
        user_id="owner-detail", conversation_id="conversation-detail"
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
            "/api/conversations/conversation-detail",
            headers=headers("owner-detail", "conversation"),
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload) == {
            "conversation_id",
            "user_id",
            "status",
            "start_node_id",
            "tail_node_id",
            "turn_count",
        }
        assert payload["conversation_id"] == "conversation-detail"
        assert payload["user_id"] == "owner-detail"
        assert payload["status"] == "active"
        assert payload["start_node_id"]
        assert payload["tail_node_id"] == payload["start_node_id"]
        assert payload["turn_count"] == 0

        denied = client.get(
            "/api/conversations/conversation-detail",
            headers=headers("foreign-user", "wisdom"),
        )
        assert denied.status_code == 403
        assert "owner-detail" not in denied.text
        assert "start_node_id" not in denied.text


def test_conversation_transcript_uses_python_defaults_and_checks_namespace(
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
        raise AssertionError("Python-only transcript read attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service._conversation_queries.create_conversation(
        user_id="owner-transcript", conversation_id="conversation-transcript"
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
            "/api/conversations/conversation-transcript/turns",
            headers=headers("owner-transcript", "conversation"),
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "conversation_id": "conversation-transcript",
            "turns": [],
        }

        denied = client.get(
            "/api/conversations/conversation-transcript/turns",
            headers=headers("foreign-user", "wisdom"),
        )
        assert denied.status_code == 403
        assert "owner-transcript" not in denied.text
        assert "conversation-transcript" not in denied.text


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
        workflow_headers = _token_header(client, role="ro", ns="workflow")
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
    engine, conversation_engine, workflow_engine = engine_triplet
    _service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    run_id = "python-only-run-evidence"
    conversation_id = "python-only-evidence-conversation"

    with TestClient(server.app) as client:
        conversation_headers = _token_header(client, role="rw", ns="conversation")
        workflow_headers = _token_header(client, role="ro", ns="workflow")
        denied_headers = _token_header(client, role="ro", ns="wisdom")
        created = client.post(
            "/api/conversations",
            json={"user_id": "python-only-user", "conversation_id": conversation_id},
            headers=conversation_headers,
        )
        assert created.status_code == 200, created.text
        start_node_id = created.json()["start_node_id"]
        registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id="python-only-workflow",
            user_id="python-only-user",
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
            run_step_seq=0,
            stage="answer",
            view=SimpleNamespace(
                messages=[{"role": "user", "content": "evidence test"}],
                items=[SimpleNamespace(node_id=start_node_id)],
                cost=None,
                token_budget=32,
            ),
            model_name="test-model",
            budget_tokens=32,
            evidence_pack_digest={"fixture_items": 1},
        )
        persisted_snapshot_metadata = conversation_engine.read.get_node_metadatas(
            ids=[snapshot_id], limit=1
        )[0]
        assert persisted_snapshot_metadata["used_node_ids"] == [start_node_id]

        response = client.get(
            f"/api/runs/{run_id}/evidence", headers=workflow_headers
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["version"] == "v1"
        assert payload["run_id"] == run_id
        assert payload["conversation_id"] == conversation_id
        assert payload["snapshot_node_id"] == snapshot_id
        assert payload["evidence_refs"] == [start_node_id]
        assert payload["source_summary"] == {
            "used_node_count": 1,
            "evidence_pack_digest": {"fixture_items": 1},
        }

        denied = client.get(
            f"/api/runs/{run_id}/evidence", headers=denied_headers
        )
        assert denied.status_code == 403
        assert run_id not in denied.text


def test_resume_route_uses_python_defaults_and_checks_namespace(
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

        denied = client.get(f"/api/runs/{run_id}/steps", headers=denied_headers)
        assert denied.status_code == 403
        assert run_id not in denied.text


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
        assert "run.cancelling" in names
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

        latest = client.get(
            f"/api/conversations/{conversation_id}/snapshots/latest",
            headers=workflow_headers,
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
        response = client.post(
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

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
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

    assert response.status_code == contract["success_status"], response.text
    body = response.json()
    for field, expected in contract["fixed_response_fields"].items():
        assert body[field] == expected
    for field in contract["integer_response_fields"]:
        assert isinstance(body[field], int)
    for field in contract["absent_response_fields"]:
        assert field not in body


def test_runtime_rest_submit_and_sse(monkeypatch, engine_triplet):
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
        wf_ro = _token_header(client, role="ro", ns="workflow")
        snapshot = client.get("/api/workflow/resources", headers=wf_ro)
        snapshot.raise_for_status()
        payload = snapshot.json()
        assert "scheduler" in payload
        assert "runs" in payload
        assert "budget_model" in payload
        assert "cost_ledger" in payload
        assert "storage_usage_bytes" in payload
        assert "policy_infra" in payload
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

        cancel = client.post(f"/api/workflow/runs/{run_id}/cancel", headers=wf_rw)
        assert cancel.status_code == 202

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
            id="tool-call-1",
            label="tool_call:list_transcript",
            type="entity",
            doc_id="tool-call-1",
            summary="Calling tool list_transcript",
            role="assistant",
            turn_index=1,
            conversation_id="conv-tools",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"tool_name": "list_transcript"},
            metadata={
                "entity_type": "tool_call",
                "tool_name": "list_transcript",
                "level_from_root": 0,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
    )
    conversation_engine.write.add_node(
        ConversationNode(
            user_id="tester",
            id="tool-result-1",
            label="tool_result:list_transcript",
            type="entity",
            doc_id="tool-result-1",
            summary="[]",
            role="tool",
            turn_index=1,
            conversation_id="conv-tools",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"tool_name": "list_transcript", "result_json": "[]"},
            metadata={
                "entity_type": "tool_result",
                "tool_name": "list_transcript",
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

        budget = client.get("/api/workflow/budget/history", headers=wf_ro)
        budget.raise_for_status()
        assert budget.json()["cost_ledger"]["event_count"] == 2
        assert len(budget.json()["events"]) == 2

        sched = client.get("/api/workflow/scheduler/timeline", headers=wf_ro)
        sched.raise_for_status()
        assert "events" in sched.json()

        audit = client.get(
            "/api/workflow/tools/audit",
            params={"conversation_id": "conv-tools"},
            headers=wf_ro,
        )
        audit.raise_for_status()
        items = audit.json()["items"]
        assert {item["entity_type"] for item in items} == {"tool_call", "tool_result"}


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
    engine, conversation_engine, workflow_engine = engine_triplet
    _service, registry = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    expected_ids = {
        "run_id": "run-status-contract",
        "conversation_id": "conversation-status-contract",
        "workflow_id": "workflow-status-contract",
        "user_id": "user-status-contract",
        "user_turn_node_id": "turn-status-contract",
    }
    registry.create_run(**expected_ids, status="succeeded")

    with TestClient(server.app) as client:
        response = client.get(
            f"/api/runs/{expected_ids['run_id']}",
            headers=_token_header(client, role="ro", ns="conversation"),
        )

    assert response.status_code == contract["success_status"]
    payload = response.json()
    assert set(contract["required_fields"]).issubset(payload)
    assert {key: payload[key] for key in expected_ids} == expected_ids
    assert payload["status"] == "succeeded"
    assert payload["terminal"] is (payload["status"] in contract["terminal_statuses"])
    assert payload["cancel_requested"] is False
    assert payload["result"] is None
    assert payload["error"] is None
    assert isinstance(payload["created_at_ms"], int)
    assert isinstance(payload["updated_at_ms"], int)


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


def test_designer_capabilities_endpoint(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
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
