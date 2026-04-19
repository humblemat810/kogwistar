from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")
pytest.importorskip("sqlalchemy")
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET", "dev-secret")
os.environ.setdefault("JWT_ALG", "HS256")

import kogwistar.server_mcp_with_admin as server
from kogwistar.conversation.agentic_answering_design import DEBUG_RAG_WORKFLOW_ID
from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import (
    WorkflowCancelledNode,
    WorkflowCompletedNode,
    WorkflowCheckpointNode,
    WorkflowStepExecNode,
)
from kogwistar.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from kogwistar.server.run_registry import RunRegistry

pytestmark = pytest.mark.ci_full


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.is_legacy = False

    def __call__(self, input):
        return [[0.01] * self._dim for _ in input]


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
        yield (
            GraphKnowledgeEngine(
                persist_directory=str(root / "kg"),
                kg_graph_type="knowledge",
                embedding_function=ef,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "conversation"),
                kg_graph_type="conversation",
                embedding_function=ef,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "workflow"),
                kg_graph_type="workflow",
                embedding_function=ef,
            ),
        )
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


def _token_header(
    client: TestClient, *, role: str, ns: str, username: str = "tester"
) -> dict[str, str]:
    resp = client.post(
        "/auth/dev-token",
        json={"username": username, "role": role, "ns": ns},
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


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
        headers = _token_header(client, role="rw", ns="docs,conversation,workflow")
        resp = client.post(
            "/api/document.upsert_tree",
            json={
                "doc_id": "doc::upsert_tree_dispatch",
                "insertion_method": "pytest",
                "nodes": [],
                "edges": [],
            },
            headers=headers,
        )

    resp.raise_for_status()
    payload = resp.json()
    assert payload["status"] == "ok"
    assert captured == {
        "doc_id": "doc::upsert_tree_dispatch",
        "parsed_type": "GraphExtractionWithIDs",
        "mode": "append",
        "node_count": 0,
        "edge_count": 0,
    }


def _visible_workflow_design_ids(
    workflow_engine, *, workflow_id: str
) -> tuple[str, set[str], set[str]]:
    start, nodes, adj, _rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )
    edge_ids = {str(edge.id) for edges in adj.values() for edge in edges}
    return str(start.id), {str(node_id) for node_id in nodes.keys()}, edge_ids


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


def _runtime_cancel_runner(req: RuntimeRunRequest) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected runtime workflow to be cancelled")


def _seed_step(
    conversation_engine,
    *,
    run_id: str,
    workflow_id: str,
    step_seq: int,
    op: str,
    state_update: list,
):
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
            "result_json": json.dumps(
                {
                    "conversation_node_id": None,
                    "state_update": state_update,
                    "next_step_names": [],
                    "status": "success",
                }
            ),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
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
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post(
            "/api/conversations", json={"user_id": "u-chat"}, headers=rw_headers
        )
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]

        submit = client.post(
            f"/api/conversations/{conversation_id}/turns:answer",
            json={"user_id": "u-chat", "text": "hello world"},
            headers=rw_headers,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

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
        turns.raise_for_status()
        transcript = turns.json()["turns"]
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

        reopened = client.get(
            f"/api/conversations/{conversation_id}", headers=ro_headers
        )
        reopened.raise_for_status()
        assert reopened.json()["turn_count"] == 2


def test_chat_rest_events_poll_returns_run_events(monkeypatch, engine_triplet):
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
        polled.raise_for_status()
        payload = polled.json()
        assert payload["run_id"] == run_id
        assert payload["events"]
        assert len(payload["events"]) <= 10
        assert payload["events"][0]["event_type"] == "run.created"


def test_chat_rest_sse_reconnects_cleanly_after_terminal_run(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _success_runner)
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


def test_chat_rest_debug_rag_workflow_returns_seeded_knowledge(monkeypatch, engine_triplet):
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
    engine, conversation_engine, workflow_engine = engine_triplet
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
        assert cancel.status_code == 202
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
        latest.raise_for_status()
        assert latest.json()["snapshot_node_id"] == snapshot_id

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
            state_update=[["u", {"counter": 1}]],
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
        assert [step["step_seq"] for step in steps.json()["steps"]] == [0, 1]

        checkpoints = client.get(
            f"/api/runs/{run_id}/checkpoints", headers=workflow_headers
        )
        checkpoints.raise_for_status()
        assert checkpoints.json()["checkpoints"][0]["state"]["counter"] == 1

        checkpoint = client.get(
            f"/api/runs/{run_id}/checkpoints/0", headers=workflow_headers
        )
        checkpoint.raise_for_status()
        assert checkpoint.json()["state"]["counter"] == 1

        replay = client.get(
            f"/api/runs/{run_id}/replay?target_step_seq=1", headers=workflow_headers
        )
        replay.raise_for_status()
        assert replay.json()["state"]["counter"] == 2

        viz = client.get("/api/viz/d3.json?graph_type=workflow")
        viz.raise_for_status()
        payload = viz.json()
        assert "nodes" in payload
        assert "links" in payload


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
            },
            headers=wf_rw,
        )
        assert submit.status_code == 202
        run_id = submit.json()["run_id"]

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
        assert [str(node.get("metadata", {}).get("wf_op")) for node in payload["nodes"]] == [
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

        forbidden = client.get("/designer/capabilities", headers=docs_ro)
        assert forbidden.status_code == 403

        resp = client.get("/designer/capabilities", headers=wf_ro)
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


def test_designer_capabilities_falls_back_to_default_resolver(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with TestClient(server.app) as client:
        wf_ro = _token_header(client, role="ro", ns="workflow")

        resp = client.get("/designer/capabilities", headers=wf_ro)
        resp.raise_for_status()
        payload = resp.json()

        runtime = payload["runtime"]
        assert runtime["resolver_found"] is True
        assert "start" in runtime["builtin_ops"]


def test_admin_delete_doc_uses_node_ids_for_node_refs(monkeypatch):
    doc_id = "doc-1"
    calls: list[tuple[str, object]] = []

    class _Read:
        def node_ids_by_doc(self, value):
            assert value == doc_id
            return ["node-1", "node-2"]

        def edge_ids_by_doc(self, value):
            assert value == doc_id
            return ["edge-1"]

    class _Backend:
        def edge_endpoints_delete(self, **kwargs):
            calls.append(("edge_endpoints_delete", kwargs))

        def node_docs_delete(self, **kwargs):
            calls.append(("node_docs_delete", kwargs))

        def edge_refs_delete(self, **kwargs):
            calls.append(("edge_refs_delete", kwargs))

        def edge_delete(self, **kwargs):
            calls.append(("edge_delete", kwargs))

        def node_refs_delete(self, **kwargs):
            calls.append(("node_refs_delete", kwargs))

        def node_delete(self, **kwargs):
            calls.append(("node_delete", kwargs))

        def document_delete(self, **kwargs):
            calls.append(("document_delete", kwargs))

    monkeypatch.setattr(
        server,
        "engine",
        _FixedResource(SimpleNamespace(read=_Read(), backend=_Backend())),
        raising=False,
    )
    monkeypatch.setattr(server, "require_role", lambda role: None, raising=False)

    payload = server.admin_delete_doc(doc_id)

    assert payload == {
        "ok": True,
        "doc_id": doc_id,
        "deleted": {"nodes": 2, "edges": 1},
    }
    assert ("node_refs_delete", {"ids": ["node-1", "node-2"]}) in calls
    assert ("edge_refs_delete", {"ids": ["edge-1"]}) in calls


def test_admin_delete_doc_raises_when_backend_delete_fails(monkeypatch):
    doc_id = "doc-2"

    class _Read:
        def node_ids_by_doc(self, value):
            assert value == doc_id
            return ["node-1"]

        def edge_ids_by_doc(self, value):
            assert value == doc_id
            return ["edge-1"]

    class _Backend:
        def edge_endpoints_delete(self, **kwargs):
            return None

        def node_docs_delete(self, **kwargs):
            return None

        def edge_refs_delete(self, **kwargs):
            raise RuntimeError("delete failed")

        def edge_delete(self, **kwargs):
            pytest.fail("edge_delete should not run after edge_refs_delete failure")

        def node_refs_delete(self, **kwargs):
            pytest.fail("node cleanup should not run after edge cleanup failure")

        def node_delete(self, **kwargs):
            pytest.fail("node cleanup should not run after edge cleanup failure")

        def document_delete(self, **kwargs):
            pytest.fail("document_delete should not run after edge cleanup failure")

    monkeypatch.setattr(
        server,
        "engine",
        _FixedResource(SimpleNamespace(read=_Read(), backend=_Backend())),
        raising=False,
    )
    monkeypatch.setattr(server, "require_role", lambda role: None, raising=False)

    with pytest.raises(HTTPException) as exc_info:
        server.admin_delete_doc(doc_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == (
        f"Failed to delete document {doc_id!r} during edge_refs_delete"
    )
