from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import graph_knowledge_engine.server_mcp_with_admin as server
from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent
from graph_knowledge_engine.conversation.models import ConversationNode, WorkflowCheckpointNode, WorkflowStepExecNode
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Grounding, Span
from graph_knowledge_engine.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from graph_knowledge_engine.server.run_registry import RunRegistry


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
            GraphKnowledgeEngine(persist_directory=str(root / "kg"), kg_graph_type="knowledge", embedding_function=ef),
            GraphKnowledgeEngine(persist_directory=str(root / "conversation"), kg_graph_type="conversation", embedding_function=ef),
            GraphKnowledgeEngine(persist_directory=str(root / "workflow"), kg_graph_type="workflow", embedding_function=ef),
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, answer_runner, runtime_runner=None):
    registry = RunRegistry(Path(workflow_engine.persist_directory) / "server_runs.sqlite")
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=answer_runner,
        runtime_runner=runtime_runner,
    )
    monkeypatch.setattr(server, "engine", _FixedResource(engine), raising=False)
    monkeypatch.setattr(server, "conversation_engine", _FixedResource(conversation_engine), raising=False)
    monkeypatch.setattr(server, "workflow_engine", _FixedResource(workflow_engine), raising=False)
    monkeypatch.setattr(server, "run_registry", _FixedResource(registry), raising=False)
    monkeypatch.setattr(server, "chat_service", _FixedResource(service), raising=False)
    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield
    monkeypatch.setattr(server.app.router, "lifespan_context", _noop_lifespan, raising=False)
    return service, registry


def _token_header(client: TestClient, *, role: str, ns: str) -> dict[str, str]:
    resp = client.post(
        "/auth/dev-token",
        json={"username": "tester", "role": role, "ns": ns},
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
    with client.stream("GET", path_template.format(run_id=run_id), headers=headers) as resp:
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


def _success_runner(req: AnswerRunRequest) -> dict:
    req.publish("run.stage", {"stage": "retrieve"})
    req.publish("reasoning.summary", {"stage": "retrieve", "summary": "Retrieving candidate evidence."})
    req.publish("run.stage", {"stage": "draft_answer"})
    req.publish("reasoning.summary", {"stage": "draft_answer", "summary": "Drafting the answer."})
    assistant_text = f"Assistant reply: {req.user_text}"
    assistant_turn_node_id = f"assistant|{uuid.uuid4().hex}"
    embedding = req.conversation_engine.iterative_defensive_emb(assistant_text)
    req.conversation_engine.add_node(
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
            metadata={"level_from_root": 0, "entity_type": "assistant_turn", "in_conversation_chain": True, "in_ui_chain": True},
            domain_id=None,
            canonical_entity_id=None,
            embedding=(embedding.tolist() if hasattr(embedding, "tolist") else embedding),
        )
    )
    req.prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(assistant_text)
    req.prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
    req.prev_turn_meta_summary.tail_turn_index += 1
    return {
        "assistant_turn_node_id": assistant_turn_node_id,
        "assistant_text": assistant_text,
        "workflow_status": "succeeded",
    }


def _cancel_runner(req: AnswerRunRequest) -> dict:
    req.publish("run.stage", {"stage": "retrieve"})
    req.publish("reasoning.summary", {"stage": "retrieve", "summary": "Retrieving candidate evidence."})
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected the run to be cancelled")


def _runtime_success_runner(req: RuntimeRunRequest) -> dict:
    req.publish("run.stage", {"stage": "execute"})
    req.publish("reasoning.summary", {"stage": "execute", "summary": "Executing workflow runtime."})
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


def _seed_step(conversation_engine, *, run_id: str, workflow_id: str, step_seq: int, op: str, state_update: list):
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
    conversation_engine.add_node(node)


def _seed_checkpoint(conversation_engine, *, run_id: str, workflow_id: str, step_seq: int, state: dict):
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
    conversation_engine.add_node(node)


def test_chat_rest_submit_and_sse(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _success_runner)
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post("/api/conversations", json={"user_id": "u-chat"}, headers=rw_headers)
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
        assert Path(workflow_engine.persist_directory, "server_runs.sqlite").exists()

        turns = client.get(f"/api/conversations/{conversation_id}/turns", headers=ro_headers)
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

        reopened = client.get(f"/api/conversations/{conversation_id}", headers=ro_headers)
        reopened.raise_for_status()
        assert reopened.json()["turn_count"] == 2


def test_chat_rest_cancelled_run_persists_no_assistant(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner)
    with TestClient(server.app) as client:
        rw_headers = _token_header(client, role="rw", ns="conversation")
        ro_headers = _token_header(client, role="ro", ns="conversation")

        created = client.post("/api/conversations", json={"user_id": "u-cancel"}, headers=rw_headers)
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
            where={"$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]},
            limit=10,
        )
        assert len(cancel_nodes) == 1

        final_run = _wait_for_status(client, run_id, ro_headers, {"cancelled"})
        assert final_run["assistant_turn_node_id"] in (None, "")

        turns = client.get(f"/api/conversations/{conversation_id}/turns", headers=ro_headers)
        turns.raise_for_status()
        transcript = turns.json()["turns"]
        assert [turn["role"] for turn in transcript] == ["user"]

        events = _collect_sse_events(client, run_id, ro_headers)
        names = [name for name, _payload in events]
        assert "run.cancelling" in names
        assert names[-1] == "run.cancelled"


def test_chat_debug_endpoints_namespace_and_workflow_viz(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service, _registry = _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _success_runner)
    with TestClient(server.app) as client:
        docs_headers = _token_header(client, role="rw", ns="docs")
        conv_headers = _token_header(client, role="rw", ns="conversation")
        workflow_headers = _token_header(client, role="ro", ns="workflow")

        forbidden = client.post("/api/conversations", json={"user_id": "u-docs"}, headers=docs_headers)
        assert forbidden.status_code == 403

        created = client.post("/api/conversations", json={"user_id": "u-snap"}, headers=conv_headers)
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
        _seed_checkpoint(conversation_engine, run_id=run_id, workflow_id=workflow_id, step_seq=0, state={"counter": 1})
        _seed_step(conversation_engine, run_id=run_id, workflow_id=workflow_id, step_seq=0, op="prepare", state_update=[["u", {"counter": 1}]])
        _seed_step(conversation_engine, run_id=run_id, workflow_id=workflow_id, step_seq=1, op="persist", state_update=[["u", {"counter": 2}]])

        steps = client.get(f"/api/runs/{run_id}/steps", headers=workflow_headers)
        steps.raise_for_status()
        assert [step["step_seq"] for step in steps.json()["steps"]] == [0, 1]

        checkpoints = client.get(f"/api/runs/{run_id}/checkpoints", headers=workflow_headers)
        checkpoints.raise_for_status()
        assert checkpoints.json()["checkpoints"][0]["state"]["counter"] == 1

        checkpoint = client.get(f"/api/runs/{run_id}/checkpoints/0", headers=workflow_headers)
        checkpoint.raise_for_status()
        assert checkpoint.json()["state"]["counter"] == 1

        replay = client.get(f"/api/runs/{run_id}/replay?target_step_seq=1", headers=workflow_headers)
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

        created = client.post("/api/conversations", json={"user_id": "u-runtime"}, headers=conv_rw)
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

        created = client.post("/api/conversations", json={"user_id": "u-runtime-cancel"}, headers=conv_rw)
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

        cancel_nodes = conversation_engine.get_nodes(
            where={"$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]},
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
