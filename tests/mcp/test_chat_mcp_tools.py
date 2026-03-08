from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
import uuid

import pytest

import graph_knowledge_engine.server_mcp_with_admin as server
from graph_knowledge_engine.conversation.models import ConversationNode, WorkflowCheckpointNode
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Grounding, Span
from graph_knowledge_engine.server.chat_service import AnswerRunRequest, ChatRunService, RunCancelledError
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
    root = Path(".tmp_chat_mcp_tests") / str(uuid.uuid4())
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


def _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, answer_runner):
    registry = RunRegistry(Path(workflow_engine.persist_directory) / "server_runs.sqlite")
    service = ChatRunService(
        get_knowledge_engine=lambda: engine,
        get_conversation_engine=lambda: conversation_engine,
        get_workflow_engine=lambda: workflow_engine,
        run_registry=registry,
        answer_runner=answer_runner,
    )
    monkeypatch.setattr(server, "engine", _FixedResource(engine), raising=False)
    monkeypatch.setattr(server, "conversation_engine", _FixedResource(conversation_engine), raising=False)
    monkeypatch.setattr(server, "workflow_engine", _FixedResource(workflow_engine), raising=False)
    monkeypatch.setattr(server, "run_registry", _FixedResource(registry), raising=False)
    monkeypatch.setattr(server, "chat_service", _FixedResource(service), raising=False)
    return service


@contextmanager
def _claims(role: str, ns: str):
    token = server.claims_ctx.set({"role": role, "ns": ns})
    try:
        yield
    finally:
        server.claims_ctx.reset(token)


def _structured(result):
    data = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if data:
        return data
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if isinstance(text, str) and text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {}


def _success_runner(req: AnswerRunRequest) -> dict:
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
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected the run to be cancelled")


@pytest.mark.asyncio
async def test_mcp_chat_tool_visibility_by_namespace(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _success_runner)

    with _claims("rw", "conversation"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "conversation.create" in names
        assert "conversation.ask" in names
        assert "conversation.cancel_run" in names
        assert "workflow.run_replay" not in names
        assert "doc_parse" not in names

    with _claims("ro", "workflow"):
        tools = await server.mcp.list_tools()
        names = {tool.name for tool in tools}
        assert "workflow.run_checkpoint_get" in names
        assert "workflow.run_replay" in names
        assert "conversation.ask" not in names


@pytest.mark.asyncio
async def test_mcp_chat_submit_cancel_and_workflow_diagnostics(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner)

    with _claims("rw", "conversation"):
        created = _structured(await server.mcp.call_tool("conversation.create", {"user_id": "u-mcp"}))
        conversation_id = created["conversation_id"]
        asked = _structured(
            await server.mcp.call_tool(
                "conversation.ask",
                {"conversation_id": conversation_id, "user_id": "u-mcp", "text": "cancel this"},
            )
        )
        run_id = asked["run_id"]

        cancelled = _structured(await server.mcp.call_tool("conversation.cancel_run", {"run_id": run_id}))
        assert cancelled["status"] == "cancelling"
        cancel_nodes = conversation_engine.get_nodes(
            where={"$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]},
            limit=10,
        )
        assert len(cancel_nodes) == 1

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(await server.mcp.call_tool("conversation.run_status", {"run_id": run_id}))
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Run did not reach cancelled status")

        transcript = _structured(
            await server.mcp.call_tool("conversation.get_transcript", {"conversation_id": conversation_id})
        )
        assert [turn["role"] for turn in transcript["turns"]] == ["user"]

    checkpoint = WorkflowCheckpointNode(
        id="wf_ckpt|diag-run|0",
        label="Checkpoint 0",
        type="entity",
        doc_id="wf_ckpt|diag-run|0",
        summary="workflow checkpoint",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={},
        metadata={
            "entity_type": "workflow_checkpoint",
            "run_id": "diag-run",
            "workflow_id": "wf.diag",
            "step_seq": 0,
            "state_json": json.dumps({"answer": "draft"}),
        },
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )
    conversation_engine.add_node(checkpoint)

    with _claims("ro", "workflow"):
        got = _structured(await server.mcp.call_tool("workflow.run_checkpoint_get", {"run_id": "diag-run", "step_seq": 0}))
        assert got["state"]["answer"] == "draft"

        # replay_to needs at least one checkpoint and no subsequent steps for target=0
        replay = _structured(await server.mcp.call_tool("workflow.run_replay", {"run_id": "diag-run", "target_step_seq": 0}))
        assert replay["state"]["answer"] == "draft"
