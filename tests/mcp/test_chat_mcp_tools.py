from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
import uuid

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")

pytestmark = pytest.mark.ci_full

import kogwistar.server_mcp_with_admin as server
from kogwistar.server import resources as server_resources
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.conversation.models import ConversationNode
from kogwistar.runtime.models import WorkflowCheckpointNode
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.fake_backend import build_fake_backend


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
            GraphKnowledgeEngine(
                persist_directory=str(root / "kg"),
                kg_graph_type="knowledge",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "conversation"),
                kg_graph_type="conversation",
                embedding_function=ef,
                backend_factory=build_fake_backend,
            ),
            GraphKnowledgeEngine(
                persist_directory=str(root / "workflow"),
                kg_graph_type="workflow",
                embedding_function=ef,
                backend_factory=build_fake_backend,
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
    # Patch both the live server module and the underlying resources module.
    # The refactor moved the real service wiring behind lazy resources, so tests
    # need to keep the two module views in sync.
    monkeypatch.setattr(
        server_resources, "engine", _FixedResource(engine), raising=False
    )
    monkeypatch.setattr(
        server_resources,
        "conversation_engine",
        _FixedResource(conversation_engine),
        raising=False,
    )
    monkeypatch.setattr(
        server_resources, "workflow_engine", _FixedResource(workflow_engine), raising=False
    )
    monkeypatch.setattr(
        server_resources, "run_registry", _FixedResource(registry), raising=False
    )
    monkeypatch.setattr(
        server_resources, "chat_service", _FixedResource(service), raising=False
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
    return service


@contextmanager
def _claims(role: str, ns: str, sub: str | None = None):
    claims = {"role": role, "ns": ns}
    if sub is not None:
        claims["sub"] = sub
    token = claims_ctx.set(claims)
    try:
        yield
    finally:
        claims_ctx.reset(token)


def _structured(result):
    data = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
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


def _cancel_runner(req: AnswerRunRequest) -> dict:
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
        "final_state": {"workflow_id": req.workflow_id},
    }


def _runtime_cancel_runner(req: RuntimeRunRequest) -> dict:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if req.is_cancel_requested():
            raise RunCancelledError()
        time.sleep(0.02)
    raise AssertionError("Expected runtime workflow to be cancelled")


@pytest.mark.asyncio
async def test_mcp_chat_tool_visibility_by_namespace(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

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
        assert "workflow.run_status" in names
        assert "workflow.run_events" in names
        assert "workflow.run_submit" not in names
        assert "workflow.design_history" in names
        assert "workflow.design_undo" not in names
        assert "workflow.run_checkpoint_get" in names
        assert "workflow.run_replay" in names
        assert "workflow.process_table" in names
        assert "workflow.operator_inbox" in names
        assert "workflow.blocked_runs" in names
        assert "workflow.process_timeline" in names
        assert "conversation.ask" not in names


@pytest.mark.asyncio
async def test_mcp_workflow_process_views(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    service.run_registry.create_run(
        run_id="run-process-1",
        conversation_id="conv-1",
        workflow_id="wf-1",
        user_id="user-1",
        user_turn_node_id="turn-1",
        status="running",
    )
    service.run_registry.append_event(
        "run-process-1", "run.stage", {"stage": "execute"}
    )
    service.run_registry.update_status(
        "run-process-1",
        status="suspended",
        started=True,
    )

    with _claims("ro", "workflow"):
        processes = _structured(
            await server.mcp.call_tool("workflow.process_table", {"limit": 10})
        )["processes"]
        assert processes
        assert processes[0]["process_id"] == "run-process-1"
        assert processes[0]["process_kind"] == "workflow_run"

        blocked = _structured(
            await server.mcp.call_tool("workflow.blocked_runs", {"limit": 10})
        )["processes"]
        assert any(item["process_id"] == "run-process-1" for item in blocked)

        timeline = _structured(
            await server.mcp.call_tool(
                "workflow.process_timeline", {"run_id": "run-process-1", "limit": 10}
            )
        )
        assert timeline["run_id"] == "run-process-1"
        assert timeline["events"]


@pytest.mark.asyncio
async def test_mcp_workflow_service_tools_round_trip(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _runtime_success_runner,
        runtime_runner=_runtime_success_runner,
    )

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
            "project_view",
            "service.manage",
            "service.inspect",
            "service.heartbeat",
        ]
        created = service.create_conversation(user_id="svc-mcp-user")

    with _claims("rw", "workflow", sub="svc-mcp"):
        claims = claims_ctx.get() or {}
        claims["capabilities"] = [
            "service.manage",
            "service.inspect",
            "service.heartbeat",
            "project_view",
            "spawn_process",
            "workflow.run.read",
            "workflow.run.write",
        ]
        declared = _structured(
            await server.mcp.call_tool(
                "workflow.service_declare",
                {
                    "service_id": "svc.mcp.demo",
                    "service_kind": "daemon",
                    "target_kind": "workflow",
                    "target_ref": "wf.service.mcp",
                    "target_config": {"conversation_id": created["conversation_id"]},
                },
            )
        )
        assert declared["service_id"] == "svc.mcp.demo"

        heartbeat = _structured(
            await server.mcp.call_tool(
                "workflow.service_heartbeat",
                {
                    "service_id": "svc.mcp.demo",
                    "instance_id": "mcp-1",
                    "payload": {"beat": 1},
                },
            )
        )
        assert heartbeat["health_status"] == "healthy"

        listed = _structured(
            await server.mcp.call_tool("workflow.service_list", {"limit": 20})
        )["services"]
        assert any(item["service_id"] == "svc.mcp.demo" for item in listed)

        triggered = _structured(
            await server.mcp.call_tool(
                "workflow.service_trigger",
                {"service_id": "svc.mcp.demo", "trigger_type": "external event"},
            )
        )
        assert triggered["current_child_run_id"]

        events = _structured(
            await server.mcp.call_tool(
                "workflow.service_events", {"service_id": "svc.mcp.demo", "limit": 50}
            )
        )["events"]
        assert any(evt["event_type"] == "service.triggered" for evt in events)


def test_chat_run_service_execution_meta_comes_from_conversation_engine(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    assert service._execution_meta_store() is conversation_engine.meta_sqlite
    assert service.run_registry.meta_store is conversation_engine.meta_sqlite


def test_chat_run_service_process_table_uses_claim_scope_not_backend_namespace(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.namespace = "backend-tenant"
    service.run_registry.create_run(
        run_id="run-scope-1",
        conversation_id="conv-scope-1",
        workflow_id="wf-scope-1",
        user_id="user-scope-1",
        user_turn_node_id="turn-scope-1",
        status="running",
    )
    token = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "store-tenant",
            "execution_ns": "exec-tenant",
            "security_scope": "tenant-a",
        }
    )
    try:
        rows = service.list_process_table(limit=10)
    finally:
        claims_ctx.reset(token)
    assert rows
    row = rows[0]
    assert row["namespace"] == "exec-tenant"
    assert row["storage_namespace"] == "store-tenant"
    assert row["security_scope"] == "tenant-a"


def test_chat_run_service_resume_contract_exposes_checkpoint_metadata(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_engine.write.add_node(
        WorkflowCheckpointNode(
            id="wf_ckpt|run-resume-1|0",
            label="Checkpoint 0",
            type="entity",
            doc_id="wf_ckpt|run-resume-1|0",
            summary="checkpoint",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={},
            metadata={
                "entity_type": "workflow_checkpoint",
                "run_id": "run-resume-1",
                "workflow_id": "wf-resume-1",
                "step_seq": 0,
                "checkpoint_schema_version": 1,
                "state_json": "{\"foo\": \"bar\"}",
            },
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
    )
    contract = service.resume_contract("run-resume-1")
    assert contract["run_id"] == "run-resume-1"
    assert contract["latest_checkpoint_step_seq"] == 0
    assert contract["checkpoint_schema_version"] == 1
    assert "checkpoint_schema_version" in contract["persisted_keys"]
    assert "_rt_join" in contract["ephemeral_keys"]


def test_chat_run_service_operator_inbox_respects_security_scope(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    token_send = claims_ctx.set(
        {
            "ns": "conversation",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-a",
        }
    )
    try:
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.private",
            payload={"hello": "world"},
            security_scope="tenant-a",
        )
        conversation_engine.send_lane_message(
            conversation_id="conv-scope-msg",
            inbox_id="inbox:ops",
            sender_id="lane:a",
            recipient_id="lane:b",
            msg_type="request.shared",
            payload={"hello": "shared"},
            security_scope="tenant-a",
            shared_scope=True,
        )
    finally:
        claims_ctx.reset(token_send)

    token_read = claims_ctx.set(
        {
            "ns": "workflow",
            "storage_ns": "conversation",
            "execution_ns": "conversation",
            "security_scope": "tenant-b",
        }
    )
    try:
        rows = service.list_operator_inbox(inbox_id="inbox:ops", limit=10)
    finally:
        claims_ctx.reset(token_read)
    assert len(rows) == 1
    assert rows[0]["msg_type"] == "request.shared"
    assert rows[0]["visibility"] == "shared"


@pytest.mark.asyncio
async def test_mcp_chat_submit_cancel_and_workflow_diagnostics(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )

    with _claims("rw", "conversation"):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-mcp"})
        )
        conversation_id = created["conversation_id"]
        asked = _structured(
            await server.mcp.call_tool(
                "conversation.ask",
                {
                    "conversation_id": conversation_id,
                    "user_id": "u-mcp",
                    "text": "cancel this",
                },
            )
        )
        run_id = asked["run_id"]

        cancelled = _structured(
            await server.mcp.call_tool("conversation.cancel_run", {"run_id": run_id})
        )
        assert cancelled["status"] == "cancelling"
        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await server.mcp.call_tool(
                    "conversation.run_status", {"run_id": run_id}
                )
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Run did not reach cancelled status")

        transcript = _structured(
            await server.mcp.call_tool(
                "conversation.get_transcript", {"conversation_id": conversation_id}
            )
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
    conversation_engine.write.add_node(checkpoint)

    with _claims("ro", "workflow"):
        got = _structured(
            await server.mcp.call_tool(
                "workflow.run_checkpoint_get", {"run_id": "diag-run", "step_seq": 0}
            )
        )
        assert got["state"]["answer"] == "draft"

        # replay_to needs at least one checkpoint and no subsequent steps for target=0
        replay = _structured(
            await server.mcp.call_tool(
                "workflow.run_replay", {"run_id": "diag-run", "target_step_seq": 0}
            )
        )
        assert replay["state"]["answer"] == "draft"


@pytest.mark.asyncio
async def test_mcp_workflow_runtime_submit_cancel(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch,
        engine,
        conversation_engine,
        workflow_engine,
        _success_runner,
        runtime_runner=_runtime_cancel_runner,
    )

    with _claims("rw", "conversation"):
        created = _structured(
            await server.mcp.call_tool("conversation.create", {"user_id": "u-wf-mcp"})
        )
        conversation_id = created["conversation_id"]

    with _claims("rw", "workflow"):
        submitted = _structured(
            await server.mcp.call_tool(
                "workflow.run_submit",
                {
                    "workflow_id": "wf.runtime.mcp.cancel",
                    "conversation_id": conversation_id,
                    "initial_state": {"counter": 1},
                },
            )
        )
        run_id = submitted["run_id"]
        cancelling = _structured(
            await server.mcp.call_tool("workflow.run_cancel", {"run_id": run_id})
        )
        assert cancelling["status"] == "cancelling"

        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = _structured(
                await server.mcp.call_tool("workflow.run_status", {"run_id": run_id})
            )
            if status["status"] == "cancelled":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Workflow run did not reach cancelled status")

        events = _structured(
            await server.mcp.call_tool("workflow.run_events", {"run_id": run_id})
        )
        names = [str(evt.get("event_type") or "") for evt in events.get("events", [])]
        assert "run.cancelling" in names
        assert "run.cancelled" in names

        cancel_nodes = conversation_engine.get_nodes(
            where={
                "$and": [{"entity_type": "workflow_cancel_request"}, {"run_id": run_id}]
            },
            limit=10,
        )
        assert len(cancel_nodes) == 1


@pytest.mark.asyncio
async def test_mcp_workflow_design_undo_redo(monkeypatch, engine_triplet):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    workflow_id = "wf.design.mcp.undo_redo"
    designer_id = "designer-mcp"
    with _claims("rw", "workflow", sub=designer_id):
        n1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|start",
                    "label": "Start",
                    "op": "start",
                    "start": True,
                },
            )
        )
        assert n1["node_id"] == f"wf|{workflow_id}|start"
        n2 = _structured(
            await server.mcp.call_tool(
                "workflow.design_node_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "node_id": f"wf|{workflow_id}|end",
                    "label": "End",
                    "op": "end",
                    "terminal": True,
                },
            )
        )
        assert n2["node_id"] == f"wf|{workflow_id}|end"
        e1 = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_upsert",
                {
                    "workflow_id": workflow_id,
                    "designer_id": designer_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "src": f"wf|{workflow_id}|start",
                    "dst": f"wf|{workflow_id}|end",
                    "relation": "wf_next",
                    "is_default": True,
                },
            )
        )
        assert e1["edge_id"] == f"wf|{workflow_id}|e|start_end"

        hist_before = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        assert int(hist_before["current_version"]) >= 3
        assert bool(hist_before["can_undo"]) is True

        undone = _structured(
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert undone["status"] in {"ok", "noop"}

        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )

        redone = _structured(
            await server.mcp.call_tool(
                "workflow.design_redo",
                {"workflow_id": workflow_id, "designer_id": designer_id},
            )
        )
        assert redone["status"] in {"ok", "noop"}

        deleted_after_redo = _structured(
            await server.mcp.call_tool(
                "workflow.design_edge_delete",
                {
                    "workflow_id": workflow_id,
                    "edge_id": f"wf|{workflow_id}|e|start_end",
                    "designer_id": designer_id,
                },
            )
        )
        assert bool(deleted_after_redo.get("deleted")) is True
        hist_after = _structured(
            await server.mcp.call_tool(
                "workflow.design_history", {"workflow_id": workflow_id}
            )
        )
        timeline_ops = [
            str(item.get("op") or "") for item in hist_after.get("timeline", [])
        ]
        assert "UNDO_APPLIED" in timeline_ops
        assert "REDO_APPLIED" in timeline_ops


@pytest.mark.asyncio
async def test_mcp_workflow_design_requires_designer_id_and_enforces_subject(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )

    with _claims("rw", "workflow", sub="alice"):
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.missing_designer"},
            )
        with pytest.raises(Exception):
            await server.mcp.call_tool(
                "workflow.design_undo",
                {"workflow_id": "wf.design.mcp.mismatch", "designer_id": "bob"},
            )
