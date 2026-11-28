from __future__ import annotations

# Shared test imports are re-exported to semantic test modules below.
# ruff: noqa: F401

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
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Document, Edge, Grounding, Node, Span
from kogwistar.runtime.design import load_workflow_design
from kogwistar.runtime.models import (
    RunSuccess,
    RunSuspended,
    WorkflowCancelledNode,
    WorkflowCheckpointNode,
    WorkflowCompletedNode,
    WorkflowStepExecNode,
)
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import RunResult, WorkflowRuntime
from kogwistar.runtime.telemetry import EventEmitter, TraceContext
from kogwistar.runtime.telemetry_otel import OpenTelemetrySink
from kogwistar.server.chat_service import (
    AnswerRunRequest,
    ChatRunService,
    RunCancelledError,
    RuntimeRunRequest,
)
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.run_registry import RunRegistry, RunRegistryLaneMessageEventSink
from tests._helpers.engine_factories import FakeEmbeddingFunction
from tests._helpers.server_fixtures import build_engine_triplet
from tests._helpers.server_http_helpers import token_header

pytestmark = pytest.mark.ci_full




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


















def _visible_workflow_design_ids(
    workflow_engine, *, workflow_id: str
) -> tuple[str, set[str], set[str]]:
    start, nodes, adj, _rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )
    edge_ids = {str(edge.id) for edges in adj.values() for edge in edges}
    return str(start.id), {str(node_id) for node_id in nodes.keys()}, edge_ids




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

__all__ = [name for name in globals() if not name.startswith("__") and not name.startswith("test_")]
