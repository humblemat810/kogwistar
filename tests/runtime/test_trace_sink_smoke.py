from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.runtime.sinks import JsonlEventSink
from kogwistar.runtime.models import RunSuccess, WorkflowEdge, WorkflowNode
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import WorkflowRuntime
from tests._helpers.fake_backend import build_fake_backend
from tests.conftest import FakeEmbeddingFunction

pytestmark = [pytest.mark.core, pytest.mark.runtime]


def _grounding() -> Grounding:
    span = Span(
        collection_page_url="demo",
        document_page_url="demo",
        doc_id="demo",
        insertion_method="demo",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="system",
            is_verified=True,
            score=1.0,
            notes="demo",
        ),
    )
    return Grounding(spans=[span])


def _make_engine(tmp_path: Path, kind: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path / kind),
        kg_graph_type=kind,
        embedding_function=FakeEmbeddingFunction(dim=8),
        backend_factory=build_fake_backend,
    )


def _add_node(
    engine: GraphKnowledgeEngine,
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> None:
    engine.write.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": workflow_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v_smoke",
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _add_edge(
    engine: GraphKnowledgeEngine,
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
) -> None:
    engine.write.add_edge(
        WorkflowEdge(
            id=edge_id,
            source_ids=[src],
            target_ids=[dst],
            relation="wf_next",
            label="wf_next",
            type="relationship",
            summary="next",
            doc_id=edge_id,
            mentions=[_grounding()],
            properties={},
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": workflow_id,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_predicate": None,
                "wf_multiplicity": "one",
            },
            source_edge_ids=[],
            target_edge_ids=[],
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _trace_db_path(engine: GraphKnowledgeEngine) -> Path:
    return Path(engine.persist_directory) / "wf_trace.sqlite"


def _fetch_event_types(db_path: Path, *, run_id: str) -> list[str]:
    assert db_path.exists(), f"trace db missing at {db_path}"
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT type
            FROM wf_trace_events
            WHERE run_id = ?
            ORDER BY ts_ms ASC
            """,
            (run_id,),
        )
        return [str(row[0]) for row in cur.fetchall()]
    finally:
        con.close()


def test_trace_sink_smoke_hello_world_workflow(tmp_path: Path) -> None:
    workflow_engine = _make_engine(tmp_path, "workflow")
    conversation_engine = _make_engine(tmp_path, "conversation")

    workflow_id = "wf_trace_sink_hello_world"
    start_id = f"wf|{workflow_id}|hello"
    done_id = f"wf|{workflow_id}|done"

    _add_node(
        workflow_engine,
        workflow_id=workflow_id,
        node_id=start_id,
        op="hello",
        start=True,
    )
    _add_node(
        workflow_engine,
        workflow_id=workflow_id,
        node_id=done_id,
        op="done",
        terminal=True,
    )
    _add_edge(
        workflow_engine,
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|hello->done",
        src=start_id,
        dst=done_id,
    )

    resolver = MappingStepResolver()

    @resolver.register("hello")
    def _hello(ctx):
        return RunSuccess(state_update=[("u", {"hello": "world"})])

    @resolver.register("done")
    def _done(ctx):
        return RunSuccess(state_update=[("u", {"done": True})])

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
    )

    result = runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv_hello_world",
        turn_node_id="turn_hello_world",
        initial_state={"greeting": "hello"},
        run_id="run_hello_world",
    )
    assert result.status == "succeeded"
    assert result.final_state["hello"] == "world"

    if runtime.sink is not None:
        runtime.sink.close()

    event_types = _fetch_event_types(_trace_db_path(workflow_engine), run_id="run_hello_world")
    assert "workflow_run_started" in event_types
    assert "step_attempt_started" in event_types
    assert "step_attempt_completed" in event_types
    assert "workflow_run_completed" in event_types


def test_trace_sink_smoke_jsonl_mirror_hello_world_workflow(tmp_path: Path) -> None:
    workflow_engine = _make_engine(tmp_path, "workflow")
    conversation_engine = _make_engine(tmp_path, "conversation")

    workflow_id = "wf_trace_sink_jsonl_hello_world"
    start_id = f"wf|{workflow_id}|hello"
    done_id = f"wf|{workflow_id}|done"
    jsonl_path = tmp_path / "trace" / "runtime_events.jsonl"

    _add_node(
        workflow_engine,
        workflow_id=workflow_id,
        node_id=start_id,
        op="hello",
        start=True,
    )
    _add_node(
        workflow_engine,
        workflow_id=workflow_id,
        node_id=done_id,
        op="done",
        terminal=True,
    )
    _add_edge(
        workflow_engine,
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|hello->done",
        src=start_id,
        dst=done_id,
    )

    resolver = MappingStepResolver()

    @resolver.register("hello")
    def _hello(ctx):
        return RunSuccess(state_update=[("u", {"hello": "world"})])

    @resolver.register("done")
    def _done(ctx):
        return RunSuccess(state_update=[("u", {"done": True})])

    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        sink=JsonlEventSink(jsonl_path=jsonl_path),
    )

    result = runtime.run(
        workflow_id=workflow_id,
        conversation_id="conv_hello_world",
        turn_node_id="turn_hello_world",
        initial_state={"greeting": "hello"},
        run_id="run_hello_world_jsonl",
    )
    assert result.status == "succeeded"
    assert result.final_state["done"] is True

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert lines
    event_types = [json.loads(line)["type"] for line in lines]
    assert "workflow_run_started" in event_types
    assert "step_attempt_started" in event_types
    assert "step_attempt_completed" in event_types
    assert "workflow_run_completed" in event_types
