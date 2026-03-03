
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    WorkflowNode,
    WorkflowEdge,
    Grounding,
    Span,
    MentionVerification,
)

from graph_knowledge_engine.workflow.runtime import WorkflowRuntime
from graph_knowledge_engine.models import RunSuccess


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(*, workflow_id: str, node_id: str, op: str, start=False, terminal=False, extra_metadata: dict | None = None) -> WorkflowNode:
    md = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": op,
        "wf_start": start,
        "wf_terminal": terminal,
        "wf_version": "v1",
    }
    if extra_metadata:
        md.update(extra_metadata)
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata=md,
        domain_id=None,
        canonical_entity_id=None,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int = 100,
    is_default: bool = False,
    multiplicity: str = "one",
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": multiplicity,
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


def _trace_db_path(conversation_engine: GraphKnowledgeEngine) -> Path:
    return Path(conversation_engine.persist_directory) / "wf_trace.sqlite"


def _fetch_events(db_path: Path, *, run_id: str) -> List[Dict[str, Any]]:
    assert db_path.exists(), f"Trace DB not found at {db_path}"
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT ts_ms, type, run_id, token_id, step_seq, node_id, span_id, parent_span_id, payload_json
            FROM wf_trace_events
            WHERE run_id = ?
            ORDER BY ts_ms ASC
            """,
            (run_id,),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for ts_ms, typ, rid, token_id, step_seq, node_id, span_id, parent_span_id, payload_json in rows:
            out.append(
                {
                    "ts_ms": int(ts_ms),
                    "type": str(typ),
                    "run_id": str(rid),
                    "token_id": str(token_id),
                    "step_seq": int(step_seq),
                    "node_id": str(node_id),
                    "span_id": str(span_id),
                    "parent_span_id": str(parent_span_id) if parent_span_id is not None else None,
                    "payload": json.loads(payload_json) if payload_json else {},
                }
            )
        return out
    finally:
        con.close()


def _events_of(events: List[Dict[str, Any]], typ: str) -> List[Dict[str, Any]]:
    return [e for e in events if e["type"] == typ]


def _group_by_step(events: List[Dict[str, Any]], typ: str) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    # key: (step_seq, node_id)
    out: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for e in events:
        if e["type"] != typ:
            continue
        k = (int(e["step_seq"]), str(e["node_id"]))
        out.setdefault(k, []).append(e)
    return out


def _assert_started_completed_pairing(events: List[Dict[str, Any]]) -> None:
    started = _group_by_step(events, "step_attempt_started")
    completed = _group_by_step(events, "step_attempt_completed")

    assert started, "No step_attempt_started events"
    assert completed, "No step_attempt_completed events"

    # Every started key must have a completed key
    for k, evs in started.items():
        assert k in completed, f"Missing completed for step {k}"
        assert len(evs) == 1, f"Expected exactly 1 started event for {k}, got {len(evs)}"
        assert len(completed[k]) == 1, f"Expected exactly 1 completed event for {k}, got {len(completed[k])}"
        assert started[k][0]["ts_ms"] <= completed[k][0]["ts_ms"], f"Completed before started for {k}"

    # And no completed without started
    for k in completed.keys():
        assert k in started, f"Completed without started for step {k}"


def test_tracing_routing_decision_end_to_end(tmp_path: Path):
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_trace_routing_gate"
    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    # gate(start) -> b if has_done_a else -> a(default); a -> end
    n_gate = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|gate", op="gate", start=True)
    n_a = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a")
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_end = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True)

    for n in [n_gate, n_a, n_b, n_end]:
        workflow_engine.add_node(n)

    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|gate->b",
        src=n_gate.id,
        dst=n_b.id,
        predicate="has_done_a",
        priority=0,
        is_default=False,
    ))
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|gate->a|default",
        src=n_gate.id,
        dst=n_a.id,
        predicate=None,
        priority=100,
        is_default=True,
    ))
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|a->end",
        src=n_a.id,
        dst=n_end.id,
        predicate=None,
        priority=100,
        is_default=True,
    ))

    predicate_registry = {
        "has_done_a": lambda wf_edge_info, st, last: ("result.a" in st),
    }

    def resolve_step(op: str):
        def _fn(ctx):
            if op == "a":
                return RunSuccess(conversation_node_id=None, state_update=[("u", {"result.a": {"value": "v_a"}})])
            if op == "end":
                return RunSuccess(conversation_node_id=None, state_update=[("u", {"result.end": {"value": "v_end"}})])
            return RunSuccess(conversation_node_id=None, state_update=[])
        return _fn

    rt = WorkflowRuntime(
        workflow_engine=GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow"),
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=9999,
        max_workers=1,
    )

    run_result = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv_trace_1",
        turn_node_id="turn_trace_1",
        initial_state={},  # predicate false at first gate
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    # Flush async SQLite writer
    rt.sink.close()

    events = _fetch_events(_trace_db_path(workflow_engine), run_id=run_id)
    types = [e["type"] for e in events]

    # Lifecycle
    assert "workflow_run_started" in types
    assert "workflow_run_completed" in types

    # Strict step pairing: every started has exactly one completed and vice versa
    _assert_started_completed_pairing(events)

    # Routing trace: should exist at least for gate and a (both have outgoing edges)
    routing = _events_of(events, "routing_decision")
    assert routing, "Expected routing_decision events"

    routing_keys = {(e["step_seq"], e["node_id"]) for e in routing}
    assert any(k[1].endswith("|gate") for k in routing_keys), routing_keys
    assert any(k[1].endswith("|a") for k in routing_keys), routing_keys

    # Gate routing decision should include a false predicate eval and select 'a'
    gate_rd = [e for e in routing if e["node_id"].endswith("|gate")]
    assert gate_rd
    payload = gate_rd[0]["payload"]
    assert "evaluated" in payload and "selected" in payload and "next_nodes" in payload
    assert any((isinstance(x, list) and len(x) >= 2 and x[1] is False) for x in payload["evaluated"]), payload
    assert payload["next_nodes"] == [n_a.id]

    # Standard per-item events exist for telemetry compatibility
    assert "predicate_evaluated" in types
    assert "edge_selected" in types

    # Sanity final state
    assert final_state["result.a"]["value"] == "v_a"
    assert final_state["result.end"]["value"] == "v_end"


def test_tracing_join_events_end_to_end(tmp_path: Path):
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_trace_join"
    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    # start -> fork(fanout) -> (a,b) -> join -> end
    n_start = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|start", op="start", start=True)
    n_fork = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|fork", op="fork", extra_metadata={"wf_fanout": True})
    n_a = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a")
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_join = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|join", op="join", extra_metadata={"wf_join": True, "wf_join_is_merge": True})
    n_end = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True)

    for n in [n_start, n_fork, n_a, n_b, n_join, n_end]:
        workflow_engine.add_node(n)

    workflow_engine.add_edge(_wf_edge(workflow_id=workflow_id, edge_id=f"wf|{workflow_id}|e|start->fork", src=n_start.id, dst=n_fork.id, predicate=None, priority=100, is_default=True))
    # fork fanout edges: predicate edges that always match so fanout returns BOTH
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|fork->a",
        src=n_fork.id,
        dst=n_a.id,
        predicate="fork_all",     # <-- force match
        priority=10,
        is_default=False,
    ))
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|fork->b",
        src=n_fork.id,
        dst=n_b.id,
        predicate="fork_all",     # <-- force match
        priority=20,
        is_default=False,
    ))

    predicate_registry = {
        "fork_all": lambda wf_edge_info, st, last: True
    }
    workflow_engine.add_edge(_wf_edge(workflow_id=workflow_id, edge_id=f"wf|{workflow_id}|e|a->join", src=n_a.id, dst=n_join.id, predicate=None, priority=100, is_default=True))
    workflow_engine.add_edge(_wf_edge(workflow_id=workflow_id, edge_id=f"wf|{workflow_id}|e|b->join", src=n_b.id, dst=n_join.id, predicate=None, priority=100, is_default=True))
    workflow_engine.add_edge(_wf_edge(workflow_id=workflow_id, edge_id=f"wf|{workflow_id}|e|join->end", src=n_join.id, dst=n_end.id, predicate=None, priority=100, is_default=True))


    def resolve_step(op: str):
        def _fn(ctx):
            return RunSuccess(conversation_node_id=None, state_update=[("u", {f"result.{op}": {"value": f"v_{op}"}})])
        return _fn

    rt = WorkflowRuntime(
        workflow_engine=GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow"),
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=9999,
        max_workers=2,
    )

    run_result = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv_trace_2",
        turn_node_id="turn_trace_2",
        initial_state={},
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    rt.sink.close()

    events = _fetch_events(_trace_db_path(workflow_engine), run_id=run_id)
    types = [e["type"] for e in events]

    assert "join_arrived" in types, f"join events missing; saw {sorted(set(types))}"
    assert "join_waiting" in types
    assert "join_released" in types

    join_events = [e for e in events if e["type"].startswith("join_")]
    assert any(e["payload"].get("join_node_id") == n_join.id for e in join_events), join_events[:5]

    # Sanity final state
    assert final_state["result.start"]["value"] == "v_start"
    assert final_state["result.end"]["value"] == "v_end"
