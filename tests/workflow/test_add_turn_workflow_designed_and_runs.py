import json
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    WorkflowNode,
    WorkflowEdge,
    Grounding,
    Span,
    MentionVerification,
    WorkflowRunNode,
    WorkflowStepExecNode,
    WorkflowCheckpointNode,
)
from graph_knowledge_engine.workflow.runtime import WorkflowRuntime


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


def _wf_node(*, workflow_id: str, node_id: str, op: str, start=False, terminal=False, fanout=False) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_fanout": fanout,
            "wf_version": "v1",
        },
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
    priority: int,
    is_default: bool,
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


def test_add_turn_workflow_design_and_run(tmp_path):
    # -----------------------
    # Producer-side: create design
    # -----------------------
    wf_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "wf"), kg_graph_type="workflow")
    conv_engine = GraphKnowledgeEngine(persist_directory=str(tmp_path / "conv"), kg_graph_type="conversation")

    workflow_id = "wf_add_turn_legacy_shape"

    # Nodes (legacy-ish pipeline)
    n_start = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|start", op="memory_retrieve", start=True)
    n_kg    = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|kg", op="kg_retrieve")
    n_mpin  = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|mpin", op="memory_pin")
    n_kpin  = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|kpin", op="kg_pin")
    n_ans   = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|ans", op="answer")
    n_dec   = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|dec", op="decide_summarize")
    n_sum   = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|sum", op="summarize")
    n_end   = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True)

    for n in [n_start, n_kg, n_mpin, n_kpin, n_ans, n_dec, n_sum, n_end]:
        wf_engine.add_node(n)

    # Edges:
    # memory_retrieve -> kg_retrieve
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|start->kg",
        src=n_start.id, dst=n_kg.id,
        predicate=None, priority=100, is_default=True
    ))

    # kg_retrieve -> memory_pin (if should_pin_memory)
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->mpin",
        src=n_kg.id, dst=n_mpin.id,
        predicate="should_pin_memory", priority=0, is_default=False
    ))
    # kg_retrieve -> kg_pin (if should_pin_kg)
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->kpin",
        src=n_kg.id, dst=n_kpin.id,
        predicate="should_pin_kg", priority=1, is_default=False
    ))
    # If neither pin branch matches, go to answer by default
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->ans|default",
        src=n_kg.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))

    # pin steps both route to answer
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|mpin->ans",
        src=n_mpin.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kpin->ans",
        src=n_kpin.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))

    # answer -> decide_summarize
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|ans->dec",
        src=n_ans.id, dst=n_dec.id,
        predicate=None, priority=100, is_default=True
    ))
    # decide_summarize -> summarize (if should_summarize)
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|dec->sum",
        src=n_dec.id, dst=n_sum.id,
        predicate="should_summarize", priority=0, is_default=False
    ))
    # decide_summarize -> end (default)
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|dec->end|default",
        src=n_dec.id, dst=n_end.id,
        predicate=None, priority=100, is_default=True
    ))
    # summarize -> end
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|sum->end",
        src=n_sum.id, dst=n_end.id,
        predicate=None, priority=100, is_default=True
    ))

    # -----------------------
    # Consumer-side: run it
    # -----------------------
    predicate_registry = {
        "should_pin_memory": lambda st, r: bool(st.get("memory", {}).get("selected_ids")),
        "should_pin_kg": lambda st, r: bool(st.get("kg", {}).get("selected_ids")),
        "should_summarize": lambda st, r: bool(st.get("decide", {}).get("need_summary")),
    }

    def resolve_step(op: str):
        def _fn(ctx):
            # keep everything JSONable
            with ctx.state_write as state:
                if op == "memory_retrieve":
                    state["memory"] = {"selected_ids": ["m1"], "text": "memory context"}
                    return {"ok": True}
                if op == "kg_retrieve":
                    state["kg"] = {"selected_ids": ["k1"], "facts": ["f1"]}
                    return {"ok": True}
                if op == "memory_pin":
                    state["memory_pin"] = {"pinned": ["m1"]}
                    return {"ok": True}
                if op == "kg_pin":
                    state["kg_pin"] = {"pinned": ["k1"]}
                    return {"ok": True}
                if op == "answer":
                    state["answer"] = {"text": "hello", "llm_decision_need_summary": True}
                    return ctx.state_view["answer"]
                if op == "decide_summarize":
                    need = bool(ctx.state_view.get("answer", {}).get("llm_decision_need_summary"))
                    state["decide"] = {"need_summary": need}
                    return ctx.state_view["decide"]
                if op == "summarize":
                    state["summary"] = {"text": "summary"}
                    return ctx.state_view["summary"]
                if op == "end":
                    return {"done": True}
            raise KeyError(op)
        return _fn

    rt = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=2,
    )

    run_result = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv1",
        turn_node_id="turn1",
        initial_state={"memory": None, "kg": None},
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    # Basic assertions: the workflow executed and checkpointed
    assert run_id.startswith("run|")
    assert "answer" in final_state
    assert final_state["summary"]["text"] == "summary"

    # Verify persisted trace/checkpoint nodes exist in conversation engine
    run_nodes = conv_engine.get_nodes(where={"entity_type": "workflow_run"}, limit=1000)
    step_nodes = conv_engine.get_nodes(where={"entity_type": "workflow_step_exec"}, limit=10000)
    ckpt_nodes = conv_engine.get_nodes(where={"entity_type": "workflow_checkpoint"}, limit=10000)

    assert any(isinstance(n, WorkflowRunNode) for n in run_nodes)
    assert any(isinstance(n, WorkflowStepExecNode) for n in step_nodes)
    assert len(ckpt_nodes) >= 1

    # Verify state_json is valid JSON in checkpoints (resume-friendly)
    for n in ckpt_nodes:
        blob = n.metadata.get("state_json")
        assert isinstance(blob, str)
        json.loads(blob)  # must parse
