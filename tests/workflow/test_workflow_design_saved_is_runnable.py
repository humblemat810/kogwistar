import json
import tempfile

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


def _wf_node(*, workflow_id: str, node_id: str, op: str, start=False, terminal=False) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id,
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


def test_persisted_workflow_design_is_runnable(tmp_path):
    wf_dir = tmp_path / "wf"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_smoke_saved_runnable"

    # --------------------
    # Producer: create + save workflow design into workflow_engine
    # --------------------
    wf_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")

    n_a = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|a", op="a", start=True)
    n_b = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|b", op="b")
    n_end = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True)

    wf_engine.add_node(n_a)
    wf_engine.add_node(n_b)
    wf_engine.add_node(n_end)

    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|a->b",
        src=n_a.id,
        dst=n_b.id,
        predicate=None,
        priority=100,
        is_default=True,
    ))
    wf_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|b->end",
        src=n_b.id,
        dst=n_end.id,
        predicate=None,
        priority=100,
        is_default=True,
    ))

    # Ensure persisted
    # wf_engine.persist()

    # --------------------
    # Consumer: reopen engines and run persisted design using WorkflowRuntime
    # --------------------
    wf_engine2 = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    conv_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    # minimal predicates (none used)
    predicate_registry = {}
    

    # step resolver that records it ran
    def resolve_step(op: str):
        def _fn(ctx):
            # Keep JSONable state so checkpoints can serialize.
            with ctx.state_write as state:
                state.setdefault("ops", [])
                state["ops"].append(op)
                # {"op": op}
            return RunSuccess(conversation_node_id=None,state_update = [('u', {"op": op})], )
        return _fn

    rt = WorkflowRuntime(
        workflow_engine=wf_engine2,
        conversation_engine=conv_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    run_result = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv1",
        turn_node_id="turn1",
        initial_state={},
    )
    final_state, run_id = run_result.final_state, run_result.run_id
    assert run_id  # non-empty
    assert final_state["ops"] == ["a", "b", "end"]

    # Optional: confirm checkpoints exist and contain JSON
    ckpts = conv_engine.get_nodes(where={"entity_type": "workflow_checkpoint"}, limit=1000)
    assert len(ckpts) >= 1
    for ckpt in ckpts:
        state_json = ckpt.metadata.get("state_json")
        assert isinstance(state_json, str)
        json.loads(state_json)
