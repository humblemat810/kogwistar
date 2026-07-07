from __future__ import annotations

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowDesignArtifact, WorkflowEdge, WorkflowNode

DREAM_MAINTENANCE_WORKFLOW_ID = "dream.maintenance.v1"


def _span(workflow_id: str) -> Span:
    return Span.from_dummy_for_workflow(workflow_id)


def _node(
    *,
    workflow_id: str,
    node_id: str,
    label: str,
    op: str,
    start: bool = False,
    terminal: bool = False,
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=label,
        type="entity",
        doc_id=node_id,
        summary=label,
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": bool(start),
            "wf_terminal": bool(terminal),
            "wf_version": "v1",
        },
        mentions=[Grounding(spans=[_span(workflow_id)])],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _edge(*, workflow_id: str, edge_id: str, src: str, dst: str) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="wf_next",
        doc_id=edge_id,
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": None,
            "wf_priority": 100,
            "wf_is_default": True,
            "wf_multiplicity": "one",
            "wf_edge_kind": "wf_next",
            "wf_version": "v1",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[Grounding(spans=[_span(workflow_id)])],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def dream_workflow_expected_ops() -> tuple[str, ...]:
    return (
        "dream_start",
        "dream_select_signals",
        "dream_build_proposals",
        "dream_persist_reasoning",
        "dream_persist_proposals",
        "dream_evaluate_proposals",
        "dream_end",
    )


def build_dream_maintenance_workflow_design(
    *, workflow_id: str = DREAM_MAINTENANCE_WORKFLOW_ID
) -> WorkflowDesignArtifact:
    wid = lambda suffix: f"wf:{workflow_id}:{suffix}"
    ops = dream_workflow_expected_ops()
    nodes = [
        _node(workflow_id=workflow_id, node_id=wid("start"), label="Dream Start", op=ops[0], start=True),
        _node(workflow_id=workflow_id, node_id=wid("select"), label="Select Signals", op=ops[1]),
        _node(workflow_id=workflow_id, node_id=wid("build"), label="Build Proposals", op=ops[2]),
        _node(workflow_id=workflow_id, node_id=wid("reason"), label="Persist Reasoning", op=ops[3]),
        _node(workflow_id=workflow_id, node_id=wid("proposal"), label="Persist Proposals", op=ops[4]),
        _node(workflow_id=workflow_id, node_id=wid("evaluate"), label="Evaluate Proposals", op=ops[5]),
        _node(workflow_id=workflow_id, node_id=wid("end"), label="Dream End", op=ops[6], terminal=True),
    ]
    edges = [
        _edge(workflow_id=workflow_id, edge_id=wid("e:start->select"), src=wid("start"), dst=wid("select")),
        _edge(workflow_id=workflow_id, edge_id=wid("e:select->build"), src=wid("select"), dst=wid("build")),
        _edge(workflow_id=workflow_id, edge_id=wid("e:build->reason"), src=wid("build"), dst=wid("reason")),
        _edge(workflow_id=workflow_id, edge_id=wid("e:reason->proposal"), src=wid("reason"), dst=wid("proposal")),
        _edge(workflow_id=workflow_id, edge_id=wid("e:proposal->evaluate"), src=wid("proposal"), dst=wid("evaluate")),
        _edge(workflow_id=workflow_id, edge_id=wid("e:evaluate->end"), src=wid("evaluate"), dst=wid("end")),
    ]
    return WorkflowDesignArtifact(
        workflow_id=workflow_id,
        workflow_version="v1",
        start_node_id=wid("start"),
        nodes=nodes,
        edges=edges,
        source_workflow_id=None,
        source_step_id=None,
    )


def materialize_dream_workflow_design(workflow_engine, design: WorkflowDesignArtifact) -> None:
    for node in design.nodes:
        workflow_engine.write.add_node(node)
    for edge in design.edges:
        workflow_engine.write.add_edge(edge)
