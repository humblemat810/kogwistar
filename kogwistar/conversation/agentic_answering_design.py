from __future__ import annotations

import copy

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowDesignArtifact, WorkflowEdge, WorkflowNode

AGENTIC_ANSWERING_WORKFLOW_ID = "agentic_answering.v2"


def _span(workflow_id: str) -> Span:
    return Span.from_dummy_for_workflow(workflow_id)


def _node(
    *,
    workflow_id: str,
    node_id: str,
    label: str,
    op: str | None,
    start: bool = False,
    terminal: bool = False,
    fanout: bool = False,
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
            "wf_fanout": bool(fanout),
            "wf_version": "v2",
        },
        mentions=[Grounding(spans=[_span(workflow_id)])],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    relation: str,
    predicate: str | None,
    priority: int = 100,
    is_default: bool = False,
    multiplicity: str = "one",
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation=relation,
        label=relation,
        type="relationship",
        summary=relation,
        doc_id=edge_id,
        mentions=[Grounding(spans=[_span(workflow_id)])],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": predicate,
            "wf_priority": priority,
            "wf_is_default": bool(is_default),
            "wf_multiplicity": multiplicity,
            "wf_edge_kind": "wf_next",
            "wf_version": "v2",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def agentic_answering_expected_ops() -> tuple[str, ...]:
    return (
        "start",
        "aa_prepare",
        "aa_get_view_and_question",
        "aa_retrieve_candidates",
        "aa_select_used_evidence",
        "aa_materialize_evidence_pack",
        "aa_generate_answer_with_citations",
        "aa_validate_or_repair_citations",
        "aa_evaluate_answer",
        "aa_project_pointers",
        "aa_maybe_iterate",
        "aa_persist_response",
        "end",
    )


def build_agentic_answering_workflow_design(
    *, workflow_id: str = AGENTIC_ANSWERING_WORKFLOW_ID
) -> WorkflowDesignArtifact:
    wid = lambda suffix: f"wf:{workflow_id}:{suffix}"

    nodes = [
        _node(workflow_id=workflow_id, node_id=wid("start"), label="Start", op="start", start=True),
        _node(
            workflow_id=workflow_id,
            node_id=wid("prepare"),
            label="Prepare",
            op="aa_prepare",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("view"),
            label="Get view + question",
            op="aa_get_view_and_question",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("retrieve"),
            label="Retrieve candidates",
            op="aa_retrieve_candidates",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("select"),
            label="Select used evidence",
            op="aa_select_used_evidence",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("materialize"),
            label="Materialize evidence pack",
            op="aa_materialize_evidence_pack",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("answer"),
            label="Generate answer with citations",
            op="aa_generate_answer_with_citations",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("repair"),
            label="Validate/repair citations",
            op="aa_validate_or_repair_citations",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("eval"),
            label="Evaluate answer",
            op="aa_evaluate_answer",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("project"),
            label="Project pointers",
            op="aa_project_pointers",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("iterate"),
            label="Maybe iterate",
            op="aa_maybe_iterate",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("persist"),
            label="Persist assistant + link run",
            op="aa_persist_response",
        ),
        _node(
            workflow_id=workflow_id,
            node_id=wid("end"),
            label="End",
            op="end",
            terminal=True,
        ),
    ]

    edges = [
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e1"),
            src=wid("start"),
            dst=wid("prepare"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e2"),
            src=wid("prepare"),
            dst=wid("view"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e3"),
            src=wid("view"),
            dst=wid("retrieve"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e4"),
            src=wid("retrieve"),
            dst=wid("select"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e5"),
            src=wid("select"),
            dst=wid("materialize"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e6"),
            src=wid("materialize"),
            dst=wid("answer"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e7"),
            src=wid("answer"),
            dst=wid("repair"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e8"),
            src=wid("repair"),
            dst=wid("eval"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e9"),
            src=wid("eval"),
            dst=wid("project"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e10"),
            src=wid("project"),
            dst=wid("iterate"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e11"),
            src=wid("iterate"),
            dst=wid("retrieve"),
            relation="wf_conditional",
            predicate="aa_should_iterate",
            priority=0,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e12"),
            src=wid("iterate"),
            dst=wid("persist"),
            relation="wf_next",
            predicate="always",
            priority=100,
            is_default=True,
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=wid("e13"),
            src=wid("persist"),
            dst=wid("end"),
            relation="wf_next",
            predicate=None,
            is_default=True,
        ),
    ]

    return WorkflowDesignArtifact(
        workflow_id=workflow_id,
        workflow_version="v2",
        start_node_id=wid("start"),
        nodes=nodes,
        edges=edges,
        source_run_id=None,
        source_workflow_id=None,
        source_step_id=None,
        notes="Canonical agentic answering workflow design.",
    )


def materialize_workflow_design_artifact(
    workflow_engine, design: WorkflowDesignArtifact
) -> None:
    for node in design.nodes:
        workflow_engine.write.add_node(node)
    for edge in design.edges:
        workflow_engine.write.add_edge(edge)


def _inject_span_insertion_method(value, *, insertion_method: str) -> None:
    if isinstance(value, dict):
        spans = value.get("spans")
        if isinstance(spans, list):
            for span in spans:
                if isinstance(span, dict) and not span.get("insertion_method"):
                    span["insertion_method"] = insertion_method
        for child in value.values():
            _inject_span_insertion_method(child, insertion_method=insertion_method)
    elif isinstance(value, list):
        for item in value:
            _inject_span_insertion_method(item, insertion_method=insertion_method)


def build_agentic_answering_backend_payload(
    *,
    workflow_id: str = AGENTIC_ANSWERING_WORKFLOW_ID,
    insertion_method: str = "system",
) -> dict:
    payload = build_agentic_answering_workflow_design(workflow_id=workflow_id).model_dump(
        mode="backend"
    )
    payload = copy.deepcopy(payload)
    _inject_span_insertion_method(payload, insertion_method=insertion_method)
    return payload


def build_agentic_answering_frontend_payload(
    *,
    workflow_id: str = AGENTIC_ANSWERING_WORKFLOW_ID,
) -> dict:
    return build_agentic_answering_workflow_design(workflow_id=workflow_id).model_dump(
        mode="frontend"
    )
