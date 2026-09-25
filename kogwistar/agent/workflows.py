"""Small ordinary-workflow templates for the thin agent harness.

The templates are graph builders only.  They do not add an agent runtime,
scheduler, token type, or special Goal executor.
"""

from __future__ import annotations

from typing import Literal

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.runtime.models import WorkflowDesignArtifact, WorkflowEdge, WorkflowNode

AgentWorkflowMode = Literal["normal", "plan", "goal"]
MODE_METADATA_KEYS = ("wf_mode", "wf_agent_contract_version")


def _grounding(workflow_id: str, text: str) -> Grounding:
    del text
    return Grounding(spans=[Span.from_dummy_for_workflow(workflow_id)])


def _node(
    *,
    workflow_id: str,
    node_id: str,
    op: str,
    label: str,
    start: bool = False,
    terminal: bool = False,
    metadata: dict[str, object] | None = None,
) -> WorkflowNode:
    node_metadata: dict[str, object] = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": op,
        "wf_start": start,
        "wf_terminal": terminal,
        "wf_version": "agent-v1",
    }
    if metadata:
        node_metadata.update(metadata)
    return WorkflowNode(
        id=node_id,
        label=label,
        type="entity",
        doc_id=node_id,
        summary=label,
        properties={},
        metadata=node_metadata,
        mentions=[_grounding(workflow_id, label)],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _edge(
    *,
    workflow_id: str,
    edge_id: str,
    source: str,
    target: str,
    predicate: str | None = None,
    is_default: bool = True,
    priority: int = 100,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[source],
        target_ids=[target],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        doc_id=edge_id,
        summary="next",
        properties={},
        mentions=[_grounding(workflow_id, edge_id)],
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_predicate": predicate,
            "wf_is_default": is_default,
            "wf_priority": priority,
            "wf_multiplicity": "one",
            "wf_version": "agent-v1",
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )


def _artifact(
    *,
    workflow_id: str,
    mode: AgentWorkflowMode,
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
    notes: str,
) -> WorkflowDesignArtifact:
    nodes[0].metadata.update(
        {"wf_mode": mode, "wf_agent_contract_version": "v1"}
    )
    return WorkflowDesignArtifact(
        workflow_id=workflow_id,
        workflow_version="agent-v1",
        start_node_id=str(nodes[0].safe_get_id()),
        nodes=nodes,
        edges=edges,
        notes=notes,
    )


def build_normal_workflow(
    *, workflow_id: str = "agent.normal.v1", execute_op: str = "agent.execute"
) -> WorkflowDesignArtifact:
    """Build a one-pass ordinary workflow."""

    start = f"wf:{workflow_id}:start"
    execute = f"wf:{workflow_id}:execute"
    done = f"wf:{workflow_id}:done"
    nodes = [
        _node(workflow_id=workflow_id, node_id=start, op="agent.observe", label="Observe", start=True),
        _node(workflow_id=workflow_id, node_id=execute, op=execute_op, label="Execute"),
        _node(workflow_id=workflow_id, node_id=done, op="agent.done", label="Done", terminal=True),
    ]
    edges = [
        _edge(workflow_id=workflow_id, edge_id=f"{start}->{execute}", source=start, target=execute),
        _edge(workflow_id=workflow_id, edge_id=f"{execute}->{done}", source=execute, target=done),
    ]
    return _artifact(
        workflow_id=workflow_id,
        mode="normal",
        nodes=nodes,
        edges=edges,
        notes="One-pass agent composition over WorkflowRuntime.",
    )


def build_plan_workflow(
    *, workflow_id: str = "agent.plan.v1", execute_op: str = "agent.execute"
) -> WorkflowDesignArtifact:
    """Build a plan/approve/execute graph; approval remains ordinary routing."""

    names = ("observe", "plan", "approve", "execute", "done")
    ids = {name: f"wf:{workflow_id}:{name}" for name in names}
    nodes = [
        _node(workflow_id=workflow_id, node_id=ids["observe"], op="agent.observe", label="Observe", start=True),
        _node(workflow_id=workflow_id, node_id=ids["plan"], op="agent.plan", label="Plan"),
        _node(workflow_id=workflow_id, node_id=ids["approve"], op="agent.approve", label="Approve"),
        _node(workflow_id=workflow_id, node_id=ids["execute"], op=execute_op, label="Execute"),
        _node(workflow_id=workflow_id, node_id=ids["done"], op="agent.done", label="Done", terminal=True),
    ]
    edges = [
        _edge(workflow_id=workflow_id, edge_id=f"{ids['observe']}->{ids['plan']}", source=ids["observe"], target=ids["plan"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['plan']}->{ids['approve']}", source=ids["plan"], target=ids["approve"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['approve']}->{ids['execute']}", source=ids["approve"], target=ids["execute"], predicate="approved"),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['execute']}->{ids['done']}", source=ids["execute"], target=ids["done"]),
    ]
    return _artifact(
        workflow_id=workflow_id,
        mode="plan",
        nodes=nodes,
        edges=edges,
        notes="Plan and approval are ordinary workflow nodes and predicates.",
    )


def build_goal_workflow(
    *,
    workflow_id: str = "agent.goal.v1",
    act_op: str = "agent.act",
    satisfied_predicate: str = "goal_satisfied",
    continue_predicate: str = "goal_not_satisfied",
) -> WorkflowDesignArtifact:
    """Build Observe -> Decide -> Act -> Check with one feedback edge."""

    names = ("observe", "decide", "act", "check", "done")
    ids = {name: f"wf:{workflow_id}:{name}" for name in names}
    nodes = [
        _node(workflow_id=workflow_id, node_id=ids["observe"], op="agent.observe", label="Observe", start=True),
        _node(workflow_id=workflow_id, node_id=ids["decide"], op="agent.decide", label="Decide"),
        _node(workflow_id=workflow_id, node_id=ids["act"], op=act_op, label="Act"),
        _node(workflow_id=workflow_id, node_id=ids["check"], op="agent.check", label="Check"),
        _node(workflow_id=workflow_id, node_id=ids["done"], op="agent.done", label="Done", terminal=True),
    ]
    edges = [
        _edge(workflow_id=workflow_id, edge_id=f"{ids['observe']}->{ids['decide']}", source=ids["observe"], target=ids["decide"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['decide']}->{ids['act']}", source=ids["decide"], target=ids["act"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['act']}->{ids['check']}", source=ids["act"], target=ids["check"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['check']}->{ids['done']}", source=ids["check"], target=ids["done"], predicate=satisfied_predicate, is_default=False),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['check']}->{ids['observe']}", source=ids["check"], target=ids["observe"], predicate=continue_predicate, is_default=False),
    ]
    return _artifact(
        workflow_id=workflow_id,
        mode="goal",
        nodes=nodes,
        edges=edges,
        notes="Goal is an annotated cyclic workflow pattern; no Goal runtime is used.",
    )


def without_agent_mode_metadata(
    design: WorkflowDesignArtifact,
) -> WorkflowDesignArtifact:
    """Remove advisory mode keys; graph execution semantics remain unchanged."""

    clone = design.model_copy(deep=True)
    for node in clone.nodes:
        for key in MODE_METADATA_KEYS:
            node.metadata.pop(key, None)
    return clone


def graph_signature(design: WorkflowDesignArtifact) -> tuple[tuple[str, ...], tuple[tuple[str, str, str | None], ...]]:
    """Return topology/op signature for metadata-removal equivalence tests."""

    nodes = tuple(sorted(f"{node.safe_get_id()}:{node.op}:{node.terminal}" for node in design.nodes))
    edges = tuple(
        sorted(
            (
                str(edge.source_ids[0]),
                str(edge.target_ids[0]),
                edge.metadata.get("wf_predicate"),
            )
            for edge in design.edges
        )
    )
    return nodes, edges
