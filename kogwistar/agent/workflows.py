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
    *,
    workflow_id: str = "agent.normal.v1",
    execute_op: str = "agent.execute",
    control_op: str | None = None,
    ack_op: str | None = None,
    control_metadata: dict[str, object] | None = None,
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
    control = f"wf:{workflow_id}:control"
    ack = f"wf:{workflow_id}:control-ack"
    if ack_op and not control_op:
        raise ValueError("ack_op requires control_op")
    if control_op:
        nodes.insert(
            1,
            _node(
                workflow_id=workflow_id,
                node_id=control,
                op=control_op,
                label="Control point",
                metadata={"wf_control_point": True, **(control_metadata or {})},
            ),
        )
        if ack_op:
            nodes.insert(
                2,
                _node(
                    workflow_id=workflow_id,
                    node_id=ack,
                    op=ack_op,
                    label="Control acknowledgement",
                    metadata={"wf_control_ack": True},
                ),
            )
    edges = [
        _edge(
            workflow_id=workflow_id,
            edge_id=f"{start}->{control if control_op else execute}",
            source=start,
            target=control if control_op else execute,
        ),
        _edge(workflow_id=workflow_id, edge_id=f"{execute}->{done}", source=execute, target=done),
    ]
    if control_op:
        edges.insert(
            1,
            _edge(
                workflow_id=workflow_id,
                edge_id=f"{control}->{ack if ack_op else execute}",
                source=control,
                target=ack if ack_op else execute,
            ),
        )
        if ack_op:
            edges.insert(
                2,
                _edge(
                    workflow_id=workflow_id,
                    edge_id=f"{ack}->{execute}",
                    source=ack,
                    target=execute,
                ),
            )
    return _artifact(
        workflow_id=workflow_id,
        mode="normal",
        nodes=nodes,
        edges=edges,
        notes="One-pass agent composition over WorkflowRuntime.",
    )


def build_plan_workflow(
    *,
    workflow_id: str = "agent.plan.v1",
    execute_op: str = "agent.execute",
    control_op: str | None = None,
    ack_op: str | None = None,
    control_metadata: dict[str, object] | None = None,
) -> WorkflowDesignArtifact:
    """Build a plan/approve/execute graph; approval remains ordinary routing."""

    names = ("observe", "plan", "approve", "execute", "rejected", "done")
    ids = {name: f"wf:{workflow_id}:{name}" for name in names}
    nodes = [
        _node(workflow_id=workflow_id, node_id=ids["observe"], op="agent.observe", label="Observe", start=True),
        _node(workflow_id=workflow_id, node_id=ids["plan"], op="agent.plan", label="Plan"),
        _node(workflow_id=workflow_id, node_id=ids["approve"], op="agent.approve", label="Approve"),
        _node(workflow_id=workflow_id, node_id=ids["execute"], op=execute_op, label="Execute"),
        _node(workflow_id=workflow_id, node_id=ids["rejected"], op="agent.rejected", label="Rejected", terminal=True),
        _node(workflow_id=workflow_id, node_id=ids["done"], op="agent.done", label="Done", terminal=True),
    ]
    control = f"wf:{workflow_id}:control"
    ack = f"wf:{workflow_id}:control-ack"
    if ack_op and not control_op:
        raise ValueError("ack_op requires control_op")
    if control_op:
        nodes.insert(
            4,
            _node(
                workflow_id=workflow_id,
                node_id=control,
                op=control_op,
                label="Control point",
                metadata={"wf_control_point": True, **(control_metadata or {})},
            ),
        )
        if ack_op:
            nodes.insert(
                5,
                _node(
                    workflow_id=workflow_id,
                    node_id=ack,
                    op=ack_op,
                    label="Control acknowledgement",
                    metadata={"wf_control_ack": True},
                ),
            )
    edges = [
        _edge(workflow_id=workflow_id, edge_id=f"{ids['observe']}->{ids['plan']}", source=ids["observe"], target=ids["plan"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['plan']}->{ids['approve']}", source=ids["plan"], target=ids["approve"]),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"{ids['approve']}->{control if control_op else ids['execute']}",
            source=ids["approve"],
            target=control if control_op else ids["execute"],
            predicate="approved",
        ),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"{ids['approve']}->{ids['rejected']}",
            source=ids["approve"],
            target=ids["rejected"],
            predicate="rejected",
            is_default=False,
            priority=101,
        ),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['execute']}->{ids['done']}", source=ids["execute"], target=ids["done"]),
    ]
    if control_op:
        edges.insert(
            3,
            _edge(
                workflow_id=workflow_id,
                edge_id=f"{control}->{ack if ack_op else ids['execute']}",
                source=control,
                target=ack if ack_op else ids["execute"],
            ),
        )
        if ack_op:
            edges.insert(
                4,
                _edge(
                    workflow_id=workflow_id,
                    edge_id=f"{ack}->{ids['execute']}",
                    source=ack,
                    target=ids["execute"],
                ),
            )
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
    control_op: str | None = None,
    ack_op: str | None = None,
    control_metadata: dict[str, object] | None = None,
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
    control = f"wf:{workflow_id}:control"
    ack = f"wf:{workflow_id}:control-ack"
    if ack_op and not control_op:
        raise ValueError("ack_op requires control_op")
    if control_op:
        nodes.insert(
            2,
            _node(
                workflow_id=workflow_id,
                node_id=control,
                op=control_op,
                label="Control point",
                metadata={"wf_control_point": True, **(control_metadata or {})},
            ),
        )
        if ack_op:
            nodes.insert(
                3,
                _node(
                    workflow_id=workflow_id,
                    node_id=ack,
                    op=ack_op,
                    label="Control acknowledgement",
                    metadata={"wf_control_ack": True},
                ),
            )
    edges = [
        _edge(workflow_id=workflow_id, edge_id=f"{ids['observe']}->{ids['decide']}", source=ids["observe"], target=ids["decide"]),
        _edge(
            workflow_id=workflow_id,
            edge_id=f"{ids['decide']}->{control if control_op else ids['act']}",
            source=ids["decide"],
            target=control if control_op else ids["act"],
        ),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['act']}->{ids['check']}", source=ids["act"], target=ids["check"]),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['check']}->{ids['done']}", source=ids["check"], target=ids["done"], predicate=satisfied_predicate, is_default=False),
        _edge(workflow_id=workflow_id, edge_id=f"{ids['check']}->{ids['observe']}", source=ids["check"], target=ids["observe"], predicate=continue_predicate, is_default=False),
    ]
    if control_op:
        edges.insert(
            2,
            _edge(
                workflow_id=workflow_id,
                edge_id=f"{control}->{ack if ack_op else ids['act']}",
                source=control,
                target=ack if ack_op else ids["act"],
            ),
        )
        if ack_op:
            edges.insert(
                3,
                _edge(
                    workflow_id=workflow_id,
                    edge_id=f"{ack}->{ids['act']}",
                    source=ack,
                    target=ids["act"],
                ),
            )
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


def validate_agent_design(
    design: WorkflowDesignArtifact,
    *,
    mode: AgentWorkflowMode | None = None,
) -> tuple[str, ...]:
    """Return design-lint diagnostics without changing runtime semantics."""

    diagnostics: list[str] = []
    if mode is not None and mode not in {"normal", "plan", "goal"}:
        diagnostics.append(f"unsupported agent mode: {mode}")
    starts = [node for node in design.nodes if node.metadata.get("wf_start")]
    terminals = [node for node in design.nodes if node.metadata.get("wf_terminal")]
    if len(starts) != 1:
        diagnostics.append("agent design must have exactly one start node")
    if not terminals:
        diagnostics.append("agent design must retain a terminal node")
    declared = design.nodes[0].metadata.get("wf_mode") if design.nodes else None
    if mode is not None and declared not in (None, mode):
        diagnostics.append("advisory workflow mode metadata disagrees with requested mode")
    if mode == "plan":
        ops = {str(node.metadata.get("wf_op")) for node in design.nodes}
        if "agent.plan" not in ops or "agent.approve" not in ops:
            diagnostics.append("plan design should contain plan and approve nodes")
    if mode == "goal":
        ops = {str(node.metadata.get("wf_op")) for node in design.nodes}
        if "agent.decide" not in ops or "agent.check" not in ops:
            diagnostics.append("goal design should contain decide and check nodes")
    return tuple(diagnostics)


def validate_control_point_placement(
    design: WorkflowDesignArtifact,
) -> tuple[str, ...]:
    """Lint placement metadata without changing runtime guards or branches."""

    diagnostics: list[str] = []
    node_ids = {str(node.safe_get_id()) for node in design.nodes}
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: 0 for node_id in node_ids}
    for edge in design.edges:
        source = str(edge.source_ids[0]) if edge.source_ids else ""
        target = str(edge.target_ids[0]) if edge.target_ids else ""
        if source in outgoing:
            outgoing[source] += 1
        if target in incoming:
            incoming[target] += 1
    forbidden = {"wf_budget_override", "wf_capability_grant", "wf_cancel_runtime"}
    for node in design.nodes:
        if not node.metadata.get("wf_control_point"):
            continue
        node_id = str(node.safe_get_id())
        if node.metadata.get("wf_terminal"):
            diagnostics.append(f"control point cannot be terminal: {node_id}")
        if incoming.get(node_id, 0) == 0 or outgoing.get(node_id, 0) == 0:
            diagnostics.append(f"control point must be reachable and continue: {node_id}")
        if forbidden.intersection(node.metadata):
            diagnostics.append(f"control point metadata cannot override runtime guards: {node_id}")
    return tuple(diagnostics)
