"""Loss-aware CrewAI-shaped static flow importer.

This accepts a mapping rather than importing CrewAI. Dynamic manager behavior,
callbacks, opaque memory, and guardrails are diagnosed instead of executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kogwistar.runtime.models import WorkflowDesignArtifact
from kogwistar.runtime.models import WorkflowInvocationRequest

from kogwistar.agent.workflows import _artifact, _edge, _node


@dataclass(frozen=True, slots=True)
class CrewAIImportDiagnostic:
    code: str
    message: str
    severity: str = "warning"


def import_crewai_flow(description: Mapping[str, Any]) -> tuple[WorkflowDesignArtifact, tuple[CrewAIImportDiagnostic, ...]]:
    """Translate static task/dependency mappings into an ordinary workflow."""

    workflow_id = str(description.get("workflow_id") or "crewai.imported.v1")
    raw_nodes = list(description.get("tasks") or description.get("nodes") or [])
    if not raw_nodes:
        raise ValueError("CrewAI import requires a non-empty static task list")
    diagnostics: list[CrewAIImportDiagnostic] = []
    for field, code in (("manager", "dynamic_manager"), ("callbacks", "callbacks"), ("memory", "opaque_memory"), ("guardrails", "guardrails")):
        if description.get(field):
            diagnostics.append(CrewAIImportDiagnostic(code, f"{field} is not imported into runtime authority"))
    ids: list[str] = []
    nodes = []
    def _task_value(raw: Any, index: int, key: str) -> Any:
        return raw.get(key) if isinstance(raw, Mapping) else None

    def _task_key(raw: Any, index: int) -> str:
        return str(_task_value(raw, index, "id") or _task_value(raw, index, "name") or index)

    for index, raw in enumerate(raw_nodes):
        item = dict(raw) if isinstance(raw, Mapping) else {"name": str(raw)}
        node_id = f"wf:{workflow_id}:{item.get('id') or item.get('name') or index}"
        ids.append(node_id)
        nodes.append(
            _node(
                workflow_id=workflow_id,
                node_id=node_id,
                op=str(item.get("op") or "agent.execute"),
                label=str(item.get("name") or item.get("id") or f"Task {index}"),
                start=index == 0,
                terminal=index == len(raw_nodes) - 1,
                metadata={
                    "interop_source": "crewai",
                    "interop_task_id": _task_key(raw, index),
                    "role": item.get("role"),
                    "agent_profile_id": item.get("agent") or item.get("role"),
                    "delegates_to": item.get("delegates_to"),
                },
            )
        )
    edges = []
    for index, raw in enumerate(raw_nodes):
        item = dict(raw) if isinstance(raw, Mapping) else {}
        depends = item.get("depends_on") or item.get("dependencies")
        if depends:
            values = [depends] if isinstance(depends, str) else list(depends)
            for dep in values:
                dep_index = next(
                    (i for i, candidate in enumerate(raw_nodes) if _task_key(candidate, i) == str(dep)),
                    None,
                )
                if dep_index is None:
                    raise ValueError(f"CrewAI task {ids[index]!r} references unknown dependency {dep!r}")
                dep_id = ids[dep_index]
                edges.append(_edge(workflow_id=workflow_id, edge_id=f"{dep_id}->{ids[index]}", source=dep_id, target=ids[index]))
        elif index > 0:
            edges.append(_edge(workflow_id=workflow_id, edge_id=f"{ids[index - 1]}->{ids[index]}", source=ids[index - 1], target=ids[index]))
    if not edges and len(ids) > 1:
        raise ValueError("CrewAI dependencies cannot form an empty execution graph")
    design = _artifact(
        workflow_id=workflow_id,
        mode="normal",
        nodes=nodes,
        edges=edges,
        notes="Imported static CrewAI subset; unsupported behavior remains diagnostic.",
    )
    return design, tuple(diagnostics)


def crewai_delegated_invocation(
    task: Mapping[str, Any], *, parent_run_id: str, step_seq: int
) -> WorkflowInvocationRequest:
    """Map an explicit CrewAI delegation target to normal nested invocation."""

    workflow_id = str(task.get("workflow_id") or task.get("delegates_to") or "").strip()
    if not workflow_id:
        raise ValueError("CrewAI delegation requires an explicit workflow_id")
    return WorkflowInvocationRequest(
        workflow_id=workflow_id,
        invocation_key=f"{parent_run_id}:{step_seq}:{workflow_id}",
        result_state_key="agent_action_result",
    )


def static_semantic_signature(description: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return a stable supported-subset signature for importer round trips."""

    raw_nodes = list(description.get("tasks") or description.get("nodes") or [])
    names = tuple(
        str(item.get("id") or item.get("name") or index)
        if isinstance(item, Mapping)
        else str(item)
        for index, item in enumerate(raw_nodes)
    )
    edges: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_nodes):
        item = dict(raw) if isinstance(raw, Mapping) else {}
        depends = item.get("depends_on") or item.get("dependencies")
        values = [depends] if isinstance(depends, str) else list(depends or ())
        if values:
            for dep in values:
                edges.append((str(dep), names[index]))
        elif index > 0:
            edges.append((names[index - 1], names[index]))
    return names, tuple(edges)


def imported_semantic_signature(design: WorkflowDesignArtifact) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    nodes = tuple(str(node.metadata.get("interop_task_id") or node.label) for node in design.nodes)
    by_id = {str(node.id): str(node.metadata.get("interop_task_id") or node.label) for node in design.nodes}
    edges = tuple(
        (by_id[str(edge.source_ids[0])], by_id[str(edge.target_ids[0])])
        for edge in design.edges
        if str(edge.source_ids[0]) in by_id and str(edge.target_ids[0]) in by_id
    )
    return nodes, edges


__all__ = [
    "CrewAIImportDiagnostic",
    "crewai_delegated_invocation",
    "import_crewai_flow",
    "imported_semantic_signature",
    "static_semantic_signature",
]
