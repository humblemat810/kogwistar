"""
workflow/contract.py

Workflow graph contract (static topology, dynamic routing).

Topology storage:
- Uses your existing GraphKnowledgeEngine node/edge persistence (Chroma collections).
- Workflow nodes/edges are distinguished by metadata:

    Node.metadata["entity_type"] == "workflow_node"
    Node.metadata["workflow_id"] == <workflow_id>

    Edge.metadata["entity_type"] == "workflow_edge"
    Edge.metadata["workflow_id"] == <workflow_id>

Dynamic routing:
- Edge.metadata["wf_predicate"] is a symbolic predicate name resolved via a registry.
- Predicates evaluate (state, last_result) -> bool.

Parallel fanout:
- A node completion can spawn multiple outgoing steps when:
  - node.metadata["wf_fanout"] == True, OR
  - edge.metadata["wf_multiplicity"] == "many"

Cycles allowed:
- Cycles are allowed, but there MUST be at least one terminal exit reachable from the start
  (over-approx, ignoring predicates).

Conventions:
- Start node: exactly one node has metadata["wf_start"] == True for a workflow_id.
- Terminal node: metadata["wf_terminal"] == True, OR node has no outgoing workflow edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from .models import get_route_next_names
from .serialize import JsonValue

if TYPE_CHECKING:
    from ..engine_core.engine import GraphKnowledgeEngine

Json = JsonValue
State = dict[str, Json]
Result = Json


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    start_node_id: str


@dataclass(frozen=True)
class WorkflowNodeInfo:
    node_id: str
    op: str
    version: str
    cacheable: bool
    terminal: bool
    fanout: bool


from ..engine_core.models import Edge, Node


class WorkflowEdgeLike(Protocol):
    """Persisted edge surface required by workflow routing."""

    @property
    def source_ids(self) -> Sequence[str]: ...

    @property
    def target_ids(self) -> Sequence[str]: ...

    @property
    def metadata(self) -> Mapping[str, object]: ...

    @property
    def label(self) -> str: ...

    def safe_get_id(self) -> str: ...


@dataclass(frozen=True)
class WorkflowEdgeInfo:
    name: str
    edge_id: str
    src: str
    dst: str
    predicate: None | str
    priority: int
    is_default: bool
    multiplicity: str  # "one" | "many"

    @staticmethod
    def from_workflow_edge(e: WorkflowEdgeLike) -> "WorkflowEdgeInfo":
        src = e.source_ids[0]
        dst = e.target_ids[0]
        md = e.metadata
        predicate_value = md.get("wf_predicate")
        priority_value = md.get("wf_priority", 100)
        info = WorkflowEdgeInfo(
            name=e.label,
            edge_id=e.safe_get_id(),
            src=str(src),
            dst=str(dst),
            predicate=predicate_value if isinstance(predicate_value, str) else None,
            priority=(
                int(priority_value)
                if isinstance(priority_value, (int, float, str))
                else 100
            ),
            is_default=bool(md.get("wf_is_default", False)),
            multiplicity=str(md.get("wf_multiplicity", "one")),
        )
        return info


class Predicate(Protocol):
    """Evaluate one workflow edge without mutating runtime state."""

    def __call__(
        self,
        edge: WorkflowEdgeInfo,
        state: Mapping[str, object],
        result: object,
    ) -> bool: ...


class CancellationChecker(Protocol):
    """Report whether a run has received a cancellation request."""

    def __call__(self, run_id: str, /) -> bool: ...


class BasePredicate:
    def __call__(
        self,
        edge: WorkflowEdgeInfo,
        state: Mapping[str, object],
        result: object,
    ) -> bool:
        route_names = get_route_next_names(result)
        if route_names:
            return edge.name in route_names
        else:
            return True  # always true if step does not specify next step names


def build_workflow_from_engine(
    *, engine: GraphKnowledgeEngine, workflow_id: str
) -> WorkflowSpec:
    """
    Loads workflow spec from engine via convention:
      - Exactly one workflow_node has wf_start=True for the workflow_id.

    If you want multiple specs per workflow_id, extend this by adding a workflow_variant field.
    """
    nodes = engine.get_nodes(
        where={
            "$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]
        },
        limit=500,
    )
    start = None
    for n in nodes:
        if (n.metadata or {}).get("wf_start") is True:
            start = n.safe_get_id()
            break
    if start is None:
        raise ValueError(
            f"No start node found for workflow_id={workflow_id!r}. "
            "Set node.metadata['wf_start']=True on exactly one workflow node."
        )
    return WorkflowSpec(workflow_id=workflow_id, start_node_id=start)


def _iter_wf_nodes(*, engine: GraphKnowledgeEngine, workflow_id: str) -> list[Node]:
    return engine.get_nodes(
        where={
            "$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]
        },
        limit=2000,
    )


def _iter_wf_edges(*, engine: GraphKnowledgeEngine, workflow_id: str) -> list[Edge]:
    return engine.get_edges(
        where={
            "$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]
        },
        limit=5000,
    )


def load_workflow_graph(
    *,
    engine: GraphKnowledgeEngine,
    spec: WorkflowSpec,
) -> tuple[dict[str, WorkflowNodeInfo], dict[str, list[WorkflowEdgeInfo]]]:
    """Loads node/edge info needed by the executor."""
    nodes_raw = _iter_wf_nodes(engine=engine, workflow_id=spec.workflow_id)
    edges_raw = _iter_wf_edges(engine=engine, workflow_id=spec.workflow_id)

    nodes: dict[str, WorkflowNodeInfo] = {}
    for n in nodes_raw:
        md = n.metadata or {}
        info = WorkflowNodeInfo(
            node_id=n.safe_get_id(),
            op=str(md.get("wf_op") or md.get("wf.op") or ""),
            version=str(md.get("wf_version") or md.get("wf.version") or "v1"),
            cacheable=bool(md.get("wf_cacheable", True)),
            terminal=bool(md.get("wf_terminal", False)),
            fanout=bool(md.get("wf_fanout", False)),
        )
        if not info.op and not info.terminal:
            raise ValueError(f"workflow node {n.safe_get_id()} missing metadata wf_op")
        nodes[n.safe_get_id()] = info

    adj: dict[str, list[WorkflowEdgeInfo]] = {nid: [] for nid in nodes.keys()}
    for e in edges_raw:
        md = e.metadata or {}

        # Your Edge model stores endpoints in source_ids/target_ids lists.
        src = e.source_ids[0] if getattr(e, "source_ids", None) else md.get("src")
        dst = e.target_ids[0] if getattr(e, "target_ids", None) else md.get("dst")

        if src not in nodes or dst not in nodes:
            raise ValueError(
                f"workflow edge endpoints not workflow nodes: {src} -> {dst}"
            )

        info = WorkflowEdgeInfo.from_workflow_edge(e)
        adj[str(src)].append(info)

    # deterministic ordering
    for s in adj:
        adj[s].sort(key=lambda x: x.priority)

    if spec.start_node_id not in nodes:
        raise ValueError(
            f"start_node_id {spec.start_node_id!r} is not a workflow node for workflow_id={spec.workflow_id!r}"
        )
    return nodes, adj


def validate_workflow(
    *,
    engine: GraphKnowledgeEngine,
    spec: WorkflowSpec,
    predicate_registry: dict[str, Predicate],
) -> None:
    """
    Validates:
    - start node exists
    - all edges connect wf nodes
    - predicate names resolve
    - cycle allowed, BUT at least one terminal must be reachable from start (ignoring predicates)
    """
    nodes, adj = load_workflow_graph(engine=engine, spec=spec)

    for edges in adj.values():
        for e in edges:
            if e.predicate is not None and e.predicate not in predicate_registry:
                raise ValueError(
                    f"Unknown predicate {e.predicate!r} on edge {e.edge_id}"
                )

    terminal = {
        nid for nid, n in nodes.items() if n.terminal or len(adj.get(nid, [])) == 0
    }
    if not terminal:
        raise ValueError(
            "Workflow has no terminal nodes (wf_terminal=True) and no sink nodes"
        )

    from kogwistar._rust_bridge import (
        runtime_implementation_mode,
        runtime_workflow_terminal_reachable,
    )

    edges = [
        (str(source), str(edge.dst))
        for source, workflow_edges in adj.items()
        for edge in workflow_edges
    ]
    if runtime_implementation_mode() == "rust":
        terminal_reachable = bool(
            runtime_workflow_terminal_reachable(
                node_ids=list(nodes),
                edges=edges,
                start_node_id=spec.start_node_id,
                terminal_ids=sorted(terminal),
            )
        )
    else:
        seen = set()
        stack = [spec.start_node_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for edge in adj.get(cur, []):
                if edge.dst not in seen:
                    stack.append(edge.dst)
        python_value = any(node_id in seen for node_id in terminal)
        selected = runtime_workflow_terminal_reachable(
            node_ids=list(nodes),
            edges=edges,
            start_node_id=spec.start_node_id,
            terminal_ids=sorted(terminal),
            python_value=python_value,
        )
        terminal_reachable = python_value if selected is None else selected

    if not terminal_reachable:
        raise ValueError(
            "No terminal reachable from start (ignoring predicates). Cyclic graph without exit."
        )
