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
from typing import Any, Callable, Dict, List, Optional, Tuple

Json = Any
State = Dict[str, Json]
Result = Json

Predicate = Callable[[State, Result], bool]


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


@dataclass(frozen=True)
class WorkflowEdgeInfo:
    edge_id: str
    src: str
    dst: str
    predicate: Optional[str]
    priority: int
    is_default: bool
    multiplicity: str  # "one" | "many"


def build_workflow_from_engine(*, engine: Any, workflow_id: str) -> WorkflowSpec:
    """
    Loads workflow spec from engine via convention:
      - Exactly one workflow_node has wf_start=True for the workflow_id.

    If you want multiple specs per workflow_id, extend this by adding a workflow_variant field.
    """
    nodes = engine.get_nodes(where={"entity_type": "workflow_node", "workflow_id": workflow_id}, limit=500)
    start = None
    for n in nodes:
        if (n.metadata or {}).get("wf_start") is True:
            start = n.id
            break
    if start is None:
        raise ValueError(
            f"No start node found for workflow_id={workflow_id!r}. "
            "Set node.metadata['wf_start']=True on exactly one workflow node."
        )
    return WorkflowSpec(workflow_id=workflow_id, start_node_id=start)


def _iter_wf_nodes(*, engine: Any, workflow_id: str) -> List[Any]:
    return engine.get_nodes(where={"entity_type": "workflow_node", "workflow_id": workflow_id}, limit=2000)


def _iter_wf_edges(*, engine: Any, workflow_id: str) -> List[Any]:
    return engine.get_edges(where={"entity_type": "workflow_edge", "workflow_id": workflow_id}, limit=5000)


def load_workflow_graph(
    *,
    engine: Any,
    spec: WorkflowSpec,
) -> Tuple[Dict[str, WorkflowNodeInfo], Dict[str, List[WorkflowEdgeInfo]]]:
    """Loads node/edge info needed by the executor."""
    nodes_raw = _iter_wf_nodes(engine=engine, workflow_id=spec.workflow_id)
    edges_raw = _iter_wf_edges(engine=engine, workflow_id=spec.workflow_id)

    nodes: Dict[str, WorkflowNodeInfo] = {}
    for n in nodes_raw:
        md = n.metadata or {}
        info = WorkflowNodeInfo(
            node_id=n.id,
            op=str(md.get("wf_op") or md.get("wf.op") or ""),
            version=str(md.get("wf_version") or md.get("wf.version") or "v1"),
            cacheable=bool(md.get("wf_cacheable", True)),
            terminal=bool(md.get("wf_terminal", False)),
            fanout=bool(md.get("wf_fanout", False)),
        )
        if not info.op and not info.terminal:
            raise ValueError(f"workflow node {n.id} missing metadata wf_op")
        nodes[n.id] = info

    adj: Dict[str, List[WorkflowEdgeInfo]] = {nid: [] for nid in nodes.keys()}
    for e in edges_raw:
        md = e.metadata or {}

        # Your Edge model stores endpoints in source_ids/target_ids lists.
        src = e.source_ids[0] if getattr(e, "source_ids", None) else md.get("src")
        dst = e.target_ids[0] if getattr(e, "target_ids", None) else md.get("dst")

        if src not in nodes or dst not in nodes:
            raise ValueError(f"workflow edge endpoints not workflow nodes: {src} -> {dst}")

        info = WorkflowEdgeInfo(
            edge_id=e.id,
            src=str(src),
            dst=str(dst),
            predicate=md.get("wf_predicate"),
            priority=int(md.get("wf_priority", 100)),
            is_default=bool(md.get("wf_is_default", False)),
            multiplicity=str(md.get("wf_multiplicity", "one")),
        )
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
    engine: Any,
    spec: WorkflowSpec,
    predicate_registry: Dict[str, Predicate],
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
                raise ValueError(f"Unknown predicate {e.predicate!r} on edge {e.edge_id}")

    terminal = {nid for nid, n in nodes.items() if n.terminal or len(adj.get(nid, [])) == 0}
    if not terminal:
        raise ValueError("Workflow has no terminal nodes (wf_terminal=True) and no sink nodes")

    seen = set()
    stack = [spec.start_node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in adj.get(cur, []):
            if e.dst not in seen:
                stack.append(e.dst)

    if not any(t in seen for t in terminal):
        raise ValueError("No terminal reachable from start (ignoring predicates). Cyclic graph without exit.")
