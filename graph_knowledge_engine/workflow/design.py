from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple

from graph_knowledge_engine.workflow.contract import WorkflowSpec

from .serialize import stable_json_dumps


Predicate = Callable[[Dict[str, Any], Any], bool]


@dataclass(frozen=True)
class WFNode:
    node_id: str
    workflow_id: str
    op: str
    version: str
    start: bool
    terminal: bool
    fanout: bool


@dataclass(frozen=True)
class WFEdge:
    edge_id: str
    workflow_id: str
    src: str
    dst: str
    predicate: Optional[str]
    priority: int
    is_default: bool
    multiplicity: str  # "one"|"many"

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

def load_workflow_design(*, workflow_engine: Any, workflow_id: str) -> Tuple[WFNode, Dict[str, WFNode], Dict[str, List[WFEdge]]]:
    """
    Load workflow graph design from workflow_engine.
    Nodes/edges must be tagged with:
      node.metadata.entity_type="workflow_node"
      edge.metadata.entity_type="workflow_edge"
    """
    nodes_raw = workflow_engine.get_nodes(where={"entity_type": "workflow_node", "workflow_id": workflow_id}, limit=5000)
    edges_raw = workflow_engine.get_edges(where={"entity_type": "workflow_edge", "workflow_id": workflow_id}, limit=20000)

    nodes: Dict[str, WFNode] = {}
    start_nodes: List[WFNode] = []
    for n in nodes_raw:
        md = n.metadata or {}
        wn = WFNode(
            node_id=n.id,
            workflow_id=md["workflow_id"],
            op=md.get("wf_op") or "",
            version=md.get("wf_version") or "v1",
            start=bool(md.get("wf_start", False)),
            terminal=bool(md.get("wf_terminal", False)),
            fanout=bool(md.get("wf_fanout", False)),
        )
        nodes[wn.node_id] = wn
        if wn.start:
            start_nodes.append(wn)

    if len(start_nodes) != 1:
        raise ValueError(f"workflow_id={workflow_id!r} must have exactly one start node (wf_start=True). Found {len(start_nodes)}")

    adj: Dict[str, List[WFEdge]] = {nid: [] for nid in nodes}
    for e in edges_raw:
        md = e.metadata or {}
        src = e.source_ids[0]
        dst = e.target_ids[0]
        if src not in nodes or dst not in nodes:
            raise ValueError(f"Workflow edge {e.id} connects non-workflow nodes: {src}->{dst}")

        we = WFEdge(
            edge_id=e.id,
            workflow_id=md["workflow_id"],
            src=src,
            dst=dst,
            predicate=md.get("wf_predicate"),
            priority=int(md.get("wf_priority", 100)),
            is_default=bool(md.get("wf_is_default", False)),
            multiplicity=str(md.get("wf_multiplicity", "one")),
        )
        adj[src].append(we)

    for src in adj:
        adj[src].sort(key=lambda x: x.priority)

    return start_nodes[0], nodes, adj


def validate_workflow_design(
    *,
    workflow_engine: Any,
    workflow_id: str,
    predicate_registry: Dict[str, Predicate],
) -> None:
    start, nodes, adj = load_workflow_design(workflow_engine=workflow_engine, workflow_id=workflow_id)

    # predicate resolution
    for edges in adj.values():
        for e in edges:
            if e.predicate is not None and e.predicate not in predicate_registry:
                raise ValueError(f"Unknown predicate {e.predicate!r} on workflow edge {e.edge_id}")

    # must have at least one terminal reachable ignoring predicates
    terminals = {nid for nid, n in nodes.items() if n.terminal or len(adj.get(nid, [])) == 0}
    if not terminals:
        raise ValueError("Workflow has no terminal nodes and no sink nodes")

    seen = set()
    stack = [start.node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in adj.get(cur, []):
            if e.dst not in seen:
                stack.append(e.dst)

    if not any(t in seen for t in terminals):
        raise ValueError("No terminal reachable from start (ignoring predicates).")
