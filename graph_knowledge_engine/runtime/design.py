from __future__ import annotations

from runtime.models import WorkflowNode

from .resolvers import BaseResolver

"""In-memory workflow design helpers.

The codebase supports two workflow "spec" shapes:

1) Engine-backed: a minimal spec (workflow_id + start_node_id) that is resolved
   into nodes/edges by reading from a GraphKnowledgeEngine. See
   :mod:`graph_knowledge_engine.runtime.contract`.

2) In-memory: a rich spec that contains explicit nodes and edges and can be run
   without any engine. This module provides that rich spec.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from runtime.models import WorkflowEdge
from graph_knowledge_engine.engine import GraphKnowledgeEngine
from .contract import BasePredicate

PredicateName = Optional[str]
Predicate = BasePredicate

@dataclass(frozen=True)
class WFNode:
    node_id: str
    workflow_id: str
    op: str
    version: str = "v1"
    start: bool = False
    terminal: bool = False
    fanout: bool = False
    cacheable: bool = True


@dataclass(frozen=True)
class WFEdge:
    edge_id: str
    workflow_id: str
    src: str
    dst: str
    predicate: PredicateName
    priority: int = 100
    is_default: bool = False
    multiplicity: str = "one"  # "one" | "many"


@dataclass(frozen=True)
class WorkflowSpec:
    """Rich in-memory workflow spec."""

    workflow_id: str
    start_node_id: str
    nodes: Dict[str, WFNode]
    out_edges: Dict[str, List[WFEdge]]


def build_workflow_from_engine(*, workflow_engine: Any, workflow_id: str) -> WorkflowSpec:
    """Load a rich :class:`WorkflowSpec` from a GraphKnowledgeEngine-like API.

    Required engine interface:
      - get_nodes(where=..., limit=...)
      - get_edges(where=..., limit=...)

    This matches the conventions in :mod:`graph_knowledge_engine.runtime.contract`:
      node.metadata["entity_type"] == "workflow_node"
      edge.metadata["entity_type"] == "workflow_edge"
    """

    nodes_raw = workflow_engine.get_nodes(where={"$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]}, limit=5000)
    edges_raw = workflow_engine.get_edges(where={"$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]}, limit=20000)

    nodes: Dict[str, WFNode] = {}
    start_node_id: Optional[str] = None

    for n in nodes_raw:
        md = n.metadata or {}
        wn = WFNode(
            node_id=n.id,
            workflow_id=str(md.get("workflow_id") or workflow_id),
            op=str(md.get("wf_op") or md.get("wf.op") or ""),
            version=str(md.get("wf_version") or md.get("wf.version") or "v1"),
            start=bool(md.get("wf_start", False)),
            terminal=bool(md.get("wf_terminal", False)),
            fanout=bool(md.get("wf_fanout", False)),
            cacheable=bool(md.get("wf_cacheable", True)),
        )
        if not wn.op and not wn.terminal:
            raise ValueError(f"workflow node {n.id!r} missing metadata wf_op")
        nodes[wn.node_id] = wn
        if wn.start:
            if start_node_id is not None and start_node_id != wn.node_id:
                raise ValueError(
                    f"workflow_id={workflow_id!r} must have exactly one start node (wf_start=True). "
                    f"Found at least two: {start_node_id!r} and {wn.node_id!r}"
                )
            start_node_id = wn.node_id

    if start_node_id is None:
        raise ValueError(
            f"No start node found for workflow_id={workflow_id!r}. "
            "Set node.metadata['wf_start']=True on exactly one workflow node."
        )

    out_edges: Dict[str, List[WFEdge]] = {nid: [] for nid in nodes.keys()}

    for e in edges_raw:
        md = e.metadata or {}
        src = e.source_ids[0] if getattr(e, "source_ids", None) else md.get("src")
        dst = e.target_ids[0] if getattr(e, "target_ids", None) else md.get("dst")
        if src not in nodes or dst not in nodes:
            raise ValueError(f"workflow edge endpoints not workflow nodes: {src!r} -> {dst!r}")

        we = WFEdge(
            edge_id=e.id,
            workflow_id=str(md.get("workflow_id") or workflow_id),
            src=str(src),
            dst=str(dst),
            predicate=md.get("wf_predicate"),
            priority=int(md.get("wf_priority", 100)),
            is_default=bool(md.get("wf_is_default", False)),
            multiplicity=str(md.get("wf_multiplicity", "one")),
        )
        out_edges[str(src)].append(we)

    for s in out_edges:
        out_edges[s].sort(key=lambda x: x.priority)

    return WorkflowSpec(workflow_id=workflow_id, start_node_id=start_node_id, nodes=nodes, out_edges=out_edges)

def load_workflow_design(*, workflow_engine: GraphKnowledgeEngine, workflow_id: str) -> Tuple[
            WorkflowNode, Dict[str, WorkflowNode], Dict[str, List[WorkflowEdge]], Dict[str, List[WorkflowEdge]]]:
    """
    Load workflow graph design from workflow_engine.
    Nodes/edges must be tagged with:
      node.metadata.entity_type="workflow_node"
      edge.metadata.entity_type="workflow_edge"
    """
    nodes_raw: list[WorkflowNode] = workflow_engine.get_nodes(
            where={"$and": [{"entity_type": "workflow_node"},
                            { "workflow_id": workflow_id}]}, 
            limit=5000, node_type = WorkflowNode)
    edges_raw: list[WorkflowEdge] = workflow_engine.get_edges(
            where={"$and": [{"entity_type": "workflow_edge"},
                            { "workflow_id": workflow_id}]}, 
            limit=20000, edge_type = WorkflowEdge)

    nodes: Dict[str, WorkflowNode] = {}
    start_nodes: List[WorkflowNode] = []
    for n in nodes_raw:

        nodes[n.id] = n
        if n.metadata.get("wf_start"):
            start_nodes.append(n)

    if len(start_nodes) != 1:
        raise ValueError(f"workflow_id={workflow_id!r} must have exactly one start node (wf_start=True). Found {len(start_nodes)}")

    adj: Dict[str, List[WorkflowEdge]] = {nid: [] for nid in nodes}
    rev_adj : Dict[str, List[WorkflowEdge]] = {nid: [] for nid in nodes}
    for e in edges_raw:
        md = e.metadata or {}
        src = e.source_ids[0]
        dst = e.target_ids[0]
        if src not in nodes or dst not in nodes:
            raise ValueError(f"Workflow edge {e.id} connects non-workflow nodes: {src}->{dst}")
     
        adj[src].append(e)
        rev_adj[dst].append(e)
    def get_node_priority(n: WorkflowEdge):
        return n.metadata['wf_priority']
    for src in adj:
        adj[src].sort(key=get_node_priority)

    return start_nodes[0], nodes, adj, rev_adj


def validate_workflow_design(
    *,
    workflow_engine: Any,
    workflow_id: str,
    predicate_registry: Dict[str, Predicate],
    resolver: Any = None
):
    """Validate an engine-backed workflow design.

    Supports both legacy metadata keys (predicate/terminal/start/priority)
    and the newer wf_* schema (wf_predicate/wf_terminal/wf_start/wf_priority).
    """
    
    start, nodes, adj, rev_adj = load_workflow_design(workflow_engine=workflow_engine, workflow_id=workflow_id)
    resolver_ops = getattr(resolver, "ops", None)
    if resolver_ops is not None:
        if unknown_ops := (set([i.op for i in nodes.values()]) - set(resolver_ops)):
            raise Exception(f"workflow_contains unregistered ops {unknown_ops}")
    # predicate resolution
    for edges in adj.values():
        for e in edges:
            pred: str | None = e.metadata.get("wf_predicate", e.metadata.get("predicate"))
            if pred is not None and pred not in predicate_registry:
                raise ValueError(f"Unknown predicate {pred!r} on workflow edge {e.id}")

    # must have at least one terminal reachable ignoring predicates
    terminals = {
        nid
        for nid, n in nodes.items()
        if (n.metadata.get("wf_terminal", n.metadata.get("terminal")) or len(adj.get(nid, [])) == 0)
    }
    if not terminals:
        raise ValueError("Workflow has no terminal nodes and no sink nodes")

    seen: set[str] = set()
    stack = [start.id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for e in adj.get(cur, []):
            for target_id in e.target_ids:
                if target_id is None:
                    continue
                target_id = str(target_id)
                if target_id not in seen:
                    stack.append(target_id)

    if not any(t in seen for t in terminals):
        raise ValueError("No terminal reachable from start (ignoring predicates).")
    return start, nodes, adj

class BaseWorkflowDesigner:

    """Base helper for building/validating workflow designs.

    This is intentionally lightweight: it does not assume any conversation-specific
    invariants. Subclasses can add domain constraints.
    """

    def __init__(self, *, workflow_engine: Any, predicate_registry: Dict[str, Predicate], resolver: BaseResolver | None = None):
        self.workflow_engine = workflow_engine
        self.predicate_registry = predicate_registry
        self.resolver = resolver

    def validate(self, *, workflow_id: str, resolver: BaseResolver | None = None):
        return validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
            resolver=resolver or self.resolver
        )


        
