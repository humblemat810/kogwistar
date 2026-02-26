from __future__ import annotations

"""In-memory workflow design helpers.

The codebase supports two workflow "spec" shapes:

1) Engine-backed: a minimal spec (workflow_id + start_node_id) that is resolved
   into nodes/edges by reading from a GraphKnowledgeEngine. See
   :mod:`graph_knowledge_engine.workflow.contract`.

2) In-memory: a rich spec that contains explicit nodes and edges and can be run
   without any engine. This module provides that rich spec.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from graph_knowledge_engine.models import WorkflowEdge, WorkflowNode
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

    This matches the conventions in :mod:`graph_knowledge_engine.workflow.contract`:
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
):
    """Validate an engine-backed workflow design.

    Supports both legacy metadata keys (predicate/terminal/start/priority)
    and the newer wf_* schema (wf_predicate/wf_terminal/wf_start/wf_priority).
    """
    start, nodes, adj, rev_adj = load_workflow_design(workflow_engine=workflow_engine, workflow_id=workflow_id)
    adj: dict[str, list[WorkflowEdge]]

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

    def __init__(self, *, workflow_engine: Any, predicate_registry: Dict[str, Predicate]):
        self.workflow_engine = workflow_engine
        self.predicate_registry = predicate_registry

    def validate(self, *, workflow_id: str):
        return validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
        )


class ConversationWorkflowDesigner(BaseWorkflowDesigner):

    """Conversation-specific designer.

    Phase 2D Step 1/2:
    - ensure a minimal backbone design exists (start -> end)
    - validate the design schema + predicate resolution

    Phase 2D Step 3+:
    - will enforce stricter conversation invariants (chain linearity, sidecar rules, etc.)
    """

    def ensure_add_turn_flow(
        self,
        *,
        workflow_id: str,
        mode: str = "full",
        include_context_snapshot: bool = True,
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Ensure the Phase-2D add_turn workflow design exists.

        This is the single source of truth for conversation workflow charts.

        - mode="backbone": creates a minimal start->end workflow (start op="start")
        - mode="full": creates the v1-parity add_turn flow using step ops resolved by the resolver.
        """
        if mode not in ("backbone", "full"):
            raise ValueError(f"Unknown mode={mode!r}; expected 'backbone' or 'full'")

        # If already present and valid, return it.
        try:
            return self.validate(workflow_id=workflow_id)
        except Exception:
            pass

        # Lazy import to avoid circular deps for model classes.
        from ..models import WorkflowNode, WorkflowEdge

        wid = lambda suffix: f"wf:{workflow_id}:{suffix}"

        def add_node(*, node_id: str, label: str, op: str | None, start: bool = False, terminal: bool = False, fanout: bool = False):
            n = WorkflowNode(
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
                    "wf_start": start,
                    "wf_terminal": terminal,
                    "wf_fanout": fanout,
                    "wf_version": "v2",
                },
            )
            self.workflow_engine.add_node(n)

        def add_edge(*, edge_id: str, src: str, dst: str, pred: str | None, priority: int = 100, is_default: bool = False, multiplicity: str = "one"):
            e = WorkflowEdge(
                id=edge_id,
                label="wf_next",
                type="entity",
                doc_id=edge_id,
                summary="wf_next",
                properties={},
                source_id=src,
                target_ids=[dst],
                metadata={
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_edge_kind": "wf_next",
                    "wf_predicate": pred,
                    "wf_priority": priority,
                    "wf_is_default": is_default,
                    "wf_multiplicity": multiplicity,
                },
            )
            self.workflow_engine.add_edge(e)

        # ----------------------------
        # Backbone: start -> end
        # ----------------------------
        if mode == "backbone":
            start_id = wid("start")
            end_id = wid("end")
            add_node(node_id=start_id, label="Start", op="start", start=True, terminal=False)
            add_node(node_id=end_id, label="End", op=None, start=False, terminal=True)
            add_edge(edge_id=wid("next_start_end"), src=start_id, dst=end_id, pred="always", priority=100, is_default=True)
            return self.validate(workflow_id=workflow_id)

        # ----------------------------
        # Full v1-parity add_turn flow
        # ----------------------------
        add_node(node_id=wid("start"), label="Start (memory_retrieve)", op="memory_retrieve", start=True, terminal=False)
        add_node(node_id=wid("kg"), label="KG retrieve", op="kg_retrieve", fanout=True)
        add_node(node_id=wid("pin_mem"), label="Pin memory", op="memory_pin")
        add_node(node_id=wid("pin_kg"), label="Pin knowledge", op="kg_pin")
        add_node(node_id=wid("answer"), label="Answer", op="answer")

        if include_context_snapshot:
            add_node(node_id=wid("ctx_snap"), label="Context snapshot", op="context_snapshot")

        add_node(node_id=wid("decide_sum"), label="Decide summarize", op="decide_summarize")
        add_node(node_id=wid("summarize"), label="Summarize", op="summarize")
        add_node(node_id=wid("end"), label="End", op=None, terminal=True)

        # edges
        add_edge(edge_id=wid("e1"), src=wid("start"), dst=wid("kg"), pred=None, is_default=True)

        # From KG retrieve, optionally pin both memory and KG (fanout node).
        add_edge(edge_id=wid("e2"), src=wid("kg"), dst=wid("pin_mem"), pred="should_pin_memory", priority=0)
        add_edge(edge_id=wid("e3"), src=wid("kg"), dst=wid("pin_kg"), pred="should_pin_kg", priority=1)

        # If no pins happen, go straight to answer.
        add_edge(edge_id=wid("e4"), src=wid("kg"), dst=wid("answer"), pred=None, is_default=True)
        # After pins, continue to answer.
        add_edge(edge_id=wid("e5"), src=wid("pin_mem"), dst=wid("answer"), pred=None, is_default=True)
        add_edge(edge_id=wid("e6"), src=wid("pin_kg"), dst=wid("answer"), pred=None, is_default=True)

        if include_context_snapshot:
            add_edge(edge_id=wid("e7"), src=wid("answer"), dst=wid("ctx_snap"), pred=None, is_default=True)
            add_edge(edge_id=wid("e7b"), src=wid("ctx_snap"), dst=wid("decide_sum"), pred=None, is_default=True)
        else:
            add_edge(edge_id=wid("e7"), src=wid("answer"), dst=wid("decide_sum"), pred=None, is_default=True)

        add_edge(edge_id=wid("e8"), src=wid("decide_sum"), dst=wid("summarize"), pred="should_summarize", priority=0)
        add_edge(edge_id=wid("e9"), src=wid("decide_sum"), dst=wid("end"), pred=None, is_default=True)
        add_edge(edge_id=wid("e10"), src=wid("summarize"), dst=wid("end"), pred=None, is_default=True)

        return self.validate(workflow_id=workflow_id)

    def ensure_backbone(self, *, workflow_id: str) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Backward-compatible alias for backbone design.

        Kept for existing tests/callers; delegates to :meth:`ensure_add_turn_flow`.
        """
        return self.ensure_add_turn_flow(workflow_id=workflow_id, mode="backbone", include_context_snapshot=False)


class AgenticAnsweringWorkflowDesigner(BaseWorkflowDesigner):

    """Agentic-answering workflow designer.

    Documents the current V2 migration: rewrite `AgenticAnsweringAgent.answer()` into
    a workflow executed by `WorkflowRuntime`.
    """

    def ensure_answer_flow(
        self,
        *,
        workflow_id: str,
        mode: str = "full",
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        """Ensure an agentic-answering workflow exists.

        Modes:
          - mode="backbone": start -> end (op="start")
          - mode="full": agentic answering loop ops with a bounded iterate edge
        """
        if mode not in ("backbone", "full"):
            raise ValueError(f"Unknown mode={mode!r}; expected 'backbone' or 'full'")

        # If already present and valid, return it.
        try:
            return self.validate(workflow_id=workflow_id)
        except Exception:
            pass

        from ..models import WorkflowNode, WorkflowEdge

        wid = lambda suffix: f"wf:{workflow_id}:{suffix}"

        def add_node(*, node_id: str, label: str, op: str | None, start: bool = False, terminal: bool = False, fanout: bool = False):
            n = WorkflowNode(
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
                    "wf_start": start,
                    "wf_terminal": terminal,
                    "wf_fanout": fanout,
                    "wf_version": "v2",
                },
            )
            self.workflow_engine.add_node(n)

        def add_edge(*, edge_id: str, src: str, dst: str, pred: str | None, priority: int = 100, is_default: bool = False, multiplicity: str = "one"):
            e = WorkflowEdge(
                id=edge_id,
                label="wf_next",
                type="entity",
                doc_id=edge_id,
                summary="wf_next",
                properties={},
                source_id=src,
                target_ids=[dst],
                metadata={
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_edge_kind": "wf_next",
                    "wf_predicate": pred,
                    "wf_priority": priority,
                    "wf_is_default": is_default,
                    "wf_multiplicity": multiplicity,
                },
            )
            self.workflow_engine.add_edge(e)

        # ----------------------------
        # Backbone: start -> end
        # ----------------------------
        if mode == "backbone":
            start_id = wid("start")
            end_id = wid("end")
            add_node(node_id=start_id, label="Start", op="start", start=True, terminal=False)
            add_node(node_id=end_id, label="End", op=None, start=False, terminal=True)
            add_edge(edge_id=wid("next_start_end"), src=start_id, dst=end_id, pred="always", priority=100, is_default=True)
            return self.validate(workflow_id=workflow_id)

        # ----------------------------
        # Full agentic answering workflow
        # ----------------------------
        add_node(node_id=wid("start"), label="Start", op="start", start=True)
        add_node(node_id=wid("prepare"), label="Prepare", op="aa_prepare")
        add_node(node_id=wid("view"), label="Get view + question", op="aa_get_view_and_question")
        add_node(node_id=wid("retrieve"), label="Retrieve candidates", op="aa_retrieve_candidates")
        add_node(node_id=wid("select"), label="Select used evidence", op="aa_select_used_evidence")
        add_node(node_id=wid("materialize"), label="Materialize evidence pack", op="aa_materialize_evidence_pack")
        add_node(node_id=wid("answer"), label="Generate answer with citations", op="aa_generate_answer_with_citations")
        add_node(node_id=wid("repair"), label="Validate/repair citations", op="aa_validate_or_repair_citations")
        add_node(node_id=wid("eval"), label="Evaluate answer", op="aa_evaluate_answer")
        add_node(node_id=wid("project"), label="Project pointers", op="aa_project_pointers")
        add_node(node_id=wid("iterate"), label="Maybe iterate", op="aa_maybe_iterate")
        add_node(node_id=wid("persist"), label="Persist assistant + link run", op="aa_persist_response")
        add_node(node_id=wid("end"), label="End", op=None, terminal=True)

        # linear edges
        add_edge(edge_id=wid("e1"), src=wid("start"), dst=wid("prepare"), pred=None, is_default=True)
        add_edge(edge_id=wid("e2"), src=wid("prepare"), dst=wid("view"), pred=None, is_default=True)
        add_edge(edge_id=wid("e3"), src=wid("view"), dst=wid("retrieve"), pred=None, is_default=True)
        add_edge(edge_id=wid("e4"), src=wid("retrieve"), dst=wid("select"), pred=None, is_default=True)
        add_edge(edge_id=wid("e5"), src=wid("select"), dst=wid("materialize"), pred=None, is_default=True)
        add_edge(edge_id=wid("e6"), src=wid("materialize"), dst=wid("answer"), pred=None, is_default=True)
        add_edge(edge_id=wid("e7"), src=wid("answer"), dst=wid("repair"), pred=None, is_default=True)
        add_edge(edge_id=wid("e8"), src=wid("repair"), dst=wid("eval"), pred=None, is_default=True)
        add_edge(edge_id=wid("e9"), src=wid("eval"), dst=wid("project"), pred=None, is_default=True)
        add_edge(edge_id=wid("e10"), src=wid("project"), dst=wid("iterate"), pred=None, is_default=True)

        # branch: iterate -> retrieve OR persist
        add_edge(edge_id=wid("e11"), src=wid("iterate"), dst=wid("retrieve"), pred="aa_should_iterate", priority=0)
        add_edge(edge_id=wid("e12"), src=wid("iterate"), dst=wid("persist"), pred="always", priority=100, is_default=True)

        add_edge(edge_id=wid("e13"), src=wid("persist"), dst=wid("end"), pred=None, is_default=True)

        return self.validate(workflow_id=workflow_id)