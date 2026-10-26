"""Workflow design helpers.

   Engine-backed: a minimal spec (workflow_id + start_node_id) that is resolved
   into nodes/edges by reading from a GraphKnowledgeEngine. See
   :mod:`kogwistar.runtime.contract`.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from kogwistar.engine_core.models import Edge as GraphEdge
from kogwistar.engine_core.models import Node as GraphNode
from kogwistar.runtime.models import WorkflowEdge
from kogwistar.typing_interfaces import ReadLike, StrategyEngineLike

from .contract import Predicate
from .models import WorkflowNode
from .resolvers import BaseResolver

PredicateName = str | None

WorkflowResolver = BaseResolver | Callable[[str], object]


def _terminal_reachable_python(
    *,
    start_node_id: str,
    adj: dict[str, list[WorkflowEdge]],
    terminals: set[str],
) -> bool:
    """Independent Python oracle for static terminal reachability."""
    seen: set[str] = set()
    stack = [start_node_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for edge in adj.get(cur, []):
            for target_id in edge.target_ids:
                if target_id is not None:
                    stack.append(str(target_id))
    return any(terminal in seen for terminal in terminals)


def _terminal_reachable(
    *,
    node_ids: list[str],
    start_node_id: str,
    adj: dict[str, list[WorkflowEdge]],
    terminals: set[str],
) -> bool:
    from kogwistar._rust_bridge import (
        runtime_implementation_mode,
        runtime_workflow_terminal_reachable,
    )

    edges = [
        (str(source), str(target))
        for source, workflow_edges in adj.items()
        for edge in workflow_edges
        for target in edge.target_ids
        if target is not None
    ]
    if runtime_implementation_mode() == "rust":
        return bool(
            runtime_workflow_terminal_reachable(
                node_ids=node_ids,
                edges=edges,
                start_node_id=start_node_id,
                terminal_ids=sorted(terminals),
            )
        )
    python_value = _terminal_reachable_python(
        start_node_id=start_node_id,
        adj=adj,
        terminals=terminals,
    )
    selected = runtime_workflow_terminal_reachable(
        node_ids=node_ids,
        edges=edges,
        start_node_id=start_node_id,
        terminal_ids=sorted(terminals),
        python_value=python_value,
    )
    return python_value if selected is None else selected


@dataclass(frozen=True)
class WFNode:
    """In-memory embedding of a workflow node.
    
    Attributes:
        node_id: Canonical graph node ID.
        workflow_id: ID of the workflow this node belongs to.
        op: The operation name (resolved via BaseResolver). Maps to metadata `wf_op`.
        version: Semantic version of the operation. Maps to metadata `wf_version`.
        environment_id: Required execution environment. Maps to `wf_environment_id`.
        start: Whether this is the entry point. Maps to metadata `wf_start`.
        terminal: Whether this is an exit point. Maps to metadata `wf_terminal`.
        fanout: Whether this step supports parallel execution. Maps to metadata `wf_fanout`.
        cacheable: Whether step results can be memoized. Maps to metadata `wf_cacheable`.
    """
    node_id: str
    workflow_id: str
    op: str
    version: str = "v1"
    start: bool = False
    terminal: bool = False
    fanout: bool = False
    cacheable: bool = True
    environment_id: str | None = None


@dataclass(frozen=True)
class WFEdge:
    """In-memory embedding of a workflow transition.
    
    Attributes:
        edge_id: Canonical graph edge ID.
        workflow_id: ID of the workflow this edge belongs to.
        src: Source node ID.
        dst: Destination node ID.
        predicate: Optional name of a transition guard. Maps to metadata `wf_predicate`.
        priority: Evaluation order for branching. Maps to metadata `wf_priority`.
        is_default: Fallback path if no predicates match. Maps to metadata `wf_is_default`.
        multiplicity: 'one' or 'many' (for fan-out). Maps to metadata `wf_multiplicity`.
    """
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
    nodes: dict[str, WFNode]
    out_edges: dict[str, list[WFEdge]]


def _engine_get_nodes(
    workflow_engine: object,
    *,
    where: object,
    limit: int,
    node_type: type[GraphNode] | None = None,
) -> list[GraphNode]:
    reader = cast(ReadLike, getattr(workflow_engine, "read", workflow_engine))
    try:
        return list(reader.get_nodes(where=where, limit=limit, node_type=node_type))
    except TypeError:
        return list(reader.get_nodes(where=where, limit=limit))


def _engine_get_edges(
    workflow_engine: object,
    *,
    where: object,
    limit: int,
    edge_type: type[GraphEdge] | None = None,
) -> list[GraphEdge]:
    reader = cast(ReadLike, getattr(workflow_engine, "read", workflow_engine))
    try:
        return list(reader.get_edges(where=where, limit=limit, edge_type=edge_type))
    except TypeError:
        return list(reader.get_edges(where=where, limit=limit))


def build_workflow_from_engine(
    *, workflow_engine: object, workflow_id: str
) -> WorkflowSpec:
    """Load a rich :class:`WorkflowSpec` from a GraphKnowledgeEngine-like API.

    Required engine interface:
      - get_nodes(where=..., limit=...)
      - get_edges(where=..., limit=...)

    This matches the conventions in :mod:`kogwistar.runtime.contract`:
      node.metadata["entity_type"] == "workflow_node"
      edge.metadata["entity_type"] == "workflow_edge"
    """

    nodes_raw = _engine_get_nodes(
        workflow_engine,
        where={
            "$and": [{"entity_type": "workflow_node"}, {"workflow_id": workflow_id}]
        },
        limit=5000,
    )
    edges_raw = _engine_get_edges(
        workflow_engine,
        where={
            "$and": [{"entity_type": "workflow_edge"}, {"workflow_id": workflow_id}]
        },
        limit=20000,
    )

    nodes: dict[str, WFNode] = {}
    start_node_id: str | None = None

    for n in nodes_raw:
        if n.id is None:
            raise ValueError("workflow nodes must have non-null IDs")
        md = n.metadata or {}
        wn = WFNode(
            node_id=n.id,
            workflow_id=str(md.get("workflow_id") or workflow_id),
            op=str(md.get("wf_op") or md.get("wf.op") or ""),
            version=str(md.get("wf_version") or md.get("wf.version") or "v1"),
            environment_id=(
                str(md["wf_environment_id"])
                if md.get("wf_environment_id") is not None
                else None
            ),
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

    out_edges: dict[str, list[WFEdge]] = {nid: [] for nid in nodes.keys()}

    for e in edges_raw:
        if e.id is None:
            raise ValueError("workflow edges must have non-null IDs")
        md = e.metadata or {}
        src = e.source_ids[0] if getattr(e, "source_ids", None) else md.get("src")
        dst = e.target_ids[0] if getattr(e, "target_ids", None) else md.get("dst")
        if src not in nodes or dst not in nodes:
            raise ValueError(
                f"workflow edge endpoints not workflow nodes: {src!r} -> {dst!r}"
            )

        predicate_value = md.get("wf_predicate")
        priority_value = md.get("wf_priority", 100)
        we = WFEdge(
            edge_id=e.id,
            workflow_id=str(md.get("workflow_id") or workflow_id),
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
        out_edges[str(src)].append(we)

    for s in out_edges:
        out_edges[s].sort(key=lambda x: x.priority)

    return WorkflowSpec(
        workflow_id=workflow_id,
        start_node_id=start_node_id,
        nodes=nodes,
        out_edges=out_edges,
    )


def load_workflow_design(
    *, workflow_engine: object, workflow_id: str
) -> tuple[
    WorkflowNode,
    dict[str, WorkflowNode],
    dict[str, list[WorkflowEdge]],
    dict[str, list[WorkflowEdge]],
]:
    """
    Load workflow graph design from workflow_engine.
    Nodes/edges must be tagged with:
      node.metadata.entity_type="workflow_node"
      edge.metadata.entity_type="workflow_edge"
    """
    nodes_raw = cast(
        list[WorkflowNode],
        _engine_get_nodes(
            workflow_engine,
            where={
                "$and": [
                    {"entity_type": "workflow_node"},
                    {"workflow_id": workflow_id},
                ]
            },
            limit=5000,
            node_type=WorkflowNode,
        ),
    )
    edges_raw = cast(
        list[WorkflowEdge],
        _engine_get_edges(
            workflow_engine,
            where={
                "$and": [
                    {"entity_type": "workflow_edge"},
                    {"workflow_id": workflow_id},
                ]
            },
            limit=20000,
            edge_type=WorkflowEdge,
        ),
    )

    nodes: dict[str, WorkflowNode] = {}
    start_nodes: list[WorkflowNode] = []
    for n in nodes_raw:
        node_id = n.id
        if node_id is None:
            raise ValueError("workflow nodes must have non-null IDs")
        nodes[node_id] = n
        if n.metadata.get("wf_start"):
            start_nodes.append(n)

    if len(start_nodes) != 1:
        raise ValueError(
            f"workflow_id={workflow_id!r} must have exactly one start node (wf_start=True). Found {len(start_nodes)}"
        )

    adj: dict[str, list[WorkflowEdge]] = {nid: [] for nid in nodes}
    rev_adj: dict[str, list[WorkflowEdge]] = {nid: [] for nid in nodes}
    for e in edges_raw:
        src = e.source_ids[0]
        dst = e.target_ids[0]
        if src not in nodes or dst not in nodes:
            raise ValueError(
                f"Workflow edge {e.id} connects non-workflow nodes: {src}->{dst}"
            )

        adj[src].append(e)
        rev_adj[dst].append(e)

    def get_node_priority(n: WorkflowEdge) -> int:
        value = n.metadata.get("wf_priority")
        return int(value) if isinstance(value, (int, float, str)) else 100

    for src in adj:
        adj[src].sort(key=get_node_priority)

    return start_nodes[0], nodes, adj, rev_adj


def validate_workflow_design(
    *,
    workflow_engine: object,
    workflow_id: str,
    predicate_registry: dict[str, Predicate],
    resolver: WorkflowResolver | None = None,
) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
    """Validate an engine-backed workflow design.

    Supports both legacy metadata keys (predicate/terminal/start/priority)
    and the newer wf_* schema (wf_predicate/wf_terminal/wf_start/wf_priority).
    """

    start, nodes, adj, rev_adj = load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )

    def _node_op(node: WorkflowNode) -> str:
        op = node.metadata.get("wf_op") or node.metadata.get("op")
        return str(op or "")

    # Resolver-based op validation is optional.
    # - If resolver exposes `.ops`, validate declared workflow ops against it.
    # - If resolver is only a callable (e.g. def resolve_step(op): ...), skip `.ops` validation.
    if resolver is not None:
        def uses_python_worker(node: WorkflowNode) -> bool:
            target = (getattr(node, "metadata", None) or {}).get(
                "wf_executor_target"
            )
            op = _node_op(node)
            if target not in {None, "python-worker", "rust-native"}:
                raise ValueError(f"workflow node has unsupported executor target {target!r}")
            if target == "python-worker" and op.startswith("rust-native/"):
                raise ValueError("rust-native operation names cannot target the Python worker")
            if target == "rust-native":
                return False
            return not (target is None and op.startswith("rust-native/"))

        python_nodes = [node for node in nodes.values() if uses_python_worker(node)]
        resolver_ops = getattr(resolver, "ops", None)
        if resolver_ops is not None:
            if unknown_ops := (
                set(_node_op(node) for node in python_nodes) - set(resolver_ops)
            ):
                raise Exception(f"workflow_contains unregistered ops {unknown_ops}")
        elif not callable(resolver):
            raise Exception("resolver must be callable or expose an iterable `.ops`.")

        supports_version = getattr(resolver, "supports_version", None)
        resolve_version = getattr(resolver, "resolve_version", None)
        resolve_async_version = getattr(resolver, "resolve_async_version", None)
        can_resolve_versions = callable(supports_version) or callable(
            resolve_version
        ) or callable(resolve_async_version)
        for node in python_nodes:
            op = _node_op(node)
            if not op:
                continue
            version = str(
                node.metadata.get("wf_version")
                or node.metadata.get("wf.version")
                or "v1"
            )
            if callable(supports_version):
                supported = bool(supports_version(op, version))
            elif version == "v1":
                # Legacy callable resolvers remain valid for the default contract.
                supported = True
            elif can_resolve_versions:
                try:
                    if callable(resolve_version):
                        versioned_handler = resolve_version(op, version)
                    else:
                        versioned_handler = resolve_async_version(op, version)
                    supported = callable(versioned_handler)
                except (LookupError, ValueError):
                    supported = False
            else:
                supported = False
            if not supported:
                raise ValueError(
                    f"workflow op {op!r} requires unsupported version {version!r}"
                )
            environment_id = node.metadata.get("wf_environment_id")
            if environment_id is not None:
                if (
                    not isinstance(environment_id, str)
                    or not environment_id.strip()
                    or environment_id != environment_id.strip()
                ):
                    raise ValueError(
                        f"workflow op {op!r} has non-canonical wf_environment_id"
                    )
                environment_for_version = getattr(
                    resolver, "environment_for_version", None
                )
                registered_environment = (
                    environment_for_version(op, version)
                    if callable(environment_for_version)
                    else None
                )
                if registered_environment != environment_id:
                    raise ValueError(
                        f"workflow op {op!r} requires environment "
                        f"{environment_id!r}, but resolver registers "
                        f"{registered_environment!r}"
                    )
    # predicate resolution
    for edges in adj.values():
        for e in edges:
            pred_value = e.metadata.get("wf_predicate", e.metadata.get("predicate"))
            pred = pred_value if isinstance(pred_value, str) else None
            if pred is not None and pred not in predicate_registry:
                raise ValueError(f"Unknown predicate {pred!r} on workflow edge {e.id}")

    # must have at least one terminal reachable ignoring predicates
    terminals = {
        nid
        for nid, n in nodes.items()
        if (
            n.metadata.get("wf_terminal", n.metadata.get("terminal"))
            or len(adj.get(nid, [])) == 0
        )
    }
    if not terminals:
        raise ValueError("Workflow has no terminal nodes and no sink nodes")

    if not _terminal_reachable(
        node_ids=[str(node_id) for node_id in nodes],
        start_node_id=str(start.id),
        adj=adj,
        terminals=terminals,
    ):
        raise ValueError("No terminal reachable from start (ignoring predicates).")
    return start, nodes, adj


class BaseWorkflowDesigner:
    """Base helper for building/validating workflow designs.

    This is intentionally lightweight: it does not assume any conversation-specific
    invariants. Subclasses can add domain constraints.
    """

    def __init__(
        self,
        *,
        workflow_engine: StrategyEngineLike,
        predicate_registry: dict[str, Predicate],
        resolver: BaseResolver | None = None,
    ) -> None:
        self.workflow_engine = workflow_engine
        self.predicate_registry = predicate_registry
        self.resolver = resolver

    def validate(
        self, *, workflow_id: str, resolver: WorkflowResolver | None = None
    ) -> tuple[WorkflowNode, dict[str, WorkflowNode], dict[str, list[WorkflowEdge]]]:
        return validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
            resolver=resolver or self.resolver,
        )
