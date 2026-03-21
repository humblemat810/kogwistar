"""LangGraph conversion helpers for workflow graphs.

This module maps the engine's workflow model onto LangGraph in two modes. "visual"
optimizes for readable diagrams and intentionally drops some token and join fidelity,
while "semantics" preserves more of the runtime's fanout and join behavior via
Send/Command plus blob-state bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Annotated,
    Mapping,
    Literal,
    NamedTuple,
    cast,
)
from typing_extensions import TypedDict

from kogwistar.runtime import design as wf_design
from kogwistar.runtime.contract import BasePredicate, WorkflowEdgeInfo

if TYPE_CHECKING:
    from langgraph.graph import StateGraph as LangGraphStateGraph
    from langgraph.types import Command as LangGraphCommand
    from langgraph.types import Send as LangGraphSend


class _LangGraphImports(NamedTuple):
    state_graph: type["LangGraphStateGraph"]
    start: str
    end: str
    command: type["LangGraphCommand"]
    send: type["LangGraphSend"]


def _import_langgraph() -> _LangGraphImports:
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, Send
    except Exception as e:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "LangGraph converter requires optional dependency group 'langgraph'. "
            "Install with: pip install 'kogwistar[langgraph]'"
        ) from e
    return _LangGraphImports(
        state_graph=StateGraph,
        start=START,
        end=END,
        command=Command,
        send=Send,
    )


StateUpdate = Tuple[str, Dict[str, Any]]  # ('u'|'a'|'e', {k: v})


@dataclass(frozen=True)
class LGConverterOptions:
    """Options for workflow -> LangGraph conversion.

    execution:
      - 'visual' (default): produce a clean, easy-to-read LangGraph diagram.
        * Fanout is modeled using normal edges so LangGraph runs all eligible downstreams.
        * Token-id / Send-based semantics are intentionally NOT modeled (best-effort).
        * Joins are NOT modeled.

      - 'semantics': preserve more of the engine's execution semantics.
        * Fanout / multiplicity and join gating may use Command+Send (token semantics).
        * Diagram may be less "pretty".

    mode:
      - 'blob_state': keep the whole workflow state inside a single dict field '__blob__'.
        Step nodes emit *deltas* encoded as state_update DSL ('u'/'a'/'e'), and the
        reducer for '__blob__' applies those deltas (so no '__apply__' node exists).

      - 'apply_node': legacy mode that accumulates '__updates__' and uses a singleton
        '__apply__' node to apply them.
    """

    execution: Literal["visual", "semantics"] = "visual"
    mode: Literal["blob_state", "apply_node"] = "blob_state"

    # blob_state mode
    blob_key: str = "__blob__"
    blob_ops_key: str = "__ops__"  # reserved key inside blob update wrapper

    # apply_node mode (legacy)
    updates_key: str = "__updates__"
    apply_node_id: str = "__apply__"


def _apply_state_update(mute_state: dict, state_update: Sequence[StateUpdate]) -> None:
    """Match WorkflowRuntime.apply_state_update semantics."""
    for kind, payload in state_update:
        if kind == "a":
            for k, v in payload.items():
                mute_state.setdefault(k, []).append(v)
        elif kind == "e":
            for k, v in payload.items():
                mute_state.setdefault(k, []).extend(v)
        elif kind == "u":
            for k, v in payload.items():
                mute_state[k] = v
        else:
            raise ValueError(f"Unknown state update kind: {kind!r}")


def _concat_updates(
    left: Optional[List[StateUpdate]], right: Optional[List[StateUpdate]]
) -> List[StateUpdate]:
    return list(left or []) + list(right or [])


def _delta_to_updates(
    delta: Mapping[str, Any], schema: Mapping[str, str] | None
) -> List[StateUpdate]:
    """Convert a native update dict into ('u'/'a'/'e') updates using schema."""
    sch = dict(schema or {})
    buckets: dict[str, dict[str, Any]] = {"u": {}, "a": {}, "e": {}}
    for k, v in delta.items():
        mode = sch.get(str(k), "u")
        if mode not in buckets:
            mode = "u"
        buckets[mode][str(k)] = v
    out: List[StateUpdate] = []
    for mode in ("u", "a", "e"):
        if buckets[mode]:
            out.append((mode, buckets[mode]))
    return out


def _resolve_start_nodes_and_adj(
    *,
    workflow_engine: Any,
    workflow_id: str,
) -> tuple[Any, Dict[str, Any], Dict[str, List[Any]], Dict[str, List[Any]]]:
    return wf_design.load_workflow_design(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )


def _route_next(
    *,
    edges: List[Any],
    state: dict,
    last_result: Any,
    fanout: bool,
    predicate_registry: Dict[str, BasePredicate],
) -> List[str]:
    """Choose next node ids from outgoing edges.

    Semantics:
    - If last_result.next_step_names is provided (non-empty), treat it as an explicit routing decision
      (duplicates preserved) filtered to valid outgoing destinations.
    - Otherwise, evaluate predicate edges, then BasePredicate (which also consults next_step_names),
      then defaults.
    - Priority: lower 'wf_priority' wins. If multiple eligible edges share the best priority,
      we *fan out* (return all best-priority destinations).
    - For fanout nodes or edges with multiplicity=='many', multiple destinations may be returned.
    """
    # Explicit routing override (duplicates preserved)
    ns = list(getattr(last_result, "next_step_names", []) or [])
    if ns:
        return ns
        # valid = {WorkflowEdgeInfo.from_workflow_edge(e).dst for e in edges}
        # return [n for n in ns if n in valid]

    candidates: list[tuple[int, str, str]] = []  # (priority, dst, multiplicity)

    def _add_candidate(info: WorkflowEdgeInfo):
        pr = int(info.priority or 0)
        candidates.append((pr, info.dst, info.multiplicity or "one"))

    # 1) predicate edges
    for e in edges:
        info = WorkflowEdgeInfo.from_workflow_edge(e)
        if info.predicate is None:
            continue
        pred = predicate_registry.get(info.predicate)
        if pred is None:
            continue
        try:
            ok = bool(pred(info, state, last_result))
        except Exception:
            ok = False
        if ok:
            _add_candidate(info)

    # 2) node-decide via BasePredicate (result.next_step_names empty => always true)
    if not candidates:
        node_decider = BasePredicate()
        for e in edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            try:
                ok = bool(node_decider(info, state, last_result))
            except Exception:
                ok = False
            if ok:
                _add_candidate(info)

    # 3) defaults (only if still nothing)
    if not candidates:
        for e in edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            if info.is_default:
                _add_candidate(info)

    if not candidates:
        return []

    # Sort by priority then dst for stability
    candidates.sort(key=lambda t: (t[0], t[1]))

    allow_many = bool(fanout) or any(m == "many" for _, __, m in candidates)

    best_prio = candidates[0][0]
    best = [dst for pr, dst, _m in candidates if pr == best_prio]

    # If fanout/many: return all best-priority destinations (may be >1)
    if allow_many:
        return best

    # # Exclusive choice but tie on best priority => fanout (policy)
    # this will create problem to the undeterminate shape
    # if len(best) > 1:
    #     return best

    return [best[0]]


# ----------------------------

# blob_state mode
# ----------------------------


def _blob_reducer(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for '__blob__' field.

    Nodes emit updates as: {'__ops__': [('u'|'a'|'e', {...}), ...]}
    The reducer applies ops to the accumulated blob dict.

    If 'right' doesn't contain '__ops__', it is treated as a plain dict overwrite (last-write-wins).
    """
    base: dict = dict(left or {})
    if not right:
        return base
    if isinstance(right, dict) and "__ops__" in right:
        ops = right.get("__ops__") or []
        if ops:
            _apply_state_update(base, cast(Sequence[StateUpdate], ops))
        return base
    # fallback: overwrite/merge
    if isinstance(right, dict):
        base.update(right)
    return base


class LGBlobState(TypedDict, total=False):
    __blob__: Annotated[dict, _blob_reducer]


# ----------------------------
# apply_node mode (legacy)
# ----------------------------


class LGApplyState(TypedDict, total=False):
    __updates__: Annotated[List[StateUpdate], _concat_updates]
    __goto__: Any


def to_langgraph(
    *,
    workflow_engine: Any,
    workflow_id: str,
    step_resolver: Any,
    predicate_registry: Dict[str, BasePredicate],
    options: Optional[LGConverterOptions] = None,
):
    """Compile one workflow into LangGraph using a diagram-first or semantics-first mapping.

    visual/blob_state favors a clean graph and best-effort routing, intentionally
    degrading token propagation and join barriers. semantics/blob_state models
    fanout, join arrivals, and child token ids more faithfully with Send/Command.
    apply_node is kept for legacy compatibility and routes updates through a
    singleton apply step.
    """
    opt = options or LGConverterOptions()
    langgraph = _import_langgraph()
    StateGraph = langgraph.state_graph
    START = langgraph.start
    END = langgraph.end
    Command = langgraph.command
    Send = langgraph.send

    start, nodes, adj, rev_adj = _resolve_start_nodes_and_adj(
        workflow_engine=workflow_engine, workflow_id=workflow_id
    )
    schema = (
        step_resolver.describe_state()
        if hasattr(step_resolver, "describe_state")
        else {}
    )

    if opt.mode == "apply_node":
        # --- legacy implementation (kept for compatibility) ---
        sg = StateGraph(LGApplyState)

        def apply_node(state: LGApplyState) -> Command:
            pending = state.get(opt.updates_key, []) or []
            if pending:
                _apply_state_update(state, pending)
            state[opt.updates_key] = []
            goto = state.pop("__goto__", None) or END
            return Command(goto=goto)

        sg.add_node(opt.apply_node_id, apply_node)

        for node_id, node in nodes.items():
            op = node.op
            fn = (
                step_resolver.resolve(op)
                if hasattr(step_resolver, "resolve")
                else step_resolver(op)
            )

            def make_step(nid: str, node_obj: Any, fn_):
                def step_node(state: LGApplyState) -> Command:
                    out = fn_(state)
                    updates: List[StateUpdate]
                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)

                        class _R:  # minimal proxy for BasePredicate
                            def __init__(self):
                                self.next_step_names = []

                        result_obj = _R()
                    else:
                        upd_native = getattr(out, "update", None) or {}
                        updates = _delta_to_updates(upd_native, schema) + list(
                            getattr(out, "state_update", []) or []
                        )
                        result_obj = out

                    edges = list(adj.get(nid, []))  # + list(rev_adj.get(nid, []))
                    next_nodes = _route_next(
                        edges=edges,
                        state=state,
                        last_result=result_obj,
                        fanout=bool(getattr(node_obj, "fanout", False)),
                        predicate_registry=predicate_registry,
                    )

                    terminal = (
                        bool(getattr(node_obj, "terminal", False)) or len(edges) == 0
                    )
                    if terminal or not next_nodes:
                        goto = END
                    elif len(next_nodes) == 1:
                        goto = next_nodes[0]
                    else:
                        goto = [Send(n, {}) for n in next_nodes]

                    state["__goto__"] = goto
                    return Command(
                        goto=opt.apply_node_id, update={opt.updates_key: updates}
                    )

                return step_node

            sg.add_node(node_id, make_step(node_id, node, fn))

        sg.add_edge(START, start.id)
        for src, edges in adj.items():
            sg.add_edge(src, opt.apply_node_id)
            for e in edges:
                info = WorkflowEdgeInfo.from_workflow_edge(e)
                sg.add_edge(opt.apply_node_id, info.dst)
        sg.add_edge(opt.apply_node_id, END)
        return sg.compile()

        # --- blob_state mode ---

    # visual execution: prefer clean diagram, drop token semantics / Send / joins.
    if opt.execution == "visual":
        sg = StateGraph(LGBlobState)

        def _init_blob(state: dict) -> dict:
            blob = dict(state.get(opt.blob_key) or {})
            return {opt.blob_key: blob}

        sg.add_node("__init_blob__", _init_blob)
        sg.add_edge(START, "__init_blob__")
        sg.add_edge("__init_blob__", start.id)

        # Helper: whether a node should be treated as fanout in visual mode.
        def _is_visual_fanout(nid: str, node_obj: Any) -> bool:
            if bool(getattr(node_obj, "fanout", False)):
                return True
            for e in list(adj.get(nid, []) or []):
                info = WorkflowEdgeInfo.from_workflow_edge(e)
                if (info.multiplicity or "one") == "many":
                    return True
            return False

        # Add step nodes
        for node_id, node in nodes.items():
            op = node.op
            fn = (
                step_resolver.resolve(op)
                if hasattr(step_resolver, "resolve")
                else step_resolver(op)
            )

            def make_step_update(nid: str, node_obj: Any, fn_):
                def step_node(state: LGBlobState) -> dict:
                    blob = cast(dict, state.get(opt.blob_key) or {})
                    out = fn_(blob)

                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)

                        class _R:
                            def __init__(self):
                                self.next_step_names = []

                        result_obj = _R()
                    else:
                        upd_native = getattr(out, "update", None) or {}
                        updates = _delta_to_updates(upd_native, schema) + list(
                            getattr(out, "state_update", []) or []
                        )
                        result_obj = out

                    # Persist next_step_names for predicate fallbacks (BasePredicate). Always clear/set.
                    ns = list(getattr(result_obj, "next_step_names", []) or [])
                    updates = list(updates) + [("u", {"__next_step_names__": ns})]

                    return {opt.blob_key: {opt.blob_ops_key: updates}}

                return step_node

            sg.add_node(node_id, make_step_update(node_id, node, fn))

        # Wire graph edges:
        # - fanout nodes: connect directly to all outgoing destinations (LangGraph will schedule all).
        # - non-fanout nodes: conditional router to choose one destination (predicate/default/priority).
        for src, out_edges in adj.items():
            node_obj = nodes[src]
            edges = list(out_edges or [])

            # terminal nodes don't need outgoing
            if bool(getattr(node_obj, "terminal", False)) or not edges:
                continue

            if _is_visual_fanout(src, node_obj):
                for e in edges:
                    info = WorkflowEdgeInfo.from_workflow_edge(e)
                    sg.add_edge(src, info.dst)
            else:
                # Router returns a single destination (exclusive choice).
                def make_router(nid: str, node_obj: Any, edges: list[Any]):
                    possible = {
                        WorkflowEdgeInfo.from_workflow_edge(e).dst for e in edges
                    }
                    path_map = {d: d for d in sorted(possible)}
                    # if node can really terminate, allow END
                    if bool(getattr(node_obj, "terminal", False)):
                        path_map[END] = END

                    class _LastResultProxy:
                        def __init__(self, next_step_names: list[str]):
                            self.next_step_names = next_step_names

                    def router(state: LGBlobState):
                        blob = cast(dict, state.get(opt.blob_key) or {})
                        proxy = _LastResultProxy(
                            list(blob.get("__next_step_names__", []) or [])
                        )
                        nxt = _route_next(
                            edges=edges,
                            state=blob,
                            last_result=proxy,
                            fanout=False,
                            predicate_registry=predicate_registry,
                        )
                        if not nxt:
                            raise RuntimeError(
                                f"No eligible outgoing edge from {nid!r} (not terminal)"
                            )
                        return nxt[0]

                    return router, path_map

                router_fn, path_map = make_router(src, node_obj, edges)
                sg.add_conditional_edges(src, router_fn, path_map)

        return sg.compile()

    # semantics execution (uses Send/token/join modelling)

    sg = StateGraph(LGBlobState)

    # Join nodes are marked via node.metadata['wf_join'].
    join_nodes: set[str] = {
        nid
        for nid, n in nodes.items()
        if bool((n.metadata or getattr(n, "metadata", {}) or {}).get("wf_join", False))
    }

    # For each join node, track the set of immediate upstream node ids that must arrive.
    join_required: dict[str, set[str]] = {jid: set() for jid in join_nodes}
    for src, out_edges in adj.items():
        for e in out_edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            if info.dst in join_required:
                join_required[info.dst].add(src)

    def _make_token_id(parent: str, idx: int) -> str:
        return f"{parent}.{idx}"

    def _needs_send_payload(nid: str, node_obj: Any) -> bool:
        """Return True if this node must route using Send/Command (fanout semantics).

        We use Send-mode if:
        - the node is explicitly fanout
        - the node is a join (needs barrier gating / may get multiple arrivals)
        - any outgoing edge has multiplicity='many'
        - any outgoing edge targets a join node (so we can record arrival side-effects)
        - there is a priority tie for the best eligible edges (policy: tie => fanout)
        """
        md = getattr(node_obj, "metadata", {}) or {}
        if bool(getattr(node_obj, "fanout", False)):
            return True
        if bool(md.get("wf_join", False)):
            return True

        out_edges = list(adj.get(nid, []) or [])
        if not out_edges:
            return False

        # edge multiplicity many or dst is join => send-mode
        prios: list[int] = []
        for e in out_edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            if info.multiplicity == "many":
                return True
            if info.dst in join_nodes:
                return True
            prios.append(int(info.priority or 0))

        # best-priority tie => fanout => needs Send-mode to preserve token semantics
        if prios:
            best = min(prios)
            if sum(1 for p in prios if p == best) > 1:
                return True

        return False

    for node_id, node in nodes.items():
        op = node.op
        fn = (
            step_resolver.resolve(op)
            if hasattr(step_resolver, "resolve")
            else step_resolver(op)
        )
        use_send = _needs_send_payload(node_id, node)

        if use_send:
            # --- Fanout-capable node: routing via Command(goto=Send/...) to preserve token semantics ---
            def make_step_send(nid: str, node_obj: Any, fn_):
                def step_node(state: LGBlobState) -> Command:
                    blob = cast(dict, state.get(opt.blob_key) or {})
                    token_id = cast(str, blob.get("__token_id__", "root"))

                    md = getattr(node_obj, "metadata", {}) or {}
                    if bool(md.get("wf_join", False)):
                        req = join_required.get(nid, set())
                        done_key = f"__join_done__:{nid}"
                        arr_key = f"__join_arrivals__:{nid}"
                        if bool(blob.get(done_key, False)):
                            # join already satisfied; ignore repeated arrivals
                            return Command(
                                goto=END, update={opt.blob_key: {opt.blob_ops_key: []}}
                            )
                        arrived = set(cast(list, blob.get(arr_key, []) or []))
                        if req and not req.issubset(arrived):
                            # barrier not satisfied yet; stop this activation
                            return Command(
                                goto=END, update={opt.blob_key: {opt.blob_ops_key: []}}
                            )

                    out = fn_(blob)

                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)

                        class _R:
                            def __init__(self):
                                self.next_step_names = []

                        result_obj = _R()
                    else:
                        upd_native = getattr(out, "update", None) or {}
                        updates = _delta_to_updates(upd_native, schema) + list(
                            getattr(out, "state_update", []) or []
                        )
                        result_obj = out

                    # persist next_step_names for routing fallbacks (BasePredicate)
                    ns = list(getattr(result_obj, "next_step_names", []) or [])
                    updates = list(updates) + [("u", {"__next_step_names__": ns})]

                    md = getattr(node_obj, "metadata", {}) or {}
                    if bool(md.get("wf_join", False)):
                        # Mark join as done and clear arrivals once it actually executes.
                        updates = list(updates) + [
                            ("u", {f"__join_done__:{nid}": True}),
                            ("u", {f"__join_arrivals__:{nid}": []}),
                        ]

                    edges = list(adj.get(nid, []))
                    next_nodes = _route_next(
                        edges=edges,
                        state=blob,
                        last_result=result_obj,
                        fanout=bool(getattr(node_obj, "fanout", False)),
                        predicate_registry=predicate_registry,
                    )

                    # Record join arrivals as side-effects when routing into join nodes.
                    for dst in next_nodes:
                        if dst in join_nodes:
                            updates = list(updates) + [
                                ("a", {f"__join_arrivals__:{dst}": nid})
                            ]

                    terminal = (
                        bool(getattr(node_obj, "terminal", False)) or len(edges) == 0
                    )
                    if terminal or not next_nodes:
                        goto = END
                    elif len(next_nodes) == 1:
                        goto = next_nodes[0]
                    else:
                        sends: list[Send] = []
                        for i, n in enumerate(next_nodes):
                            child_tid = _make_token_id(token_id, i)
                            # put token id into blob via ops, not as a top-level key
                            child_state = {
                                opt.blob_key: {
                                    opt.blob_ops_key: [
                                        ("u", {"__token_id__": child_tid})
                                    ]
                                }
                            }
                            sends.append(Send(n, child_state))
                        goto = sends

                    return Command(
                        goto=goto, update={opt.blob_key: {opt.blob_ops_key: updates}}
                    )

                return step_node

            sg.add_node(node_id, make_step_send(node_id, node, fn))
            continue

        # --- Exclusive-choice node: routing via conditional edges (nice diagram + correct semantics) ---
        def make_step_update(nid: str, node_obj: Any, fn_):
            def step_node(state: LGBlobState) -> dict:
                blob = cast(dict, state.get(opt.blob_key) or {})
                out = fn_(blob)

                if isinstance(out, dict):
                    updates = _delta_to_updates(out, schema)

                    class _R:
                        def __init__(self):
                            self.next_step_names = []

                    result_obj = _R()
                else:
                    upd_native = getattr(out, "update", None) or {}
                    updates = _delta_to_updates(upd_native, schema) + list(
                        getattr(out, "state_update", []) or []
                    )
                    result_obj = out

                # persist next_step_names for routing fallbacks (BasePredicate)
                ns = list(getattr(result_obj, "next_step_names", []) or [])

                updates = list(updates) + [("u", {"__next_step_names__": ns})]

                # Emit blob delta as ops for reducer (unknown/dynamic keys supported via DSL ops)
                return {opt.blob_key: {opt.blob_ops_key: updates}}

            return step_node

        sg.add_node(node_id, make_step_update(node_id, node, fn))

        # Conditional router that recomputes next node(s) using the updated blob state.
        def make_router(nid: str, node_obj: Any):
            edges = list(adj.get(nid, []))

            # Precompute possible destinations for the path_map (helps with diagram)
            possible: set[str] = set()
            for e in edges:
                info = WorkflowEdgeInfo.from_workflow_edge(e)
                possible.add(info.dst)

            path_map = {d: d for d in sorted(possible)}

            # Only include END as an explicit transition if this node can really terminate.
            # Otherwise the diagram will imply "everything can end here".
            if bool(getattr(node_obj, "terminal", False)) or len(edges) == 0:
                path_map[END] = END

            class _LastResultProxy:
                def __init__(self, next_step_names: list[str]):
                    self.next_step_names = next_step_names

            def router(state: LGBlobState):
                blob = cast(dict, state.get(opt.blob_key) or {})
                proxy = _LastResultProxy(
                    list(blob.get("__next_step_names__", []) or [])
                )

                next_nodes = _route_next(
                    edges=edges,
                    state=blob,
                    last_result=proxy,
                    fanout=False,
                    predicate_registry=predicate_registry,
                )

                terminal = bool(getattr(node_obj, "terminal", False)) or len(edges) == 0
                if terminal:
                    return END
                if not next_nodes:
                    # No matching edge, and not terminal: this is a design/config error.
                    raise RuntimeError(
                        f"No eligible outgoing edge from {nid!r} (not terminal)"
                    )
                return next_nodes  # filtering/ priority logic goes in _route_next

            return router, path_map

        router_fn, path_map = make_router(node_id, node)
        sg.add_conditional_edges(node_id, router_fn, path_map)

    # Blob init: ensure '__blob__' exists and carries a root token_id.
    # If caller doesn't provide __blob__, start with {'__token_id__': 'root'}.
    def _init_blob(state: dict) -> dict:
        blob = dict(state.get(opt.blob_key) or {})
        blob.setdefault("__token_id__", "root")
        return {opt.blob_key: blob}

    sg.add_node("__init_blob__", _init_blob)
    sg.add_edge(START, "__init_blob__")
    sg.add_edge("__init_blob__", start.id)

    return sg.compile()
