from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Annotated, Mapping, Literal, cast
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from graph_knowledge_engine.workflow import design as wf_design
from graph_knowledge_engine.workflow.contract import BasePredicate, WorkflowEdgeInfo


StateUpdate = Tuple[str, Dict[str, Any]]  # ('u'|'a'|'e', {k: v})


@dataclass(frozen=True)
class LGConverterOptions:
    """Options for workflow -> LangGraph conversion.

    mode:
      - 'blob_state': keep the whole workflow state inside a single dict field '__blob__'.
        Step nodes emit *deltas* encoded as state_update DSL ('u'/'a'/'e'), and the
        reducer for '__blob__' applies those deltas (so no '__apply__' node exists).

      - 'apply_node': legacy mode that accumulates '__updates__' and uses a singleton
        '__apply__' node to apply them.
    """

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


def _concat_updates(left: Optional[List[StateUpdate]], right: Optional[List[StateUpdate]]) -> List[StateUpdate]:
    return list(left or []) + list(right or [])


def _delta_to_updates(delta: Mapping[str, Any], schema: Mapping[str, str] | None) -> List[StateUpdate]:
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
) -> tuple[Any, Dict[str, Any], Dict[str, List[Any]]]:
    return wf_design.load_workflow_design(workflow_engine=workflow_engine, workflow_id=workflow_id)


def _route_next(
    *,
    edges: List[Any],
    state: dict,
    last_result: Any,
    fanout: bool,
    predicate_registry: Dict[str, BasePredicate],
) -> List[str]:
    matched: List[str] = []

    # 1) predicate edges
    for e in edges:
        info = WorkflowEdgeInfo.from_workflow_edge(e)
        if info.predicate is None:
            continue
        pred = predicate_registry.get(info.predicate)
        if pred is None:
            continue
        ok = False
        try:
            ok = bool(pred(info, state, last_result))
        except Exception:
            ok = False
        if ok:
            matched.append(info.dst)
            if not fanout and info.multiplicity != "many":
                return matched

    if matched and (fanout or any(WorkflowEdgeInfo.from_workflow_edge(ed).multiplicity == "many" for ed in edges)):
        return matched

    # 2) node-decide via BasePredicate (uses result.next_step_names)
    if not matched:
        node_decider = BasePredicate()
        for e in edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            ok = False
            try:
                ok = bool(node_decider(info, state, last_result))
            except Exception:
                ok = False
            if ok:
                matched.append(info.dst)
                if not fanout and info.multiplicity != "many":
                    return matched

        if matched:
            return matched

        # 3) defaults
        for e in edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            if info.is_default:
                return [info.dst] if not fanout else [info.dst]

    return matched


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
    opt = options or LGConverterOptions()

    start, nodes, adj = _resolve_start_nodes_and_adj(workflow_engine=workflow_engine, workflow_id=workflow_id)
    schema = step_resolver.describe_state() if hasattr(step_resolver, "describe_state") else {}

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
            fn = step_resolver.resolve(op) if hasattr(step_resolver, "resolve") else step_resolver(op)

            def make_step(nid: str, node_obj: Any, fn_):
                def step_node(state: LGApplyState) -> Command:
                    out = fn_(state)
                    updates: List[StateUpdate]
                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)
                        class _R:  # minimal proxy for BasePredicate
                            def __init__(self): self.next_step_names = []
                        result_obj = _R()
                    else:
                        upd_native = getattr(out, "update", None) or {}
                        updates = _delta_to_updates(upd_native, schema) + list(getattr(out, "state_update", []) or [])
                        result_obj = out

                    edges = list(adj.get(nid, []))
                    next_nodes = _route_next(
                        edges=edges,
                        state=state,
                        last_result=result_obj,
                        fanout=bool(getattr(node_obj, "fanout", False)),
                        predicate_registry=predicate_registry,
                    )

                    terminal = bool(getattr(node_obj, "terminal", False)) or len(edges) == 0
                    if terminal or not next_nodes:
                        goto = END
                    elif len(next_nodes) == 1:
                        goto = next_nodes[0]
                    else:
                        goto = [Send(n, {}) for n in next_nodes]

                    state["__goto__"] = goto
                    return Command(goto=opt.apply_node_id, update={opt.updates_key: updates})

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

    
    
    # --- blob_state mode (default) ---
    sg = StateGraph(LGBlobState)

    def _make_token_id(parent: str, idx: int) -> str:
        return f"{parent}.{idx}"

    def _needs_send_payload(nid: str, node_obj: Any) -> bool:
        # Fanout nodes (or any outgoing edge with multiplicity='many') require Send payloads
        if bool(getattr(node_obj, "fanout", False)):
            return True
        for e in adj.get(nid, []) or []:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            if info.multiplicity == "many":
                return True
        return False

    for node_id, node in nodes.items():
        op = node.op
        fn = step_resolver.resolve(op) if hasattr(step_resolver, "resolve") else step_resolver(op)
        use_send = _needs_send_payload(node_id, node)

        if use_send:
            # --- Fanout-capable node: routing via Command(goto=Send/...) to preserve token semantics ---
            def make_step_send(nid: str, node_obj: Any, fn_):
                def step_node(state: LGBlobState) -> Command:
                    blob = cast(dict, state.get(opt.blob_key) or {})
                    token_id = cast(str, blob.get("__token_id__", "root"))

                    out = fn_(blob)

                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)
                        class _R:
                            def __init__(self): self.next_step_names = []
                        result_obj = _R()
                    else:
                        upd_native = getattr(out, "update", None) or {}
                        updates = _delta_to_updates(upd_native, schema) + list(getattr(out, "state_update", []) or [])
                        result_obj = out

                    # persist next_step_names for routing fallbacks (BasePredicate)
                    ns = list(getattr(result_obj, "next_step_names", []) or [])
                    if ns:
                        updates = list(updates) + [("u", {"__next_step_names__": ns})]

                    edges = list(adj.get(nid, []))
                    next_nodes = _route_next(
                        edges=edges,
                        state=blob,
                        last_result=result_obj,
                        fanout=bool(getattr(node_obj, "fanout", False)),
                        predicate_registry=predicate_registry,
                    )

                    terminal = bool(getattr(node_obj, "terminal", False)) or len(edges) == 0
                    if terminal or not next_nodes:
                        goto = END
                    elif len(next_nodes) == 1:
                        goto = next_nodes[0]
                    else:
                        sends: list[Send] = []
                        for i, n in enumerate(next_nodes):
                            child_tid = _make_token_id(token_id, i)
                            # put token id into blob via ops, not as a top-level key
                            child_state = {opt.blob_key: {opt.blob_ops_key: [("u", {"__token_id__": child_tid})]}}
                            sends.append(Send(n, child_state))
                        goto = sends

                    return Command(goto=goto, update={opt.blob_key: {opt.blob_ops_key: updates}})

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
                        def __init__(self): self.next_step_names = []
                    result_obj = _R()
                else:
                    upd_native = getattr(out, "update", None) or {}
                    updates = _delta_to_updates(upd_native, schema) + list(getattr(out, "state_update", []) or [])
                    result_obj = out

                # persist next_step_names for routing fallbacks (BasePredicate)
                ns = list(getattr(result_obj, "next_step_names", []) or [])
                if ns:
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
                proxy = _LastResultProxy(list(blob.get("__next_step_names__", []) or []))

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
                    raise RuntimeError(f"No eligible outgoing edge from {nid!r} (not terminal)")
                # Exclusive choice: pick the first (route_next already respects priority/default)
                return next_nodes[0]

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
