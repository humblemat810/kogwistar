from __future__ import annotations

"""Convert a GraphKnowledgeEngine-style workflow graph into a LangGraph StateGraph.

This module supports two conversion modes:

1) blob_state (default)
   - No singleton '__apply__' node.
   - The whole workflow state lives in a single dict field (default: '__blob__').
   - Step nodes emit *ops* encoded in your existing DSL: ('u'|'a'|'e', {k: v}).
   - A reducer on '__blob__' applies those ops.
   - Fanout introduces stable token ids so reducer order is deterministic.

2) apply_node (legacy)
   - Keeps '__apply__' node and '__updates__' accumulator.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Annotated, Literal, cast
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from graph_knowledge_engine.workflow import design as wf_design
from graph_knowledge_engine.workflow.contract import BasePredicate, WorkflowEdgeInfo


StateUpdate = Tuple[str, Dict[str, Any]]  # ('u'|'a'|'e', {k: v})


@dataclass(frozen=True)
class LGConverterOptions:
    """Options for workflow -> LangGraph conversion."""

    mode: Literal["blob_state", "apply_node"] = "blob_state"

    # blob_state
    blob_key: str = "__blob__"
    blob_ops_key: str = "__ops__"
    token_id_key: str = "__token_id__"
    token_step_key: str = "__token_step__"

    # apply_node (legacy)
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
    """Use existing design loader (GraphKnowledgeEngine-like API)."""
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

    # 2) node-decide via BasePredicate (result.next_step_names)
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


BlobOp = Tuple[str, int, int, str, Dict[str, Any]]  # (token_id, token_step, local_idx, kind, payload)


def _blob_reducer(left: Optional[dict], right: Optional[dict]) -> dict:
    """Reducer for '__blob__' field.

    Nodes emit updates as: {'__ops__': [ ... ]}

    Supported op formats inside '__ops__':
      - (token_id, token_step, local_idx, kind, payload)  # deterministic ordering
      - (kind, payload)                                   # legacy

    Ordering:
      - When tagged ops are present, we sort ops by (token_id, token_step, local_idx)
        before applying. This makes reducer application deterministic even with parallel fanout.
      - Legacy ops preserve arrival order (best effort).
    """
    base: dict = dict(left or {})
    if not right:
        return base

    if isinstance(right, dict) and "__ops__" in right:
        ops_raw = right.get("__ops__") or []
        tagged: List[BlobOp] = []
        legacy: List[StateUpdate] = []

        for item in ops_raw:
            if isinstance(item, tuple) and len(item) == 5:
                token_id, token_step, local_idx, kind, payload = item
                if isinstance(token_id, str) and isinstance(token_step, int) and isinstance(local_idx, int) and kind in ("u", "a", "e") and isinstance(payload, dict):
                    tagged.append((token_id, token_step, local_idx, kind, payload))
                    continue
            if isinstance(item, tuple) and len(item) == 2:
                kind, payload = item
                if kind in ("u", "a", "e") and isinstance(payload, dict):
                    legacy.append((cast(str, kind), payload))

        if tagged:
            tagged.sort(key=lambda t: (t[0], t[1], t[2]))
            _apply_state_update(base, [(k, p) for _, _, _, k, p in tagged])
        if legacy:
            _apply_state_update(base, legacy)
        return base

    # fallback: overwrite/merge
    if isinstance(right, dict):
        base.update(right)
    return base


class LGBlobState(TypedDict, total=False):
    __blob__: Annotated[dict, _blob_reducer]
    __token_id__: str
    __token_step__: int


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

    # Schema for native updates (known keys) -> bucketed DSL ops
    if hasattr(step_resolver, "infer_state_schema_best_effort"):
        # ensures schema contains both explicit + inferred patterns
        try:
            step_resolver.infer_state_schema_best_effort()
        except Exception:
            pass
    schema = step_resolver.describe_state() if hasattr(step_resolver, "describe_state") else {}

    if opt.mode == "apply_node":
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
            op = node.metadata.get("wf_op") if hasattr(node, "metadata") else getattr(node, "op", None)
            fn = step_resolver.resolve(op) if hasattr(step_resolver, "resolve") else step_resolver(op)

            def make_step(nid: str, node_obj: Any, fn_):
                def step_node(state: LGApplyState, config: RunnableConfig | None = None) -> Command:
                    out = fn_(state)
                    if isinstance(out, dict):
                        updates = _delta_to_updates(out, schema)
                        class _R:
                            def __init__(self):
                                self.next_step_names = []
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
                        fanout=bool((node_obj.metadata or {}).get("wf_fanout")) if hasattr(node_obj, "metadata") else bool(getattr(node_obj, "fanout", False)),
                        predicate_registry=predicate_registry,
                    )

                    terminal = bool((node_obj.metadata or {}).get("wf_terminal")) if hasattr(node_obj, "metadata") else bool(getattr(node_obj, "terminal", False))
                    terminal = terminal or len(edges) == 0
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

    for node_id, node in nodes.items():
        op = node.metadata.get("wf_op") if hasattr(node, "metadata") else getattr(node, "op", None)
        fn = step_resolver.resolve(op) if hasattr(step_resolver, "resolve") else step_resolver(op)

        def make_step(nid: str, node_obj: Any, fn_):
            def step_node(state: LGBlobState, config: RunnableConfig | None = None) -> Command:
                blob = cast(dict, state.get(opt.blob_key) or {})
                token_id = cast(str, state.get(opt.token_id_key) or "root")
                token_step = int(state.get(opt.token_step_key) or 0)

                out = fn_(blob)

                if isinstance(out, dict):
                    updates = _delta_to_updates(out, schema)
                    class _R:
                        def __init__(self):
                            self.next_step_names = []
                    result_obj = _R()
                else:
                    upd_native = getattr(out, "update", None) or {}
                    updates = _delta_to_updates(upd_native, schema) + list(getattr(out, "state_update", []) or [])
                    result_obj = out

                edges = list(adj.get(nid, []))
                next_nodes = _route_next(
                    edges=edges,
                    state=blob,
                    last_result=result_obj,
                    fanout=bool((node_obj.metadata or {}).get("wf_fanout")) if hasattr(node_obj, "metadata") else bool(getattr(node_obj, "fanout", False)),
                    predicate_registry=predicate_registry,
                )

                terminal = bool((node_obj.metadata or {}).get("wf_terminal")) if hasattr(node_obj, "metadata") else bool(getattr(node_obj, "terminal", False))
                terminal = terminal or len(edges) == 0

                # Tag ops for deterministic reducer ordering
                tagged_ops: List[BlobOp] = [(token_id, token_step, i, kind, payload) for i, (kind, payload) in enumerate(updates)]

                if terminal or not next_nodes:
                    goto = END
                    return Command(
                        goto=goto,
                        update={
                            opt.blob_key: {opt.blob_ops_key: tagged_ops},
                            opt.token_id_key: token_id,
                            opt.token_step_key: token_step + 1,
                        },
                    )

                if len(next_nodes) == 1:
                    goto = next_nodes[0]
                    return Command(
                        goto=goto,
                        update={
                            opt.blob_key: {opt.blob_ops_key: tagged_ops},
                            opt.token_id_key: token_id,
                            opt.token_step_key: token_step + 1,
                        },
                    )

                # fanout: spawn children with stable token ids
                sends: List[Send] = []
                for i, n in enumerate(next_nodes):
                    child_token = f"{token_id}.{i}"
                    sends.append(Send(n, {opt.token_id_key: child_token, opt.token_step_key: token_step + 1}))
                return Command(
                    goto=sends,
                    update={
                        opt.blob_key: {opt.blob_ops_key: tagged_ops},
                        opt.token_id_key: token_id,
                        opt.token_step_key: token_step + 1,
                    },
                )

            return step_node

        sg.add_node(node_id, make_step(node_id, node, fn))

    sg.add_edge(START, start.id)

    # Add static edges only for visualization / compilation sanity.
    for src, edges in adj.items():
        for e in edges:
            info = WorkflowEdgeInfo.from_workflow_edge(e)
            sg.add_edge(src, info.dst)

    # allow END to appear even if routing ends early
    sg.add_edge(start.id, END)

    return sg.compile()
