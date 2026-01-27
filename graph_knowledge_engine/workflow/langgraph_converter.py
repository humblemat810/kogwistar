
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from graph_knowledge_engine.workflow import design as wf_design
from graph_knowledge_engine.workflow.contract import BasePredicate, WorkflowEdgeInfo


StateUpdate = Tuple[str, Dict[str, Any]]  # ('u'|'a'|'e', {k: v})


@dataclass(frozen=True)
class LGConverterOptions:
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
    a = left or []
    b = right or []
    return list(a) + list(b)


class LGState(TypedDict, total=False):
    __updates__: Annotated[List[StateUpdate], _concat_updates]
    __goto__: Any


def _resolve_start_nodes_and_adj(
    *,
    workflow_engine: Any,
    workflow_id: str,
) -> tuple[Any, Dict[str, Any], Dict[str, List[Any]]]:
    """Load workflow design. We avoid validate_workflow_design due to metadata key mismatches."""
    return wf_design.load_workflow_design(workflow_engine=workflow_engine, workflow_id=workflow_id)


def _route_next(
    *,
    edges: List[Any],
    state: dict,
    last_result: Any,
    fanout: bool,
    predicate_registry: Dict[str, BasePredicate],
) -> List[str]:
    """Mirror WorkflowRuntime._route_next (contract.BasePredicate signature)."""
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


def to_langgraph(
    *,
    workflow_engine: Any,
    workflow_id: str,
    step_resolver: Any,
    predicate_registry: Dict[str, BasePredicate],
    options: Optional[LGConverterOptions] = None,
):
    """Convert workflow to a LangGraph compiled runnable."""
    opt = options or LGConverterOptions()
    if opt.updates_key != "__updates__":
        raise ValueError("This converter version requires updates_key='__updates__'.")

    start, nodes, adj = _resolve_start_nodes_and_adj(workflow_engine=workflow_engine, workflow_id=workflow_id)

    sg = StateGraph(LGState)

    def apply_node(state: LGState, config: RunnableConfig | None = None) -> Command:
        pending = state.get(opt.updates_key, []) or []
        if pending:
            _apply_state_update(state, pending)
        state[opt.updates_key] = []
        goto = state.pop("__goto__", None)
        return Command(goto=goto)

    sg.add_node(opt.apply_node_id, apply_node)

    for node_id, node in nodes.items():
        op = node.op
        fn = step_resolver.resolve(op) if hasattr(step_resolver, "resolve") else step_resolver(op)

        def make_step(nid: str, node_obj: Any, fn_):
            def step_node(state: LGState, config: RunnableConfig | None = None) -> Command:
                out = fn_(state)

                updates: List[StateUpdate] = []
                if isinstance(out, dict):
                    updates = [("u", out)] if out else []
                    next_step_names: List[str] = []
                    # minimal proxy result for BasePredicate
                    class _R:
                        def __init__(self, ns): self.next_step_names = ns
                    result_obj = _R(next_step_names)
                else:
                    updates = list(getattr(out, "state_update", []) or [])
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
                    goto = [Send(n, state) for n in next_nodes]

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
