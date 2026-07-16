from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import BasePredicate, WorkflowEdgeInfo
from .models import get_route_next_names


@dataclass(frozen=True)
class RouteComputation:
    next_node_ids: list[str]
    selected_edges: list[Any]
    evaluated: list[tuple[str, bool]]
    selected: list[tuple[str, str, str]]


def compute_route_next(
    *,
    edges: list[Any],
    state: dict[str, Any],
    last_result: Any,
    fanout: bool,
    predicate_registry: dict[str, Any],
    nodes: dict[str, Any] | None = None,
    _native_disabled: bool = False,
) -> RouteComputation:
    matched: list[tuple[Any, str]] = []
    evaluated: list[tuple[str, bool]] = []
    selected: list[tuple[str, str, str]] = []

    def _edge_id(edge: Any) -> str:
        return str(
            getattr(edge, "id", None)
            or getattr(edge, "edge_id", None)
            or f"{getattr(edge, 'predicate', None)}->{(getattr(edge, 'target_ids', None) or [''])[0]}"
        )

    def _first_target(edge: Any) -> str | None:
        tids = getattr(edge, "target_ids", None) or []
        if not tids:
            return None
        return str(tids[0])

    def _edge_multiplicity(edge: Any) -> str:
        value = getattr(edge, "multiplicity", None)
        if value is not None:
            return str(value)
        md = getattr(edge, "metadata", {}) or {}
        return str(md.get("wf_multiplicity", "one"))

    def _edge_is_default(edge: Any) -> bool:
        value = getattr(edge, "is_default", None)
        if value is not None:
            return bool(value)
        md = getattr(edge, "metadata", {}) or {}
        return bool(md.get("wf_is_default", False))

    def _stop_on_first(edge: Any) -> bool:
        return (not fanout) and (_edge_multiplicity(edge) != "many")

    def _target_aliases(target_id: str) -> set[str]:
        aliases = {str(target_id), str(target_id).split("|")[-1]}
        if nodes is not None:
            target_node = nodes.get(str(target_id))
            if target_node is not None:
                label = getattr(target_node, "label", None)
                if label:
                    aliases.add(str(label))
                op = getattr(target_node, "op", None)
                if op:
                    aliases.add(str(op))
        return aliases

    def _edge_aliases(edge: Any, target_id: str) -> set[str]:
        aliases = _target_aliases(target_id)
        label = getattr(edge, "label", None)
        if label:
            aliases.add(str(label))
        name = getattr(edge, "name", None)
        if name:
            aliases.add(str(name))
        return aliases

    def _edge_info(edge: Any) -> WorkflowEdgeInfo:
        try:
            return WorkflowEdgeInfo.from_workflow_edge(edge)
        except Exception:
            src_ids = list(getattr(edge, "source_ids", None) or [""])
            tgt_ids = list(getattr(edge, "target_ids", None) or [""])
            md = getattr(edge, "metadata", {}) or {}
            return WorkflowEdgeInfo(
                name=str(getattr(edge, "label", None) or getattr(edge, "name", None) or ""),
                edge_id=_edge_id(edge),
                src=str(src_ids[0] if src_ids else ""),
                dst=str(tgt_ids[0] if tgt_ids else ""),
                predicate=md.get("wf_predicate"),
                priority=int(md.get("wf_priority", 100)),
                is_default=bool(md.get("wf_is_default", False)),
                multiplicity=str(md.get("wf_multiplicity", "one")),
            )

    from kogwistar._rust_bridge import runtime_implementation_mode, runtime_select_route

    runtime_mode = "python" if _native_disabled else runtime_implementation_mode()
    if runtime_mode != "python":
        explicit_next = get_route_next_names(last_result)
        explicit_has_match = bool(explicit_next) and any(
            alias in _edge_aliases(edge, target)
            for alias in explicit_next
            for edge in edges
            for target in [_first_target(edge)]
            if target is not None
        )
        predicate_results: dict[int, bool] = {}
        if not explicit_has_match:
            for index, edge in enumerate(edges):
                predicate = getattr(edge, "predicate", None)
                if predicate is None or _first_target(edge) is None:
                    continue
                pred = predicate_registry.get(str(predicate))
                if pred is None:
                    predicate_results[index] = False
                    continue
                try:
                    predicate_results[index] = bool(
                        pred(_edge_info(edge), state, last_result)
                    )
                except Exception:
                    predicate_results[index] = False

        failure_only = getattr(last_result, "status", None) == "failure"
        base_results: dict[int, bool] = {}
        if (
            not explicit_has_match
            and not any(predicate_results.values())
            and not failure_only
        ):
            node_decider = BasePredicate()
            for index, edge in enumerate(edges):
                if getattr(edge, "predicate", None) is not None:
                    continue
                if _first_target(edge) is None:
                    continue
                try:
                    base_results[index] = bool(
                        node_decider(_edge_info(edge), state, last_result)
                    )
                except Exception:
                    base_results[index] = False

        route_payload: list[dict[str, Any]] = []
        for index, edge in enumerate(edges):
            targets = [str(value) for value in (getattr(edge, "target_ids", None) or [])]
            first_target = targets[0] if targets else ""
            try:
                priority = int(getattr(edge, "priority"))
            except Exception:
                priority = int(_edge_info(edge).priority)
            route_payload.append(
                {
                    "edge_id": _edge_id(edge),
                    "target_ids": targets,
                    "aliases": sorted(_edge_aliases(edge, first_target)) if first_target else [],
                    "predicate": (
                        str(getattr(edge, "predicate"))
                        if getattr(edge, "predicate", None) is not None
                        else None
                    ),
                    "multiplicity": _edge_multiplicity(edge),
                    "is_default": _edge_is_default(edge),
                    "priority": priority,
                    "predicate_result": predicate_results.get(index),
                    "base_result": base_results.get(index),
                }
            )
        native = runtime_select_route(
            payload={
                "edges": route_payload,
                "explicit_next": explicit_next,
                "fanout": fanout,
                "failure_only": failure_only,
            }
        )
        indices = [int(value) for value in native["selected_edge_indices"]]
        native_computation = RouteComputation(
            next_node_ids=[str(value) for value in native["next_node_ids"]],
            selected_edges=[edges[index] for index in indices],
            evaluated=[(str(key), bool(value)) for key, value in native["evaluated"]],
            selected=[
                (str(edge_id), str(target), str(reason))
                for edge_id, target, reason in native["selected"]
            ],
        )
        if runtime_mode == "rust":
            return native_computation

        def _recorded_predicate(info: WorkflowEdgeInfo, _state: Any, _result: Any) -> bool:
            for index, edge in enumerate(edges):
                if _edge_id(edge) == info.edge_id:
                    return predicate_results.get(index, False)
            return False

        shadow_registry = {
            name: _recorded_predicate for name in predicate_registry
        }
        python_computation = compute_route_next(
            edges=edges,
            state=state,
            last_result=last_result,
            fanout=fanout,
            predicate_registry=shadow_registry,
            nodes=nodes,
            _native_disabled=True,
        )
        if (
            native_computation.next_node_ids != python_computation.next_node_ids
            or native_computation.selected_edges != python_computation.selected_edges
            or native_computation.evaluated != python_computation.evaluated
            or native_computation.selected != python_computation.selected
        ):
            from kogwistar._rust_bridge import RustParityError

            raise RustParityError(
                "Rust parity mismatch for runtime route selection: "
                f"python={python_computation!r}, rust={native_computation!r}"
            )
        return python_computation

    explicit_next = get_route_next_names(last_result)
    if explicit_next:
        explicit_matches: list[str] = []
        explicit_edges: list[Any] = []
        for alias in explicit_next:
            matched_edge = None
            matched_target = None
            for edge in edges:
                tgt = _first_target(edge)
                if tgt is None:
                    continue
                if alias in _edge_aliases(edge, tgt):
                    matched_edge = edge
                    matched_target = tgt
                    selected.append((_edge_id(edge), tgt, "explicit"))
                    break
            evaluated.append((f"_route_next:{alias}", matched_target is not None))
            if matched_edge is not None and matched_target is not None:
                explicit_edges.append(matched_edge)
                explicit_matches.append(str(matched_target))
        if explicit_matches:
            return RouteComputation(
                next_node_ids=explicit_matches,
                selected_edges=explicit_edges,
                evaluated=evaluated,
                selected=selected,
            )

    failure_only = getattr(last_result, "status", None) == "failure"

    for edge in edges:
        if getattr(edge, "predicate", None) is None:
            continue
        tgt = _first_target(edge)
        if tgt is None:
            continue
        pred_name = str(getattr(edge, "predicate", ""))
        pred = predicate_registry.get(pred_name)
        if pred is None:
            evaluated.append((f"{_edge_id(edge)}:{pred_name}", False))
            continue
        workflow_info = _edge_info(edge)
        try:
            ok = bool(pred(workflow_info, state, last_result))
        except Exception:
            ok = False
        evaluated.append((f"{_edge_id(edge)}:{pred_name}", ok))
        if ok:
            matched.append((edge, tgt))
            selected.append((_edge_id(edge), tgt, "predicate"))

    matched.sort(key=lambda item: item[0].priority, reverse=True)

    candidate_edges: list[Any] = []
    candidate_ids: list[str] = []
    for edge, next_node_id in matched:
        if _stop_on_first(edge):
            if not candidate_ids:
                candidate_edges.append(edge)
                candidate_ids.append(next_node_id)
            return RouteComputation(
                next_node_ids=candidate_ids,
                selected_edges=candidate_edges,
                evaluated=evaluated,
                selected=selected,
            )
        candidate_edges.append(edge)
        candidate_ids.append(next_node_id)

    if candidate_ids:
        return RouteComputation(
            next_node_ids=candidate_ids,
            selected_edges=candidate_edges,
            evaluated=evaluated,
            selected=selected,
        )

    if failure_only:
        return RouteComputation(
            next_node_ids=[],
            selected_edges=[],
            evaluated=evaluated,
            selected=selected,
        )

    node_decider = BasePredicate()
    for edge in edges:
        if getattr(edge, "predicate", None) is not None:
            continue
        tgt = _first_target(edge)
        if tgt is None:
            continue
        workflow_info = _edge_info(edge)
        try:
            ok = bool(node_decider(workflow_info, state, last_result))
        except Exception:
            ok = False
        evaluated.append((f"{_edge_id(edge)}:<base>", ok))
        if ok:
            matched.append((edge, tgt))
            selected.append((_edge_id(edge), tgt, "base"))
            if _stop_on_first(edge):
                return RouteComputation(
                    next_node_ids=[tgt],
                    selected_edges=[edge],
                    evaluated=evaluated,
                    selected=selected,
                )

    if matched:
        allow_many = fanout or any(
            _edge_multiplicity(edge) == "many" for edge, _ in matched
        )
        picked = matched if allow_many else matched[0:1]
        return RouteComputation(
            next_node_ids=[item[1] for item in picked],
            selected_edges=[item[0] for item in picked],
            evaluated=evaluated,
            selected=selected,
        )

    for edge in edges:
        if _edge_is_default(edge):
            tids = [str(x) for x in (getattr(edge, "target_ids", None) or [])]
            if not tids:
                continue
            picked_ids = tids if fanout else tids[0:1]
            selected.append((_edge_id(edge), picked_ids[0], "default"))
            return RouteComputation(
                next_node_ids=picked_ids,
                selected_edges=[edge],
                evaluated=evaluated,
                selected=selected,
            )

    return RouteComputation(
        next_node_ids=[],
        selected_edges=[],
        evaluated=evaluated,
        selected=selected,
    )
