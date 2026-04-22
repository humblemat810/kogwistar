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
