from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

RUN_META_KEYS_PREFIXES = (
    "run_",
    "attempt_",
    "worker_",
    "timestamp",
    "ts_",
    "duration",
    "latency",
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _meta(obj: Any) -> dict[str, Any]:
    m = _get(obj, "metadata", None)
    return m or {}


def _strip_run_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if any(k.startswith(p) for p in RUN_META_KEYS_PREFIXES):
            continue
        if k in {"run_id", "run_step_seq", "attempt_seq"}:
            continue
        out[k] = v
    return out


def _edge_tuple(e: Any) -> tuple[str, str, str, str]:
    eid = str(_get(e, "id", _get(e, "edge_id", "")))
    rel = str(_get(e, "relation", _get(e, "type", _get(e, "label", ""))))
    src = str(_get(e, "source", _get(e, "source_id", _get(e, "src", ""))))
    dst = str(_get(e, "target", _get(e, "target_id", _get(e, "dst", ""))))
    return (eid, rel, src, dst)


def _is_workflow_trace_node(n: Any) -> bool:
    et = _meta(n).get("entity_type")
    return et in {
        "workflow_node",
        "workflow_edge",
        "workflow_run",
        "workflow_step_exec",
        "workflow_checkpoint",
    }


def _is_workflow_trace_edge(e: Any) -> bool:
    rel = str(_get(e, "relation", _get(e, "type", _get(e, "label", ""))))
    return rel.startswith("workflow_")


@dataclass(frozen=True)
class ConvGraphView:
    ui_chain_turn_ids: tuple[str, ...]
    next_turn_edges: tuple[tuple[str, str, str, str], ...]
    references_edges: tuple[tuple[str, str, str, str], ...]
    summarizes_edges: tuple[tuple[str, str, str, str], ...]
    depends_on_edges: tuple[tuple[str, str, str, str], ...]
    summary_node_ids: tuple[str, ...]
    snapshot_node_ids: tuple[str, ...]
    snapshot_costs: tuple[tuple[str, int, int | None], ...]


def extract_conv_view(
    conversation_engine: Any, *, conversation_id: str | None = None
) -> ConvGraphView:
    if conversation_id is not None:
        try:
            nodes = conversation_engine.get_nodes(
                where={"conversation_id": conversation_id}
            )
        except Exception:
            nodes = conversation_engine.get_nodes()
    else:
        nodes = conversation_engine.get_nodes()

    domain_nodes = [n for n in nodes if not _is_workflow_trace_node(n)]

    turns: list[Any] = []
    summaries: list[Any] = []
    snaps: list[Any] = []

    for n in domain_nodes:
        md = _meta(n)
        et = md.get("entity_type")
        if et == "conversation_turn" and md.get("in_ui_chain") is True:
            turns.append(n)
        elif et == "conversation_summary":
            summaries.append(n)
        elif et == "context_snapshot":
            snaps.append(n)

    def _turn_sort_key(n: Any) -> tuple[int, str]:
        md = _meta(n)
        idx = md.get("turn_index", _get(n, "turn_index", None))
        try:
            i = int(idx)
        except Exception:
            i = 10**9
        return (i, str(_get(n, "id", "")))

    turns_sorted = sorted(turns, key=_turn_sort_key)
    ui_chain_turn_ids = tuple(str(_get(n, "id", "")) for n in turns_sorted)

    summary_node_ids = tuple(sorted(str(_get(n, "id", "")) for n in summaries))
    snapshot_node_ids = tuple(sorted(str(_get(n, "id", "")) for n in snaps))

    snapshot_costs_list: list[tuple[str, int, int | None]] = []
    for s in snaps:
        sid = str(_get(s, "id", ""))
        md = _strip_run_meta(_meta(s))
        cc = md.get("cost.char_count", 0)
        tc = md.get("cost.token_count", None)
        try:
            cc_i = int(cc)
        except Exception:
            cc_i = 0
        try:
            tc_i = None if tc is None else int(tc)
        except Exception:
            tc_i = None
        snapshot_costs_list.append((sid, cc_i, tc_i))
    snapshot_costs = tuple(sorted(snapshot_costs_list, key=lambda t: t[0]))

    def _safe_get_edges(where: dict[str, Any]) -> list[Any]:
        try:
            return conversation_engine.get_edges(where=where)
        except Exception:
            return []

    def _canon_edges(rel: str) -> tuple[tuple[str, str, str, str], ...]:
        edges = _safe_get_edges({"relation": rel})
        edges = [e for e in edges if not _is_workflow_trace_edge(e)]
        return tuple(sorted(_edge_tuple(e) for e in edges))

    next_turn_edges = _canon_edges("next_turn")
    references_edges = _canon_edges("references")
    summarizes_edges = _canon_edges("summarizes")
    depends_on_edges = _canon_edges("depends_on")

    return ConvGraphView(
        ui_chain_turn_ids=ui_chain_turn_ids,
        next_turn_edges=next_turn_edges,
        references_edges=references_edges,
        summarizes_edges=summarizes_edges,
        depends_on_edges=depends_on_edges,
        summary_node_ids=summary_node_ids,
        snapshot_node_ids=snapshot_node_ids,
        snapshot_costs=snapshot_costs,
    )


def diff_views(a: ConvGraphView, b: ConvGraphView) -> str:
    """Human-friendly diff for pytest failures."""
    parts: list[str] = []
    for field in a.__dataclass_fields__.keys():
        av = getattr(a, field)
        bv = getattr(b, field)
        if av != bv:
            parts.append(f"- {field} differs:\n  a={av}\n  b={bv}")
    return "\n".join(parts) if parts else "(no diff)"


def assert_tier0_invariants(view: ConvGraphView) -> None:
    outgoing: dict[str, int] = {}
    for _, _, src, _ in view.next_turn_edges:
        outgoing[src] = outgoing.get(src, 0) + 1
    bad = {src: c for src, c in outgoing.items() if c > 1}
    if bad:
        raise AssertionError(
            f"Tier0 violation: multiple outgoing next_turn edges: {bad}"
        )


def _diff(a: Any, b: Any) -> str:
    parts: list[str] = []
    for field in a.__dataclass_fields__.keys():
        av = getattr(a, field)
        bv = getattr(b, field)
        if av != bv:
            parts.append(f"- {field} differs:\n  a={av}\n  b={bv}")
    return "\n".join(parts) if parts else "(no diff)"


def assert_views_equivalent(a: ConvGraphView, b: ConvGraphView) -> None:
    if a != b:
        raise AssertionError(diff_views(a, b))


def assert_tier1_equal(a: ConvGraphView, b: ConvGraphView) -> None:
    if a != b:
        raise AssertionError(_diff(a, b))
