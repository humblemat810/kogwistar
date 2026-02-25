
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# NOTE:
# This helper is intentionally defensive about the underlying Node/Edge shapes.
# In this repo, nodes/edges are typically Pydantic models, but some backends/tests may surface dict-like shapes.


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _meta(obj: Any) -> dict[str, Any]:
    m = _get(obj, "metadata", None)
    return m or {}


def _edge_tuple(e: Any) -> tuple[str, str, str, str]:
    """(edge_id, relation, source_id, target_id) canonical tuple."""
    eid = str(_get(e, "id", _get(e, "edge_id", "")))
    rel = str(_get(e, "relation", _get(e, "type", _get(e, "label", ""))))
    src = str(_get(e, "source", _get(e, "source_id", _get(e, "src", ""))))
    dst = str(_get(e, "target", _get(e, "target_id", _get(e, "dst", ""))))
    return (eid, rel, src, dst)


def _node_tuple(n: Any) -> tuple[str, str]:
    """(node_id, entity_type)"""
    nid = str(_get(n, "id", ""))
    et = str(_meta(n).get("entity_type", _get(n, "entity_type", "")))
    return (nid, et)


@dataclass(frozen=True)
class ConvGraphView:
    """Canonical, comparison-friendly projection of a conversation graph.

    Designed for v1-v2 reconciliation:
    - focuses on domain artifacts (turns, pins, summaries, snapshots)
    - ignores workflow trace artifacts (which will be filtered upstream by the extractor once v2 exists)
    """
    ui_chain_turn_ids: tuple[str, ...]
    next_turn_edges: tuple[tuple[str, str, str, str], ...]          # (edge_id, relation, src, dst)
    references_edges: tuple[tuple[str, str, str, str], ...]
    summarizes_edges: tuple[tuple[str, str, str, str], ...]
    depends_on_edges: tuple[tuple[str, str, str, str], ...]

    summary_node_ids: tuple[str, ...]
    snapshot_node_ids: tuple[str, ...]
    # snapshot_id -> (char_count, token_count)
    snapshot_costs: tuple[tuple[str, int, int | None], ...]


def extract_conv_view(conversation_engine: Any, *, conversation_id: str | None = None) -> ConvGraphView:
    """Extract a canonical view.

    If conversation_id is provided, we scope nodes/edges to that conversation when possible.
    """
    # --- Nodes ---
    if conversation_id is not None:
        try:
            nodes = conversation_engine.get_nodes(where={"conversation_id": conversation_id})
        except Exception:
            nodes = conversation_engine.get_nodes()
    else:
        nodes = conversation_engine.get_nodes()

    # Partition nodes by entity_type
    turns: list[Any] = []
    summaries: list[Any] = []
    snaps: list[Any] = []

    for n in nodes:
        md = _meta(n)
        et = md.get("entity_type")
        if et == "conversation_turn" and md.get("in_ui_chain") is True:
            turns.append(n)
        elif et == "conversation_summary":
            summaries.append(n)
        elif et == "context_snapshot":
            snaps.append(n)

    # Sort turns deterministically (prefer turn_index if present)
    def _turn_sort_key(n: Any) -> tuple[int, str]:
        idx = _get(n, "turn_index", None)
        if idx is None:
            idx = _get(_get(n, "properties", {}), "turn_index", None)
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
        md = _meta(s)
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

    # --- Edges ---
    def _safe_get_edges(where: dict[str, Any]) -> list[Any]:
        try:
            return conversation_engine.get_edges(where=where)
        except Exception:
            return []

    next_turn = [_edge_tuple(e) for e in _safe_get_edges({"relation": "next_turn"})]
    refs = [_edge_tuple(e) for e in _safe_get_edges({"relation": "references"})]
    summarizes = [_edge_tuple(e) for e in _safe_get_edges({"relation": "summarizes"})]
    depends = [_edge_tuple(e) for e in _safe_get_edges({"relation": "depends_on"})]

    # Sort edge tuples for stable comparisons
    next_turn_edges = tuple(sorted(next_turn))
    references_edges = tuple(sorted(refs))
    summarizes_edges = tuple(sorted(summarizes))
    depends_on_edges = tuple(sorted(depends))

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


def assert_views_equivalent(a: ConvGraphView, b: ConvGraphView) -> None:
    if a != b:
        raise AssertionError(diff_views(a, b))
