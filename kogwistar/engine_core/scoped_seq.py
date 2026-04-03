from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine


ScopedSeqPredicate = Callable[["GraphKnowledgeEngine", Any], bool]
ScopedSeqScopeIdGetter = Callable[["GraphKnowledgeEngine", Any], str | None]


@dataclass(frozen=True)
class ScopedSeqHookConfig:
    """Config for installing reusable scoped-sequence stamping hooks."""

    metadata_field: str = "seq"
    should_stamp_node: ScopedSeqPredicate | None = None
    scope_id_for_node: ScopedSeqScopeIdGetter | None = None
    should_stamp_edge: ScopedSeqPredicate | None = None
    scope_id_for_edge: ScopedSeqScopeIdGetter | None = None


def _next_scoped_seq(engine: "GraphKnowledgeEngine", scope_id: str) -> int:
    meta = getattr(engine, "meta_sqlite", None)
    alloc = getattr(meta, "next_scoped_seq", None)
    if not callable(alloc):
        alloc = getattr(meta, "next_user_seq", None)
    if not callable(alloc):
        raise AttributeError(
            "engine.meta_sqlite must provide next_scoped_seq(scope_id) or next_user_seq(user_id)"
        )
    return int(alloc(str(scope_id)))


def _maybe_assign_subject_scoped_seq(
    engine: "GraphKnowledgeEngine",
    subject: Any,
    *,
    metadata_field: str,
    predicate: ScopedSeqPredicate | None,
    scope_id_getter: ScopedSeqScopeIdGetter | None,
) -> bool:
    if predicate is not None and not bool(predicate(engine, subject)):
        return False
    if scope_id_getter is None:
        return False
    scope_id = scope_id_getter(engine, subject)
    if scope_id is None or str(scope_id) == "":
        return False

    md = dict(getattr(subject, "metadata", {}) or {})
    try:
        existing_seq = md.get(metadata_field)
        if existing_seq is not None and int(existing_seq) > 0:
            return False
    except Exception:
        pass

    md[metadata_field] = _next_scoped_seq(engine, str(scope_id))
    subject.metadata = md
    return True


def maybe_assign_node_scoped_seq(
    engine: "GraphKnowledgeEngine",
    node: Any,
    *,
    config: ScopedSeqHookConfig,
) -> bool:
    return _maybe_assign_subject_scoped_seq(
        engine,
        node,
        metadata_field=str(config.metadata_field or "seq"),
        predicate=config.should_stamp_node,
        scope_id_getter=config.scope_id_for_node,
    )


def maybe_assign_edge_scoped_seq(
    engine: "GraphKnowledgeEngine",
    edge: Any,
    *,
    config: ScopedSeqHookConfig,
) -> bool:
    return _maybe_assign_subject_scoped_seq(
        engine,
        edge,
        metadata_field=str(config.metadata_field or "seq"),
        predicate=config.should_stamp_edge,
        scope_id_getter=config.scope_id_for_edge,
    )


def install_scoped_seq_hooks(
    engine: "GraphKnowledgeEngine",
    config: ScopedSeqHookConfig,
    *,
    ready_attr: str = "_scoped_seq_hooks_ready",
) -> None:
    """Install node/edge hooks that stamp a per-scope sequence onto metadata."""

    if getattr(engine, ready_attr, False):
        return

    if config.scope_id_for_node is not None:

        def _node_hook(node: Any) -> None:
            maybe_assign_node_scoped_seq(engine, node, config=config)

        node_hooks = getattr(engine, "pre_add_node_hooks", None)
        if isinstance(node_hooks, list):
            node_hooks.append(_node_hook)

    if config.scope_id_for_edge is not None:

        def _edge_hook(edge: Any) -> bool:
            maybe_assign_edge_scoped_seq(engine, edge, config=config)
            return False

        edge_hooks = getattr(engine, "pre_add_edge_hooks", None)
        if isinstance(edge_hooks, list):
            edge_hooks.append(_edge_hook)
        pure_edge_hooks = getattr(engine, "pre_add_pure_edge_hooks", None)
        if isinstance(pure_edge_hooks, list):
            pure_edge_hooks.append(_edge_hook)

    setattr(engine, ready_attr, True)
