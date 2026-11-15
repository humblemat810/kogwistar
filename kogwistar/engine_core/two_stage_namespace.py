"""Namespace lookup shared by two-stage projection adapters."""

from __future__ import annotations


def active_graph_namespace(engine: object) -> str:
    """Return the active scoped namespace, including subsystem proxy views."""

    indexing = getattr(engine, "indexing", None)
    indexing_engine = getattr(indexing, "engine", None)
    namespace = getattr(indexing_engine, "namespace", None)
    if namespace is None:
        namespace = getattr(engine, "namespace", "default")
    return str(namespace or "default")


__all__ = ["active_graph_namespace"]
