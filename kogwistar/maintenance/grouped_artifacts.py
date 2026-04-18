from __future__ import annotations

"""Grouped replacement-artifact helpers for maintenance flows."""

from typing import Any, Callable

from kogwistar.maintenance.artifacts import (
    write_versioned_artifact,
)
from kogwistar.maintenance.models import GroupedArtifactWriteResult


def write_grouped_versioned_artifacts(
    source_engine: Any,
    *,
    target_engine: Any,
    source_namespace: str,
    target_namespace: str,
    source_where: dict[str, Any],
    group_key_for_node: Callable[[Any], str],
    build_node_for_group: Callable[[str, list[Any], list[Any], int], Any],
    match_where_for_group: Callable[[str], dict[str, Any]],
    replace_existing: bool = True,
) -> list[GroupedArtifactWriteResult]:
    """Write one replacement artifact per grouped source slice."""
    from kogwistar.engine_core.engine import scoped_namespace

    with scoped_namespace(source_engine, source_namespace):
        source_nodes = list(source_engine.read.get_nodes(where=source_where))

    grouped: dict[str, list[Any]] = {}
    for node in source_nodes:
        grouped.setdefault(str(group_key_for_node(node)), []).append(node)

    results: list[GroupedArtifactWriteResult] = []
    for group_key, nodes in grouped.items():
        write_result = write_versioned_artifact(
            target_engine,
            namespace=target_namespace,
            match_where=match_where_for_group(group_key),
            build_node=lambda existing, created_at_ms, _group_key=group_key, _nodes=list(nodes): build_node_for_group(
                _group_key,
                _nodes,
                list(existing),
                created_at_ms,
            ),
            replace_existing=replace_existing,
        )
        results.append(
            GroupedArtifactWriteResult(
                group_key=group_key,
                source_node_count=len(nodes),
                write_result=write_result,
            )
        )
    return results
