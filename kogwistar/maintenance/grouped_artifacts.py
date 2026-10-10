"""Grouped replacement-artifact helpers for maintenance flows."""

from __future__ import annotations

from collections.abc import Mapping

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Node
from kogwistar.maintenance.artifacts import (
    write_versioned_artifact,
)
from kogwistar.maintenance.contracts import (
    BeforeWrite,
    GroupedArtifactNodeBuilder,
    GroupKeyForNode,
    MatchWhereForGroup,
)
from kogwistar.maintenance.models import GroupedArtifactWriteResult


def write_grouped_versioned_artifacts(
    source_engine: GraphKnowledgeEngine,
    *,
    target_engine: GraphKnowledgeEngine,
    source_namespace: str,
    target_namespace: str,
    source_where: Mapping[str, object],
    group_key_for_node: GroupKeyForNode,
    build_node_for_group: GroupedArtifactNodeBuilder,
    match_where_for_group: MatchWhereForGroup,
    replace_existing: bool = True,
    before_write: BeforeWrite[str] | None = None,
) -> list[GroupedArtifactWriteResult]:
    """Write one replacement artifact per grouped source slice."""
    from kogwistar.engine_core.engine import scoped_namespace

    with scoped_namespace(source_engine, source_namespace):
        source_nodes: list[Node] = list(
            source_engine.read.get_nodes(where=dict(source_where))
        )

    grouped: dict[str, list[Node]] = {}
    for node in source_nodes:
        grouped.setdefault(str(group_key_for_node(node)), []).append(node)

    results: list[GroupedArtifactWriteResult] = []
    for group_key, nodes in grouped.items():
        if before_write is not None:
            before_write(group_key)
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
