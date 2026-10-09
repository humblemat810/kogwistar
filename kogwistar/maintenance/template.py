"""Reusable maintenance templates."""

from __future__ import annotations

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.maintenance.contracts import BeforeWrite
from kogwistar.maintenance.contracts import (
    GroupKeyForNode,
    GroupedArtifactNodeBuilder,
    MatchWhereForGroup,
)
from kogwistar.maintenance.grouped_artifacts import (
    write_grouped_versioned_artifacts,
)
from kogwistar.maintenance.models import MaintenanceTemplateResult


def run_grouped_maintenance_template(
    source_engine: GraphKnowledgeEngine,
    *,
    target_engine: GraphKnowledgeEngine,
    source_namespace: str,
    target_namespace: str,
    source_where: dict[str, object],
    group_key_for_node: GroupKeyForNode,
    build_node_for_group: GroupedArtifactNodeBuilder,
    match_where_for_group: MatchWhereForGroup,
    replace_existing: bool = True,
    before_write: BeforeWrite[str] | None = None,
) -> MaintenanceTemplateResult:
    grouped_results = write_grouped_versioned_artifacts(
        source_engine,
        target_engine=target_engine,
        source_namespace=source_namespace,
        target_namespace=target_namespace,
        source_where=source_where,
        group_key_for_node=group_key_for_node,
        build_node_for_group=build_node_for_group,
        match_where_for_group=match_where_for_group,
        replace_existing=replace_existing,
        before_write=before_write,
    )
    return MaintenanceTemplateResult(
        grouped_results=grouped_results,
        source_node_count=sum(result.source_node_count for result in grouped_results),
    )
