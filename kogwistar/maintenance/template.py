from __future__ import annotations

"""Reusable maintenance templates."""

from typing import Any, Callable

from kogwistar.maintenance.grouped_artifacts import (
    write_grouped_versioned_artifacts,
)
from kogwistar.maintenance.models import MaintenanceTemplateResult


def run_grouped_maintenance_template(
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
    )
    return MaintenanceTemplateResult(
        grouped_results=grouped_results,
        source_node_count=sum(result.source_node_count for result in grouped_results),
    )
