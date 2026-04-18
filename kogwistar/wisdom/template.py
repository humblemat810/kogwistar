from __future__ import annotations

"""Reusable execution-history wisdom emission helpers."""

from typing import Any, Callable

from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.maintenance.artifacts import write_versioned_artifact
from kogwistar.wisdom.models import ExecutionWisdomTemplateResult
from kogwistar.workflow.analytics import (
    ExecutionFailurePattern,
    summarize_execution_failure_patterns,
)


def write_execution_wisdom_artifacts(
    source_engine: Any,
    *,
    target_engine: Any,
    source_namespace: str,
    target_namespace: str,
    source_where: dict[str, Any],
    build_node_for_pattern: Callable[[ExecutionFailurePattern, list[Any], int], Any],
    match_where_for_pattern: Callable[[ExecutionFailurePattern], dict[str, Any]],
    min_failure_signals: int = 2,
) -> list[ExecutionWisdomTemplateResult]:
    """Write one wisdom artifact per repeated execution-failure pattern."""
    with scoped_namespace(source_engine, source_namespace):
        step_exec_nodes = list(source_engine.read.get_nodes(where=source_where))

    patterns = summarize_execution_failure_patterns(
        step_exec_nodes,
        min_failure_signals=min_failure_signals,
    )

    results: list[ExecutionWisdomTemplateResult] = []
    for pattern in patterns:
        write_result = write_versioned_artifact(
            target_engine,
            namespace=target_namespace,
            match_where=match_where_for_pattern(pattern),
            build_node=lambda existing, created_at_ms, _pattern=pattern: build_node_for_pattern(
                _pattern,
                list(existing),
                created_at_ms,
            ),
            replace_existing=True,
        )
        results.append(
            ExecutionWisdomTemplateResult(
                step_op=pattern.step_op,
                failure_count=len(pattern.failure_nodes),
                run_ids=pattern.run_ids,
                write_result=write_result,
            )
        )
    return results
