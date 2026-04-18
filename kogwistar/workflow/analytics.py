from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExecutionFailurePattern:
    """Grouped repeated workflow failure evidence for one step operation."""

    step_op: str
    failure_nodes: tuple[Any, ...]
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowStepExecutionStats:
    """Generic analytics for workflow step execution health."""

    step_op: str
    total_count: int
    failure_count: int
    success_count: int
    error_count: int
    duration_ms_total: int
    duration_ms_max: int
    run_ids: tuple[str, ...]


def summarize_execution_failure_patterns(
    step_exec_nodes: Iterable[Any],
    *,
    min_failure_signals: int = 2,
) -> list[ExecutionFailurePattern]:
    """Group repeated workflow failure records by step operation."""
    failures_by_op: dict[str, list[Any]] = defaultdict(list)
    for node in step_exec_nodes:
        metadata = getattr(node, "metadata", {}) or {}
        status = metadata.get("status")
        if status not in {"failure", "error"}:
            continue
        step_op = metadata.get("step_op") or metadata.get("op") or metadata.get("wf_op") or "unknown"
        failures_by_op[str(step_op)].append(node)

    patterns: list[ExecutionFailurePattern] = []
    for step_op, failure_nodes in sorted(failures_by_op.items(), key=lambda item: item[0]):
        if len(failure_nodes) < int(min_failure_signals):
            continue
        run_ids = sorted(
            {
                str((getattr(node, "metadata", {}) or {}).get("run_id", ""))
                for node in failure_nodes
            }
            - {""}
        )
        patterns.append(
            ExecutionFailurePattern(
                step_op=step_op,
                failure_nodes=tuple(failure_nodes),
                run_ids=tuple(run_ids),
            )
        )
    return patterns


def summarize_workflow_step_execution_stats(
    step_exec_nodes: Iterable[Any],
) -> list[WorkflowStepExecutionStats]:
    """Summarize per-step execution health and coarse latency statistics."""
    stats_by_op: dict[str, list[Any]] = defaultdict(list)
    for node in step_exec_nodes:
        metadata = getattr(node, "metadata", {}) or {}
        step_op = metadata.get("step_op") or metadata.get("op") or metadata.get("wf_op") or "unknown"
        stats_by_op[str(step_op)].append(node)

    summaries: list[WorkflowStepExecutionStats] = []
    for step_op, nodes in sorted(stats_by_op.items(), key=lambda item: item[0]):
        run_ids = sorted(
            {
                str((getattr(node, "metadata", {}) or {}).get("run_id", ""))
                for node in nodes
            }
            - {""}
        )
        failure_count = 0
        success_count = 0
        error_count = 0
        duration_ms_total = 0
        duration_ms_max = 0
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            status = str(metadata.get("status") or "").lower()
            if status == "failure":
                failure_count += 1
            elif status == "error":
                error_count += 1
            elif status in {"success", "ok", "succeeded"}:
                success_count += 1
            try:
                duration = int(metadata.get("duration_ms") or metadata.get("duration") or 0)
            except Exception:
                duration = 0
            if duration > 0:
                duration_ms_total += duration
                duration_ms_max = max(duration_ms_max, duration)
        summaries.append(
            WorkflowStepExecutionStats(
                step_op=step_op,
                total_count=len(nodes),
                failure_count=failure_count,
                success_count=success_count,
                error_count=error_count,
                duration_ms_total=duration_ms_total,
                duration_ms_max=duration_ms_max,
                run_ids=tuple(run_ids),
            )
        )
    return summaries
