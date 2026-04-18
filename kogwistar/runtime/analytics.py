from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ExecutionFailurePattern:
    """Grouped repeated workflow failure evidence for one step operation."""

    step_op: str
    failure_nodes: tuple[Any, ...]
    run_ids: tuple[str, ...]


def summarize_execution_failure_patterns(
    step_exec_nodes: Iterable[Any],
    *,
    min_failure_signals: int = 2,
) -> list[ExecutionFailurePattern]:
    """Group repeated workflow failure records by step operation.

    This is intentionally policy-light: it only identifies repeated failure
    evidence and leaves wisdom authorship to the caller.
    """
    from collections import defaultdict

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
