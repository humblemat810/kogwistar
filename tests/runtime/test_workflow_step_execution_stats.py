from __future__ import annotations

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.core

from kogwistar.workflow.analytics import summarize_workflow_step_execution_stats


@dataclass
class _Node:
    metadata: dict[str, object]


def test_summarize_workflow_step_execution_stats_groups_counts_and_latency():
    nodes = [
        _Node({"step_op": "distill", "status": "ok", "run_id": "run-a", "duration_ms": 10}),
        _Node({"step_op": "distill", "status": "failure", "run_id": "run-b", "duration_ms": 30}),
        _Node({"step_op": "distill", "status": "error", "run_id": "run-c", "duration_ms": 20}),
        _Node({"step_op": "check_done", "status": "success", "run_id": "run-d", "duration_ms": 5}),
    ]

    stats = summarize_workflow_step_execution_stats(nodes)

    assert [s.step_op for s in stats] == ["check_done", "distill"]
    distill = next(item for item in stats if item.step_op == "distill")
    assert distill.total_count == 3
    assert distill.failure_count == 1
    assert distill.error_count == 1
    assert distill.success_count == 1
    assert distill.duration_ms_total == 60
    assert distill.duration_ms_max == 30
    assert distill.run_ids == ("run-a", "run-b", "run-c")
