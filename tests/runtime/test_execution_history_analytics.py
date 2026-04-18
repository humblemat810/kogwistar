from __future__ import annotations

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.core

from kogwistar.runtime.analytics import summarize_execution_failure_patterns


@dataclass
class _Node:
    metadata: dict[str, object]


def test_summarize_execution_failure_patterns_groups_repeated_failures_by_step_op():
    nodes = [
        _Node({"entity_type": "workflow_step_exec", "status": "failure", "step_op": "distill", "run_id": "run-a"}),
        _Node({"entity_type": "workflow_step_exec", "status": "error", "step_op": "distill", "run_id": "run-b"}),
        _Node({"entity_type": "workflow_step_exec", "status": "failure", "step_op": "check_done", "run_id": "run-c"}),
        _Node({"entity_type": "workflow_step_exec", "status": "success", "step_op": "distill", "run_id": "run-d"}),
    ]

    patterns = summarize_execution_failure_patterns(nodes, min_failure_signals=2)

    assert [p.step_op for p in patterns] == ["distill"]
    assert patterns[0].run_ids == ("run-a", "run-b")
    assert len(patterns[0].failure_nodes) == 2


def test_summarize_execution_failure_patterns_ignores_below_threshold():
    nodes = [
        _Node({"entity_type": "workflow_step_exec", "status": "failure", "step_op": "distill", "run_id": "run-a"}),
    ]

    patterns = summarize_execution_failure_patterns(nodes, min_failure_signals=2)

    assert patterns == []
