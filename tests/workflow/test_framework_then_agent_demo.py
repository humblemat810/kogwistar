# ruff: noqa: E402
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

from kogwistar.demo import (
    run_framework_then_agent_demo,
    run_framework_then_agent_demo_suite,
)


def test_framework_then_agent_demo_runs_and_organizes_mock_notes():
    result = run_framework_then_agent_demo(auto_approve=True)

    assert result["framework_step_order"] == [
        "plan",
        "approve",
        "act",
        "observe",
        "end",
    ]
    assert result["transition_map"]["approve"] == {"approved": "act", "denied": "end"}
    assert result["run_status"] == "succeeded"
    assert result["final_state"]["completed"] is True
    assert result["final_state"]["final_status"] == "completed"
    assert result["staged_moves"] == [
        {"note_id": "note-1", "note_title": "Vendor invoice", "destination": "finance/"},
        {"note_id": "note-2", "note_title": "Team meeting", "destination": "meetings/"},
        {"note_id": "note-3", "note_title": "Loose idea", "destination": "archive/"},
    ]
    assert result["runtime_step_ops"] == [
        "plan",
        "approve",
        "act",
        "observe",
        "plan",
        "approve",
        "act",
        "observe",
        "plan",
        "approve",
        "act",
        "observe",
        "end",
    ]
    assert [event["step"] for event in result["execution_history"]] == [
        "plan",
        "approve",
        "act",
        "observe",
        "plan",
        "approve",
        "act",
        "observe",
        "plan",
        "approve",
        "act",
        "observe",
        "end",
    ]
    assert result["trace_counts"]["workflow_step_exec"] == 13


def test_guard_step_blocks_action_cleanly():
    result = run_framework_then_agent_demo(auto_approve=False)

    assert result["run_status"] == "succeeded"
    assert result["final_state"]["blocked"] is True
    assert result["final_state"]["blocked_reason"] == "approval required"
    assert result["final_state"]["completed"] is False
    assert result["staged_moves"] == []
    assert result["runtime_step_ops"] == ["plan", "approve", "end"]
    assert [event["step"] for event in result["execution_history"]] == [
        "plan",
        "approve",
        "end",
    ]


def test_easy_and_harder_framework_variants_show_swap_and_adapter():
    result = run_framework_then_agent_demo_suite(auto_approve=True)

    easy = result["easy"]
    harder = result["harder"]

    assert easy["framework_name"] == "PlanActObserveNoApprovalFramework"
    assert easy["framework_step_order"] == ["plan", "act", "observe", "end"]
    assert easy["staged_moves"] == [
        {"note_id": "note-1", "note_title": "Vendor invoice", "destination": "finance/"},
        {"note_id": "note-2", "note_title": "Team meeting", "destination": "meetings/"},
        {"note_id": "note-3", "note_title": "Loose idea", "destination": "archive/"},
    ]
    assert [event["step"] for event in easy["execution_history"]] == [
        "plan",
        "act",
        "observe",
        "plan",
        "act",
        "observe",
        "plan",
        "act",
        "observe",
        "end",
    ]

    assert harder["framework_name"] == "BatchClassifyThenApplyFramework"
    assert harder["framework_step_order"] == [
        "collect",
        "classify_batch",
        "apply_batch",
        "end",
    ]
    assert harder["transition_map"]["classify_batch"] == {"has_plan": "apply_batch"}
    assert harder["staged_moves"] == [
        {"note_id": "note-1", "note_title": "Vendor invoice", "destination": "finance/"},
        {"note_id": "note-2", "note_title": "Team meeting", "destination": "meetings/"},
        {"note_id": "note-3", "note_title": "Loose idea", "destination": "archive/"},
    ]
    assert harder["final_state"]["batch_applied"] is True
    assert harder["final_state"]["completed"] is True
    assert [event["step"] for event in harder["execution_history"]] == [
        "collect",
        "classify_batch",
        "apply_batch",
        "end",
    ]
