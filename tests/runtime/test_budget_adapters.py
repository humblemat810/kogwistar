from __future__ import annotations

import pytest

from kogwistar.runtime.budget import BudgetAttribution, BudgetEvent
from kogwistar.runtime.budget_adapters import adapt_budget_events, summarize_budget_events

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def test_generic_usage_adapter_maps_usage_to_canonical_events() -> None:
    events = adapt_budget_events(
        {"usage": {"input_tokens": 5, "output_tokens": 7, "total_cost": 1.5}},
        run_id="run-1",
    )
    assert [evt.kind for evt in events] == ["token", "token", "cost"]
    assert events[0].unit == "input_tokens"
    assert events[-1].source == "generic-usage"


def test_generic_usage_adapter_ignores_non_usage_payloads() -> None:
    assert adapt_budget_events({"x": 1}, run_id="run-1") == []


def test_generic_usage_adapter_preserves_attribution_context() -> None:
    attribution = BudgetAttribution(
        workspace_id="ws-1",
        source_document_id="doc-1",
        operation_id="op-1",
        maintenance_job_id="job-1",
        dream_job_id="dream-1",
    )

    events = adapt_budget_events(
        {"usage": {"input_tokens": 5}},
        run_id="run-1",
        scope="operation",
        attribution=attribution,
    )

    assert events[0].attribution == attribution
    assert events[0].scope == "operation"


def test_summarize_budget_events_tracks_mixed_runtime_events() -> None:
    summary = summarize_budget_events(
        [
            BudgetEvent(run_id="run-1", source="generic-usage", kind="token", amount=5, unit="input_tokens"),
            BudgetEvent(run_id="run-1", source="generic-usage", kind="token", amount=7, unit="output_tokens"),
            BudgetEvent(run_id="run-1", source="generic-usage", kind="cost", amount=1.5, unit="total_cost"),
            BudgetEvent(run_id="run-1", source="runtime", kind="debit", amount=2, unit="ms"),
        ]
    )
    assert summary["input_tokens"] == 5
    assert summary["output_tokens"] == 7
    assert summary["total_tokens"] == 12
    assert summary["total_cost"] == 1.5
    assert summary["time_ms"] == 2
    assert summary["event_count"] == 4
