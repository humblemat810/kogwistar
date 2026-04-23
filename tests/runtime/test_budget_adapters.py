from __future__ import annotations

import pytest

from kogwistar.runtime.budget_adapters import adapt_budget_events

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
