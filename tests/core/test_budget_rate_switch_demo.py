from __future__ import annotations

import pytest

from kogwistar.demo.budget_rate_switch_demo import run_budget_rate_switch_demo


pytestmark = [pytest.mark.core]


@pytest.mark.ci
def test_budget_rate_switch_demo_smoke() -> None:
    result = run_budget_rate_switch_demo()

    assert result["submissions"]["heavy"]["admission"] == "accepted"
    assert result["submissions"]["tiny"]["admission"] == "accepted"
    assert result["submissions"]["free"]["admission"] == "accepted"
    assert result["submissions"]["heavy_resume"]["admission"] == "accepted"
    assert result["order"] == ["tiny", "free", "heavy"]
    assert result["result"]["tiny_before_refresh"] is True
    assert result["result"]["free_before_refresh"] is True
    assert result["result"]["heavy_resumed_after_refresh"] is True
    assert any(item["event"] == "heavy.rate_blocked" for item in result["timeline"])
    assert any(item["event"] == "token_window.refreshed" for item in result["timeline"])
