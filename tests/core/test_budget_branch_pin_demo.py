from __future__ import annotations

import pytest

from kogwistar.demo.budget_branch_pin_demo import run_budget_branch_pin_demo


pytestmark = [pytest.mark.core]


@pytest.mark.ci
def test_budget_branch_pin_demo_smoke() -> None:
    result = run_budget_branch_pin_demo()

    assert result["resume"]["first"] == "suspended"
    assert result["resume"]["second"] == "succeeded"
    assert result["result"]["branch_pinned_until_refresh"] is True
    assert result["result"]["resume_blocked_before_refresh"] is True
    assert result["result"]["heavy_after_light"] is True
    assert any(item["event"] == "heavy.2.paused" for item in result["timeline"])
    assert any(item["event"] == "token_window.refreshed" for item in result["timeline"])
