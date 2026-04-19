from __future__ import annotations

import pytest

from kogwistar.demo.scheduler_priority_demo import run_scheduler_priority_demo

pytestmark = [pytest.mark.core]


@pytest.mark.ci
def test_scheduler_priority_demo_smoke() -> None:
    result = run_scheduler_priority_demo()

    assert result["scheduler"]["max_active"] == 1
    assert result["scheduler"]["cooperative_pause_required"] is True
    assert result["submissions"]["low"]["admission"] == "accepted"
    assert result["submissions"]["high"]["admission"] == "accepted"
    assert result["submissions"]["low_resume"]["admission"] == "accepted"
    assert result["order"] == ["high", "low"]
    assert result["result"]["high_finished_before_low"] is True
    assert result["result"]["low_blocked_before_high_started"] is True
