from __future__ import annotations

from kogwistar.demo.recovery_repair_demo import run_recovery_repair_demo


def test_recovery_repair_demo_smoke() -> None:
    result = run_recovery_repair_demo()
    assert result["repair"]["service_id"] == "svc.repair.demo"
    assert "dashboard" in result
    assert "dead_letters" in result

