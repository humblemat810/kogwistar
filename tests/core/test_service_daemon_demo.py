from __future__ import annotations

from kogwistar.demo.operator_views_demo import run_operator_views_demo
from kogwistar.demo.service_daemon_demo import run_service_daemon_demo


def test_service_daemon_demo_smoke() -> None:
    result = run_service_daemon_demo()
    assert result["service_id"] == "svc.demo"
    assert "service" in result["process_kinds"]
    assert result["health"]


def test_operator_views_demo_smoke() -> None:
    result = run_operator_views_demo()
    assert result["process_table"]
    assert result["dashboard"]["process_table"]
    assert result["dashboard"]["resources"]["services"]["total_services"] >= 1

