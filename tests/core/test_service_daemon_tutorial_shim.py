from __future__ import annotations

from kogwistar.demo.operator_views_demo import run_operator_views_demo
from kogwistar.demo.service_daemon_demo import run_service_daemon_demo


def test_service_daemon_tutorial_shim_smoke() -> None:
    service_demo = run_service_daemon_demo()
    operator_demo = run_operator_views_demo()

    assert service_demo["service_id"] == "svc.demo"
    assert service_demo["dashboard"]["process_table"]
    assert operator_demo["dashboard"]["process_table"]
    assert operator_demo["dashboard"]["resources"]["services"]["total_services"] >= 1

