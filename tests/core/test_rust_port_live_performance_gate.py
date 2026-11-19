from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.rust_port_persistent_benchmark import run_benchmark

pytestmark = [pytest.mark.performance, pytest.mark.slow, pytest.mark.requires_cargo]


def test_live_persistent_python_rust_performance_gate(tmp_path: Path) -> None:
    if os.environ.get("KOGWISTAR_RUN_PERFORMANCE_GATES") != "1":
        pytest.skip("set KOGWISTAR_RUN_PERFORMANCE_GATES=1 to run live measurements")

    output_root = tmp_path / "persistent-performance-gate"
    report = run_benchmark(
        output_root,
        items=100,
        read_cycles=20,
        # More fresh databases stabilize tail-latency estimates on hosted runners.
        samples=10,
        selected_modes=("python", "rust_native"),
    )
    report_path = Path(
        os.environ.get("KOGWISTAR_PERFORMANCE_REPORT", output_root / "report.v5.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert report["gate"]["status"] == "passed", json.dumps(
        {
            "gate": report["gate"],
            "ratios": report["ratios"].get("rust_native"),
            "mixed_interleaved_component_p95_over_python": report[
                "mixed_interleaved_component_p95_over_python"
            ],
            "peak_rss_rust_native_over_python": report[
                "peak_rss_rust_native_over_python"
            ],
            "report": str(report_path),
        },
        indent=2,
        sort_keys=True,
    )
    assert report["modes"]["python"]["state_digest"] == report["modes"]["rust_native"][
        "state_digest"
    ]
    assert set(report["modes"]) == {"python", "rust_native"}
    assert report["gate"]["mode"] == "rust_native"
    assert report["gate"]["status"] == "passed"
