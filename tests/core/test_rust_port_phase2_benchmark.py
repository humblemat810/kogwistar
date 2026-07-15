from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rust_port_phase2_benchmark import (
    DATASET_VERSION,
    OPERATIONS,
    THRESHOLDS,
    evaluate_gate,
    run_phase2_benchmark,
)


pytestmark = [pytest.mark.ci, pytest.mark.core]
ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "contracts" / "benchmarks" / "rust-memory-phase2-windows.json"


def _assert_report(report: dict) -> None:
    assert report["benchmark_version"] == 1
    assert report["dataset"] == DATASET_VERSION
    assert report["operations_per_cycle"] == list(OPERATIONS)
    assert report["database_load"] == {"kind": "in-memory", "external_queries": 0}
    assert report["authoritative_writes"] == 0
    python = report["modes"]["python"]
    rust = report["modes"]["rust"]
    assert python["output_digest"] == rust["output_digest"]
    assert python["output_count_per_sample"] == rust["output_count_per_sample"]
    for mode in (python, rust):
        assert mode["external_db_queries"] == 0
        assert mode["authoritative_writes"] == 0
        assert mode["p95_cycle_latency_ms"] > 0
        assert mode["throughput_ops_per_sec"] > 0
        assert mode["peak_rss_bytes"] > 0
        assert len(mode["samples"]) == report["samples_per_mode"]
    gate = report["gate"]
    assert gate["thresholds"] == THRESHOLDS
    assert gate["status"] in {"passed", "recorded"}
    if gate["status"] == "passed":
        assert all(gate["checks"].values())
        assert "exception" not in gate
    else:
        assert gate["exception"]["failed_thresholds"]
        assert gate["exception"]["reason"]
        assert any(not value for value in gate["checks"].values())


def test_committed_phase2_report_has_digest_parity_and_complete_gate() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    _assert_report(report)


def test_threshold_math_is_exact_for_pass_and_recorded_exception() -> None:
    python = {
        "p95_cycle_latency_ms": 100.0,
        "throughput_ops_per_sec": 100.0,
        "peak_rss_bytes": 100,
    }
    passed = evaluate_gate(
        python,
        {
            "p95_cycle_latency_ms": 110.0,
            "throughput_ops_per_sec": 95.0,
            "peak_rss_bytes": 110,
        },
    )
    assert passed["status"] == "passed"
    recorded = evaluate_gate(
        python,
        {
            "p95_cycle_latency_ms": 110.1,
            "throughput_ops_per_sec": 94.9,
            "peak_rss_bytes": 110,
        },
    )
    assert recorded["status"] == "recorded"
    assert recorded["exception"]["failed_thresholds"] == ["p95_latency", "throughput"]


def test_phase2_benchmark_small_smoke_has_no_performance_assertion() -> None:
    report = run_phase2_benchmark(items=8, warmups=0, samples=2, cycles=1)
    _assert_report(report)
