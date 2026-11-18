from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _benchmark():
    path = ROOT / "scripts" / "rust_runtime_serving_benchmark.py"
    spec = importlib.util.spec_from_file_location("rust_runtime_serving_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _persistent_benchmark():
    path = ROOT / "scripts" / "rust_port_persistent_benchmark.py"
    spec = importlib.util.spec_from_file_location("rust_port_persistent_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("ratios", "expected_status", "expected_checks"),
    [
        ({"retry": 2.5, "read": 1.0}, "passed", {"retry": True, "read": True}),
        ({"retry": 2.5001}, "failed", {"retry": False}),
        ({"retry": float("nan")}, "failed", {"retry": False}),
        ({"retry": float("inf")}, "failed", {"retry": False}),
    ],
)
@pytest.mark.ci
def test_growth_gate_threshold_and_invalid_measurements_are_deterministic(
    ratios: dict[str, float], expected_status: str, expected_checks: dict[str, bool]
) -> None:
    gate = _benchmark().evaluate_growth_gate(ratios, max_growth_ratio=2.5)

    assert gate["status"] == expected_status
    assert gate["checks"] == expected_checks


@pytest.mark.ci
def test_growth_gate_rejects_invalid_configuration() -> None:
    benchmark = _benchmark()

    with pytest.raises(ValueError, match="finite and positive"):
        benchmark.evaluate_growth_gate({"retry": 1.0}, max_growth_ratio=float("inf"))
    with pytest.raises(ValueError, match="at least one"):
        benchmark.evaluate_growth_gate({})


@pytest.mark.parametrize(
    ("latency", "throughput", "rss", "expected_status"),
    [
        (1.10, 0.95, 1.10, "passed"),
        (1.1001, 1.0, 1.0, "failed"),
        (1.0, 0.9499, 1.0, "failed"),
        (1.0, 1.0, float("nan"), "failed"),
    ],
)
@pytest.mark.ci
def test_persistent_performance_gate_enforces_thresholds_without_benchmark(
    latency: float, throughput: float, rss: float, expected_status: str
) -> None:
    gate = _persistent_benchmark().evaluate_persistent_performance_gate(
        {
            "rust": {
                "mixed_interleaved": {
                    "p95_latency_over_python": latency,
                    "throughput_over_python": throughput,
                }
            }
        },
        rss,
    )

    assert gate["status"] == expected_status


@pytest.mark.ci
def test_persistent_performance_gate_requires_rust_workload_data() -> None:
    with pytest.raises(ValueError, match="Rust workload ratios"):
        _persistent_benchmark().evaluate_persistent_performance_gate({}, 1.0)


@pytest.mark.ci
def test_runtime_serving_benchmark_smoke_preserves_near_constant_lookup(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _benchmark()

    report = benchmark.run(tmp_path, samples=5, iterations=10)

    assert report["history_multiplier"] == 100
    assert report["gate"]["status"] == "passed", report
    assert all(report["gate"]["checks"].values())
