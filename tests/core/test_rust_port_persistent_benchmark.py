from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.requires_cargo]


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "rust_port_persistent_benchmark.py"
    spec = importlib.util.spec_from_file_location("rust_port_persistent_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_harness_compares_python_and_rust_native_worker(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _module()
    report = benchmark.run_benchmark(tmp_path, items=3, read_cycles=1, samples=2)
    assert report["dataset"] == benchmark.DATASET
    assert report["modes"]["python"]["state_digest"] == report["modes"]["rust"]["state_digest"]
    assert report["modes"]["python"]["state_digest"] == report["modes"]["rust_native"]["state_digest"]
    assert report["performance_gate_mode"] == "rust_native"
    assert set(report["mixed_interleaved_component_p95_over_python"]) == {
        "rust",
        "rust_cached",
        "rust_batched",
        "rust_native",
    }
    for mode in report["modes"].values():
        assert set(mode["mixed_interleaved_components"]) == set(
            benchmark.MIXED_COMPONENTS
        )
        assert all(
            mode["mixed_interleaved_components"][operation]["operation_count"] == 6
            for operation in benchmark.MIXED_COMPONENTS
        )
    assert set(report["ratios"]) == set(report["modes"]) - {"python"}
    expected_workloads = set(benchmark.WORKLOADS)
    assert all(set(ratios) == expected_workloads for ratios in report["ratios"].values())


def test_selected_modes_skip_native_build_and_keep_comparable_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _module()
    calls: Counter[str] = Counter()

    def fake_child(mode, path, items, read_cycles, trace_path):
        calls[mode] += 1
        return {
            "mode": mode,
            "metrics": {
                **{
                    workload: {
                        "elapsed_ms": 1.0,
                        "cpu_elapsed_ms": 0.5,
                        "operations": 1,
                        "throughput_ops_per_sec": 100.0,
                        "operation_latency_ns": [1_000_000],
                    }
                    for workload in benchmark.WORKLOADS
                },
                "mixed_interleaved_components": {
                    operation: {"operation_latency_ns": [1_000_000]}
                    for operation in benchmark.MIXED_COMPONENTS
                },
            },
            "state_digest": "same-state",
            "event_count": 1,
            "projection_count": 1,
            "database_bytes": 100,
            "peak_rss_bytes": 100,
            "workload_peak_rss_delta_bytes": 10,
        }

    monkeypatch.setattr(benchmark, "write_trace", lambda path, items: [])
    monkeypatch.setattr(benchmark, "_child", fake_child)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("native Rust build should be skipped"),
    )

    report = benchmark.run_benchmark(
        tmp_path,
        items=3,
        read_cycles=1,
        samples=2,
        selected_modes=("python", "rust", "rust_batched"),
    )

    assert set(report["modes"]) == {"python", "rust", "rust_batched"}
    assert calls == Counter({"python": 3, "rust": 3, "rust_batched": 3})
    assert report["gate"]["status"] == "not_evaluated"
    assert set(report["mixed_interleaved_component_p95_over_python"]) == {
        "rust",
        "rust_batched",
    }


def test_persistent_summary_reports_operation_tails_separately_from_trace_tails() -> None:
    benchmark = _module()
    samples = []
    for elapsed_ms, offset in ((10.0, 0), (100.0, 3)):
        metrics = {
            workload: {
                "elapsed_ms": elapsed_ms,
                "cpu_elapsed_ms": elapsed_ms / 2,
                "operations": 1,
                "throughput_ops_per_sec": 100.0,
                "operation_latency_ns": [
                    (offset + value) * 1_000_000 for value in (1, 2, 3)
                ],
            }
            for workload in benchmark.WORKLOADS
        }
        metrics["mixed_interleaved_components"] = {
            "write_record": {"operation_latency_ns": [2_000_000 + offset]},
            "read_event": {"operation_latency_ns": [3_000_000 + offset]},
            "read_projection": {"operation_latency_ns": [4_000_000 + offset]},
        }
        samples.append(
            {
                "state_digest": "same",
                "metrics": metrics,
                "peak_rss_bytes": 100,
                "workload_peak_rss_delta_bytes": 10,
            }
        )

    summary = benchmark._summary(samples)
    metric = summary["metrics"]["event_append"]
    assert metric["p95_elapsed_ms"] == 100.0
    assert metric["operation_latency_ms"] == {
        "operation_count": 6,
        "p50_ms": 3.0,
        "p95_ms": 6.0,
        "p99_ms": 6.0,
        "max_ms": 6.0,
    }
    assert all(
        "operation_latency_ns" not in sample["metrics"]["event_append"]
        for sample in summary["samples"]
    )
    assert summary["mixed_interleaved_components"]["write_record"] == {
        "operation_count": 2,
        "p50_ms": 2.0,
        "p95_ms": 2.000003,
        "p99_ms": 2.000003,
        "max_ms": 2.000003,
    }


@pytest.mark.parametrize(
    "selected_modes",
    [
        ("python",),
        ("rust",),
        ("python", "rust", "rust"),
        ("python", "rust", "unknown"),
    ],
)
def test_selected_modes_reject_incomplete_or_ambiguous_comparison(
    tmp_path: Path, selected_modes: tuple[str, ...]
) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _module()
    with pytest.raises(ValueError, match="benchmark modes"):
        benchmark.run_benchmark(
            tmp_path,
            items=1,
            read_cycles=1,
            samples=2,
            selected_modes=selected_modes,
        )

    assert list(tmp_path.iterdir()) == []


def test_performance_gate_uses_only_the_selected_mode() -> None:
    benchmark = _module()
    ratios = {
        "rust": {
            "mixed_interleaved": {
                "p95_latency_over_python": 0.5,
                "throughput_over_python": 2.0,
            }
        },
        "rust_native": {
            "mixed_interleaved": {
                "p95_latency_over_python": 1.0,
                "throughput_over_python": 1.0,
            }
        },
    }

    result = benchmark.evaluate_persistent_performance_gate(
        ratios, 1.0, rust_mode="rust_native"
    )

    assert result["mode"] == "rust_native"
    assert result["status"] == "passed"
