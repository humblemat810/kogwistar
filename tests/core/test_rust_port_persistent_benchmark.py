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


def test_persistent_benchmark_compares_equivalent_python_and_rust_state(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _module()
    report = benchmark.run_benchmark(tmp_path, items=3, read_cycles=1, samples=2)
    assert report["dataset"] == benchmark.DATASET
    assert report["modes"]["python"]["state_digest"] == report["modes"]["rust"]["state_digest"]
    assert set(report["ratios"]) == set(report["modes"]) - {"python"}
    expected_workloads = set(report["modes"]["python"]["metrics"])
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
                workload: {
                    "elapsed_ms": 1.0,
                    "cpu_elapsed_ms": 0.5,
                    "operations": 1,
                    "throughput_ops_per_sec": 100.0,
                }
                for workload in benchmark.WORKLOADS
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
    assert report["gate"]["status"] == "passed"


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
