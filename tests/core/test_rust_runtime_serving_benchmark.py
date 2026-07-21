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


@pytest.mark.ci
def test_runtime_serving_benchmark_smoke_preserves_near_constant_lookup(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    benchmark = _benchmark()

    report = benchmark.run(tmp_path, samples=2, iterations=3)

    assert report["history_multiplier"] == 100
    assert report["gate"]["status"] == "passed"
    assert all(report["gate"]["checks"].values())
