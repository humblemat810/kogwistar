from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = [pytest.mark.ci, pytest.mark.core]


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "rust_port_persistent_benchmark.py"
    spec = importlib.util.spec_from_file_location("rust_port_persistent_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persistent_benchmark_compares_equivalent_python_and_rust_state(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    report = _module().run_benchmark(tmp_path, items=3, read_cycles=1, samples=2)
    assert report["dataset"] == "rust-port-persistent-sqlite-v1"
    assert report["modes"]["python"]["state_digest"] == report["modes"]["rust"]["state_digest"]
    assert set(report["ratios"]) == set(report["modes"]["python"]["metrics"])
