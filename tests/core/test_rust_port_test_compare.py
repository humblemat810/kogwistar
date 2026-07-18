from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.regression


def _comparer():
    path = ROOT / "scripts" / "rust_port_test_compare.py"
    spec = importlib.util.spec_from_file_location("rust_port_test_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(*, groups: list[str], duration: float = 2.0) -> dict[str, object]:
    return {
        "status": "passed",
        "execution": {"container_orchestrator_sha256": "orchestrator"},
        "candidate": {
            "candidate_source_sha256": "source",
            "verification_harness_sha256": "harness",
            "test_profile": "feature",
        },
        "layers": [
            {
                "name": "core",
                "duration_seconds": duration,
                "runs": [{"group_key": key} for key in groups],
            }
        ],
    }


def test_compare_requires_equal_coverage_and_reports_speedup() -> None:
    comparer = _comparer()

    result = comparer.compare(
        _report(groups=["a", "b"], duration=8.0),
        _report(groups=["b", "a"], duration=4.0),
    )

    assert result["coverage_equal"] is True
    assert result["wall_speedup"] == 2.0
    assert result["layers"] == {"core": {"group_count": 2}}

    serial = _report(groups=["a"], duration=100.0)
    parallel = _report(groups=["a"], duration=100.0)
    serial["execution"] = {"orchestrator_wall_seconds": 12.0}
    parallel["execution"] = {"orchestrator_wall_seconds": 4.0}
    assert comparer.compare(serial, parallel)["wall_speedup"] == 3.0

    with pytest.raises(ValueError, match="coverage differs"):
        comparer.compare(_report(groups=["a"]), _report(groups=["b"]))


def test_compare_rejects_duplicate_groups_and_candidate_drift() -> None:
    comparer = _comparer()

    with pytest.raises(ValueError, match="duplicate group keys"):
        comparer.compare(_report(groups=["a", "a"]), _report(groups=["a"]))

    parallel = _report(groups=["a"])
    parallel["candidate"]["candidate_source_sha256"] = "different"
    with pytest.raises(ValueError, match="candidate_source_sha256 differs"):
        comparer.compare(_report(groups=["a"]), parallel)

    parallel = _report(groups=["a"])
    parallel["execution"]["container_orchestrator_sha256"] = "different"
    with pytest.raises(ValueError, match="orchestrator identity differs"):
        comparer.compare(_report(groups=["a"]), parallel)


def test_compare_selected_layer_uses_layer_elapsed_not_outer_wall() -> None:
    comparer = _comparer()
    serial = _report(groups=["a"], duration=12.0)
    parallel = _report(groups=["a"], duration=4.0)
    serial["execution"]["orchestrator_wall_seconds"] = 100.0
    parallel["execution"]["orchestrator_wall_seconds"] = 200.0

    result = comparer.compare(serial, parallel, selected_layers=["core"])

    assert result["serial_duration_seconds"] == 12.0
    assert result["parallel_duration_seconds"] == 4.0
    assert result["wall_speedup"] == 3.0
