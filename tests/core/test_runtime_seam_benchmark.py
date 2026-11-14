from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "runtime_seam_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("runtime_seam_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_runtime_seam_modes_share_topology_and_expected_final_state() -> None:
    python = _MODULE._workflow_payload(mode="python_worker", run_id="r1", nodes=3)
    native = _MODULE._workflow_payload(mode="rust_inline", run_id="r2", nodes=3)

    assert python["start_node_id"] == native["start_node_id"] == "step-00000"
    assert python["runtime_routes"] == native["runtime_routes"]
    assert set(python["node_ops"].values()) == {"benchmark.increment"}
    assert set(native["node_ops"].values()) == {"increment-counter"}
    assert set(python["node_executor_targets"].values()) == {"python-worker"}
    assert set(native["node_executor_targets"].values()) == {"rust-native"}
    assert _MODULE._digest({"counter": 3}) == _MODULE._digest({"counter": 3})


def test_runtime_seam_summary_reports_median_and_tail_sample() -> None:
    summary = _MODULE._summary([3.0, 1.0, 2.0])

    assert summary == {"median": 2.0, "p95": 3.0, "min": 1.0, "max": 3.0}


def test_runtime_seam_requires_exact_cpu_and_memory_limits() -> None:
    assert _MODULE._verify_limits("200000 100000", "2147483648", 2.0, 2 * 1024**3)
    assert not _MODULE._verify_limits("max 100000", "2147483648", 2.0, 2 * 1024**3)
    assert not _MODULE._verify_limits("200000 100000", None, 2.0, 2 * 1024**3)
