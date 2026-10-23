from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from typing import Any


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_mixed_runtime_matrix.py"
_SPEC = importlib.util.spec_from_file_location("verify_mixed_runtime_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_runtime_report(root: Path, runtime: str, *, corrupt_graph_digest: bool = False) -> None:
    if runtime == "cpython314-gil":
        implementation, version, gil = "CPython", "3.14.9", True
    elif runtime == "cpython315-gil":
        implementation, version, gil = "CPython", "3.15.1", True
    elif runtime in {"cpython315-nogil", "cpython315-nogil-rust"}:
        implementation, version, gil = "CPython", "3.15.1", False
    else:
        implementation, version, gil = "PyPy", "3.11.13", None
    child_reports: list[dict[str, Any]] = []
    for size in (1000, 2000):
        mixes = {}
        for mix in ("balanced", "read_heavy"):
            mixes[mix] = {
                "state_digest_equal": True,
                "trace_path": f"{size}/concurrent_{mix}.trace.v3.json",
                "trace_sha256": f"trace-{size}-{mix}",
                "mode_results": {
                    ("rust" if runtime == "cpython315-nogil-rust" else "python"): {
                        "state_digest": f"graph-{size}-{mix}-{runtime if corrupt_graph_digest else 'same'}",
                        "samples": [{"sample": index} for index in range(5)],
                    }
                },
            }
        child_reports.append({
            "items": size,
            "status": "complete",
            "backends": {
                "sqlite_event_projection": {
                    "mixes": mixes,
                }
            },
        })
        for mix in ("balanced", "read_heavy"):
            trace_path = root / runtime / str(size) / f"concurrent_{mix}.trace.v3.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_body = f"trace-{size}-{mix}".encode()
            trace_path.write_bytes(trace_body)
        for mix in ("balanced", "read_heavy"):
            mixes[mix]["trace_sha256"] = hashlib.sha256(
                (root / runtime / str(size) / f"concurrent_{mix}.trace.v3.json").read_bytes()
            ).hexdigest()
    report = {
        "expected_runtime": runtime,
        "authority": "rust" if runtime == "cpython315-nogil-rust" else "python",
        "status": "complete",
        "runtime_verified": True,
        "limits_verified": True,
        "samples_per_mix": 5,
        "runtime": {
            "implementation": implementation,
            "version": version,
            "gil_enabled_before_benchmark_import": gil,
            "gil_enabled_after_benchmark": gil,
            "gil_enabled_after_native_import": gil,
            "expected_cpus": 4,
            "expected_memory_bytes": 8 * 1024**3,
            "cgroup_cpu_max": "400000 100000",
            "cgroup_memory_max": str(8 * 1024**3),
            "packages": {"pytest": "9.1.1"},
        },
        "container_reports": child_reports,
    }
    target = root / runtime / "runtime-report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report), encoding="utf-8")


def test_runtime_matrix_verifier_requires_cross_runtime_graph_and_trace_parity(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "complete"
    assert report["dependency_versions_equal"] is True
    assert not report["errors"]


def test_runtime_matrix_verifier_rejects_graph_state_divergence(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(
            tmp_path,
            runtime,
            corrupt_graph_digest=runtime == "pypy311",
        )

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("final graph-state digests differ" in error for error in report["errors"])


def test_runtime_matrix_verifier_rejects_reenabled_gil(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)
    target = tmp_path / "cpython315-nogil" / "runtime-report.json"
    report_data = json.loads(target.read_text(encoding="utf-8"))
    report_data["runtime"]["gil_enabled_after_benchmark"] = True
    target.write_text(json.dumps(report_data), encoding="utf-8")

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("interpreter/GIL or cgroup evidence" in error for error in report["errors"])


def test_runtime_matrix_verifier_rejects_rust_import_that_enables_gil(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)
    target = tmp_path / "cpython315-nogil-rust" / "runtime-report.json"
    report_data = json.loads(target.read_text(encoding="utf-8"))
    report_data["runtime"]["gil_enabled_after_native_import"] = True
    target.write_text(json.dumps(report_data), encoding="utf-8")

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("importing Rust enabled" in error for error in report["errors"])


def test_runtime_matrix_verifier_requires_cpython314_gil(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)
    target = tmp_path / "cpython314-gil" / "runtime-report.json"
    report_data = json.loads(target.read_text(encoding="utf-8"))
    report_data["runtime"]["gil_enabled_after_benchmark"] = False
    target.write_text(json.dumps(report_data), encoding="utf-8")

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("cpython314-gil" in error for error in report["errors"])


def test_runtime_matrix_verifier_checks_exported_trace_bytes(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)
    trace = tmp_path / "pypy311" / "1000" / "concurrent_balanced.trace.v3.json"
    trace.write_text("modified after the report", encoding="utf-8")

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("exported trace digest does not match" in error for error in report["errors"])


def test_runtime_matrix_verifier_requires_expected_state_parity_and_all_samples(tmp_path: Path) -> None:
    for runtime in _MODULE.RUNTIMES:
        _write_runtime_report(tmp_path, runtime)
    target = tmp_path / "pypy311" / "runtime-report.json"
    report_data = json.loads(target.read_text(encoding="utf-8"))
    balanced = report_data["container_reports"][0]["backends"]["sqlite_event_projection"]["mixes"]["balanced"]
    balanced["state_digest_equal"] = False
    balanced["mode_results"]["python"]["samples"].pop()
    target.write_text(json.dumps(report_data), encoding="utf-8")

    report = _MODULE.verify_matrix(tmp_path)

    assert report["status"] == "failed"
    assert any("final state did not match expected digest" in error for error in report["errors"])
    assert any("expected 5 samples, got 4" in error for error in report["errors"])
