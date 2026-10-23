"""Verify parity and runtime/limit evidence for the fixed-resource matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RUNTIMES = ("cpython314-gil", "cpython315-gil", "cpython315-nogil", "pypy311")
SIZES = (1000, 2000)
MIXES = ("balanced", "read_heavy")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _runtime_evidence_matches(runtime: str, report: dict[str, Any]) -> bool:
    info = report.get("runtime")
    if not isinstance(info, dict):
        return False
    implementation = info.get("implementation")
    version = info.get("version")
    gil_before = info.get("gil_enabled_before_benchmark_import")
    gil_after = info.get("gil_enabled_after_benchmark")
    if runtime == "cpython314-gil":
        identity_matches = (
            implementation == "CPython"
            and isinstance(version, str)
            and version.startswith("3.14.")
            and gil_before is True
            and gil_after is True
        )
    elif runtime == "cpython315-gil":
        identity_matches = (
            implementation == "CPython"
            and isinstance(version, str)
            and version.startswith("3.15.")
            and gil_before is True
            and gil_after is True
        )
    elif runtime == "cpython315-nogil":
        identity_matches = (
            implementation == "CPython"
            and isinstance(version, str)
            and version.startswith("3.15.")
            and gil_before is False
            and gil_after is False
        )
    else:
        identity_matches = (
            implementation == "PyPy"
            and isinstance(version, str)
            and version.startswith("3.11.")
        )
    cpu_max = info.get("cgroup_cpu_max")
    memory_max = info.get("cgroup_memory_max")
    try:
        quota, period = str(cpu_max).split()
        cpu_matches = int(quota) / int(period) == float(info["expected_cpus"])
        memory_matches = int(str(memory_max)) == int(info["expected_memory_bytes"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return identity_matches and cpu_matches and memory_matches


def verify_matrix(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    runtime_reports: dict[str, dict[str, Any]] = {}
    for runtime in RUNTIMES:
        path = root / runtime / "runtime-report.json"
        try:
            report = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{runtime}: cannot read runtime report ({exc})")
            continue
        runtime_reports[runtime] = report
        if report.get("status") != "complete":
            errors.append(f"{runtime}: report status is {report.get('status')!r}")
        if report.get("runtime_verified") is not True:
            errors.append(f"{runtime}: interpreter/GIL identity was not verified")
        if report.get("limits_verified") is not True:
            errors.append(f"{runtime}: fixed cgroup limits were not verified")
        if report.get("expected_runtime") != runtime or not _runtime_evidence_matches(runtime, report):
            errors.append(f"{runtime}: recorded interpreter/GIL or cgroup evidence does not match")

    graph_digests: dict[str, dict[str, str]] = {}
    trace_digests: dict[str, dict[str, str]] = {}
    for runtime, report in runtime_reports.items():
        child_reports = report.get("container_reports")
        if not isinstance(child_reports, list):
            errors.append(f"{runtime}: container reports are missing")
            continue
        for size in SIZES:
            matches = [item for item in child_reports if item.get("items") == size]
            if len(matches) != 1:
                errors.append(f"{runtime}: expected one child report for {size} items")
                continue
            child = matches[0]
            backend = child.get("backends", {}).get("sqlite_event_projection", {})
            if child.get("status") != "complete":
                errors.append(f"{runtime}/{size}: child benchmark did not complete")
                continue
            for mix in MIXES:
                result = backend.get("mixes", {}).get(mix)
                if not isinstance(result, dict):
                    errors.append(f"{runtime}/{size}/{mix}: mix report is missing")
                    continue
                mode = result.get("mode_results", {}).get("python", {})
                expected_samples = report.get("samples_per_mix")
                samples = mode.get("samples")
                if result.get("state_digest_equal") is not True:
                    errors.append(f"{runtime}/{size}/{mix}: final state did not match expected digest")
                if not isinstance(samples, list) or len(samples) != expected_samples:
                    actual_count = len(samples) if isinstance(samples, list) else "missing"
                    errors.append(
                        f"{runtime}/{size}/{mix}: expected {expected_samples} samples, got {actual_count}"
                    )
                state_digest = mode.get("state_digest")
                trace_digest = result.get("trace_sha256")
                if not isinstance(state_digest, str) or not isinstance(trace_digest, str):
                    errors.append(f"{runtime}/{size}/{mix}: digests are missing")
                    continue
                trace_path_value = result.get("trace_path")
                if not isinstance(trace_path_value, str):
                    errors.append(f"{runtime}/{size}/{mix}: exported trace path is missing")
                    continue
                trace_path = (root / runtime / trace_path_value).resolve()
                if not trace_path.is_relative_to((root / runtime).resolve()) or not trace_path.is_file():
                    errors.append(f"{runtime}/{size}/{mix}: exported trace file is missing or escapes output")
                    continue
                actual_trace_digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                if actual_trace_digest != trace_digest:
                    errors.append(f"{runtime}/{size}/{mix}: exported trace digest does not match report")
                    continue
                key = f"{size}/{mix}"
                graph_digests.setdefault(key, {})[runtime] = state_digest
                trace_digests.setdefault(key, {})[runtime] = trace_digest

    for key, values in graph_digests.items():
        if len(values) != len(RUNTIMES) or len(set(values.values())) != 1:
            errors.append(f"{key}: final graph-state digests differ or are incomplete")
    for key, values in trace_digests.items():
        if len(values) != len(RUNTIMES) or len(set(values.values())) != 1:
            errors.append(f"{key}: operation-trace digests differ or are incomplete")

    packages = {
        runtime: report.get("runtime", {}).get("packages", {})
        for runtime, report in runtime_reports.items()
    }
    all_package_names = sorted({name for listing in packages.values() for name in listing})
    package_differences = {
        name: {
            runtime: packages[runtime].get(name, "<missing>")
            for runtime in packages
        }
        for name in all_package_names
        if len({packages[runtime].get(name, "<missing>") for runtime in packages}) > 1
    }
    return {
        "schema": "kogwistar-fixed-resource-runtime-verification/v1",
        "status": "complete" if not errors else "failed",
        "runtimes": list(runtime_reports),
        "state_digest_parity": graph_digests,
        "trace_digest_parity": trace_digests,
        "dependency_versions_equal": not package_differences,
        "dependency_version_differences": package_differences,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_matrix(args.root)
    output = args.root / "runtime-matrix-verification.json"
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
