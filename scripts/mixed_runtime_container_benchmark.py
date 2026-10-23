"""Run a concurrent trace and record runtime/container facts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

def _gil_enabled() -> bool | None:
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if callable(probe) else None


def _cgroup_value(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    if value:
        return value if value != "max" else None

    if name == "cpu.max":
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        try:
            quota = quota_path.read_text(encoding="ascii").strip()
            period = period_path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return f"{quota} {period}" if quota and period else None

    if name == "memory.max":
        path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        try:
            value = path.read_text(encoding="ascii").strip()
        except OSError:
            return None
        return value or None

    return None


def _installed_packages() -> dict[str, str]:
    return {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }


def _write_report(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_runtime(
    expected: str,
    implementation: str,
    version: str,
    gil_enabled_before: bool | None,
    gil_enabled_after: bool | None,
) -> bool:
    if expected == "cpython314-gil":
        return (
            implementation == "CPython"
            and version.startswith("3.14.")
            and gil_enabled_before is True
            and gil_enabled_after is True
        )
    if expected in {"cpython315-nogil", "cpython315-nogil-rust"}:
        return (
            implementation == "CPython"
            and version == "3.15.1"
            and gil_enabled_before is False
            and gil_enabled_after is False
        )
    if expected == "cpython315-gil":
        return (
            implementation == "CPython"
            and version == "3.15.1"
            and gil_enabled_before is True
            and gil_enabled_after is True
        )
    if expected == "pypy311":
        return implementation == "PyPy" and version.startswith("3.11.")
    return False


def _verify_cgroup_limits(cpu_max: str | None, memory_max: str | None, cpus: float, memory: int) -> bool:
    if not cpu_max or not memory_max:
        return False
    quota_text, separator, period_text = cpu_max.partition(" ")
    try:
        cpu_actual = int(quota_text) / int(period_text) if separator else None
        memory_actual = int(memory_max)
    except ValueError:
        return False
    return cpu_actual == cpus and memory_actual == memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, action="append", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--expected-runtime", required=True)
    parser.add_argument("--authority", choices=("python", "rust"), default="python")
    parser.add_argument("--expected-cpus", type=float, required=True)
    parser.add_argument("--expected-memory-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.samples < 2 or not args.items or any(item < 1 for item in args.items):
        parser.error("provide positive --items values and at least two samples")

    runtime_before_import = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "gil_enabled_before_benchmark_import": _gil_enabled(),
    }
    if args.authority == "rust":
        if args.expected_runtime != "cpython315-nogil-rust":
            parser.error("Rust authority is currently supported only for the no-GIL comparison")
        from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite as _RustEngineSQLite

        del _RustEngineSQLite
    gil_after_native_import = _gil_enabled()
    from mixed_concurrent_benchmark import run_benchmark

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "runtime-report.json"
    result: dict[str, Any] = {
        "schema": "kogwistar-fixed-resource-runtime-report/v1",
        "status": "running",
        "expected_runtime": args.expected_runtime,
        "authority": args.authority,
        "runtime_verified": False,
        "limits_verified": False,
        "runtime_before_import": runtime_before_import,
        "items": args.items,
        "samples_per_mix": args.samples,
        "container_reports": [],
        "active_size": None,
    }
    _write_report(output_path, result)
    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kogwistar-runtime-benchmark-") as work_root:
        for size in args.items:
            report_dir = Path(work_root) / str(size)
            result["active_size"] = size
            _write_report(output_path, result)
            try:
                child_report = run_benchmark(
                    report_dir,
                    items=size,
                    samples=args.samples,
                    modes=(args.authority,),
                    backends=("sqlite_event_projection",),
                )
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                _write_report(output_path, result)
                raise
            export_dir = args.output / str(size)
            export_dir.mkdir(parents=True, exist_ok=True)
            for mix in ("balanced", "read_heavy"):
                trace_name = f"concurrent_{mix}.trace.v3.json"
                shutil.copyfile(report_dir / trace_name, export_dir / trace_name)
                child_report["backends"]["sqlite_event_projection"]["mixes"][mix][
                    "trace_path"
                ] = f"{size}/{trace_name}"
            reports.append(child_report)
            result["container_reports"] = reports
            result["active_size"] = None
            _write_report(output_path, result)

    gil_after_benchmark = _gil_enabled()
    actual_implementation = platform.python_implementation()
    expected = args.expected_runtime
    runtime_verified = _verify_runtime(
        expected,
        actual_implementation,
        platform.python_version(),
        runtime_before_import["gil_enabled_before_benchmark_import"],
        gil_after_benchmark,
    )
    limits_verified = _verify_cgroup_limits(
        _cgroup_value("cpu.max"),
        _cgroup_value("memory.max"),
        args.expected_cpus,
        args.expected_memory_bytes,
    )
    result.update({
        "status": "complete" if runtime_verified and limits_verified else "invalid_runtime_or_limits",
        "expected_runtime": expected,
        "runtime_verified": runtime_verified,
        "limits_verified": limits_verified,
        "runtime": {
            **runtime_before_import,
            "gil_enabled_after_benchmark": gil_after_benchmark,
            "gil_enabled_after_native_import": gil_after_native_import,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "expected_cpus": args.expected_cpus,
            "expected_memory_bytes": args.expected_memory_bytes,
            "cgroup_cpu_max": _cgroup_value("cpu.max"),
            "cgroup_memory_max": _cgroup_value("memory.max"),
            "packages": _installed_packages(),
            "pip_freeze": subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines(),
        },
        "items": args.items,
        "samples_per_mix": args.samples,
        "container_reports": reports,
        "active_size": None,
    })
    _write_report(output_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return (
        0
        if runtime_verified
        and limits_verified
        and all(row["status"] == "complete" for row in reports)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
