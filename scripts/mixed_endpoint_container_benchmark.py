"""Compare Python, mixed-authority, and Rust-native MCP paths under one limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .mixed_mcp_benchmark import run_benchmark
    from .mixed_runtime_container_benchmark import (
        _cgroup_value,
        _installed_packages,
        _verify_cgroup_limits,
    )
except ImportError:
    from mixed_mcp_benchmark import run_benchmark
    from mixed_runtime_container_benchmark import (
        _cgroup_value,
        _installed_packages,
        _verify_cgroup_limits,
    )

ENDPOINT_MODES = ("http", "rust_authority_http", "rust_native")
MIN_RUST_NATIVE_THROUGHPUT_RATIO = 0.95
MAX_RUST_NATIVE_P95_LATENCY_RATIO = 1.10


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_json_artifacts(source: Path, destination: Path) -> None:
    """Copy inspectable reports/traces, never SQLite databases, to result mount."""
    for artifact in source.rglob("*.json"):
        target = destination / artifact.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)


def _verify_endpoint_reports(
    reports: list[dict[str, Any]],
    *,
    output: Path,
    items: list[int],
    samples: int,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    latency_summary: dict[str, dict[str, dict[str, float]]] = {}
    throughput_summary: dict[str, dict[str, float]] = {}
    if len(reports) != len(items):
        errors.append("report count does not match requested item sizes")

    for size, report in zip(items, reports, strict=False):
        if report.get("items") != size:
            errors.append(f"{size}: report item count differs")
        if (
            report.get("samples_per_mode") != samples
            or report.get("warmup_samples_per_mode") != 1
        ):
            errors.append(f"{size}: sample or warm-up count differs")
        run_id = report.get("run_id")
        trace_sha = report.get("trace_sha256")
        digest_by_mode: dict[str, str] = {}
        report_modes = report.get("modes")
        if not isinstance(report_modes, dict) or set(report_modes) != set(modes):
            errors.append(f"{size}: mode set differs from requested modes")
            continue
        if not isinstance(run_id, str) or not isinstance(trace_sha, str):
            errors.append(f"{size}: run or trace identity is missing")
            continue

        trace_path = output / str(size) / run_id / "mcp_trace.v1.json"
        try:
            trace_bytes = trace_path.read_bytes()
            trace = json.loads(trace_bytes)
            operations = trace.get("operations") if isinstance(trace, dict) else None
            if (
                not isinstance(trace, dict)
                or not isinstance(operations, list)
                or trace.get("schema") != "kogwistar-mixed-rewrite-trace/v2"
                or trace.get("profile") != "read_after_write"
                or len(operations) != size * 3
                or any(not isinstance(row, dict) for row in operations)
                or [row.get("operation") for row in operations]
                != ["write_record", "read_event", "read_projection"] * size
                or hashlib.sha256(trace_bytes).hexdigest() != trace_sha
            ):
                errors.append(f"{size}: trace content or digest is invalid")
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            errors.append(f"{size}: copied canonical trace is missing or malformed")
            continue

        for mode in modes:
            result = report_modes[mode]
            if not isinstance(result, dict):
                errors.append(f"{size}/{mode}: mode report is not an object")
                continue
            digest = result.get("state_digest")
            mode_samples = result.get("samples")
            warmup = result.get("warmup")
            if (
                not isinstance(digest, str)
                or not isinstance(mode_samples, list)
                or len(mode_samples) != samples
            ):
                errors.append(f"{size}/{mode}: digest or measured samples are missing")
                continue
            digest_by_mode[mode] = digest
            if not isinstance(warmup, dict) or warmup.get("state_digest") != digest:
                errors.append(f"{size}/{mode}: warm-up digest differs")
            variation = result.get("elapsed_sample_variation")
            if not isinstance(variation, dict) or not isinstance(
                variation.get("coefficient_of_variation_pct"), (int, float)
            ):
                errors.append(f"{size}/{mode}: sample variation is missing")

            operation_metrics: dict[str, list[dict[str, float]]] = {
                name: [] for name in ("write_record", "read_event", "read_projection")
            }
            measured_metrics: list[dict[str, float]] = []
            samples_to_check = (
                [warmup, *mode_samples] if isinstance(warmup, dict) else mode_samples
            )
            for sample in samples_to_check:
                if not isinstance(sample, dict):
                    errors.append(f"{size}/{mode}: sample is not an object")
                    continue
                identity = sample.get("trace_join")
                if (
                    sample.get("run_id") != run_id
                    or sample.get("trace_sha256") != trace_sha
                    or sample.get("state_digest") != digest
                    or sample.get("operations") != size * 3
                    or not isinstance(identity, dict)
                    or identity.get("run_id") != run_id
                    or identity.get("trace_sha256") != trace_sha
                    or identity.get("expected_operations") != size * 3
                    or identity.get("mode") != mode
                ):
                    errors.append(
                        f"{size}/{mode}: sample identity or operation total differs"
                    )
                elapsed = sample.get("elapsed_ms")
                startup = sample.get("startup_ms")
                cpu = sample.get("host_process_cpu_elapsed_ms")
                throughput = sample.get("throughput_ops_per_sec")
                rss_delta = sample.get("host_process_rss_delta_bytes")
                if (
                    not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in (elapsed, startup, cpu, throughput, rss_delta)
                    )
                    or elapsed <= 0
                    or startup < 0
                    or cpu < 0
                    or throughput <= 0
                ):
                    errors.append(
                        f"{size}/{mode}: resource or timing metrics are invalid"
                    )
                if sample is not warmup and all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (elapsed, startup, cpu, throughput, rss_delta)
                ):
                    measured_metrics.append(
                        {
                            "elapsed_ms": float(elapsed),
                            "startup_ms": float(startup),
                            "cpu_ms": float(cpu),
                            "throughput_ops_per_sec": float(throughput),
                            "rss_delta_bytes": float(rss_delta),
                        }
                    )
                metrics = sample.get("per_operation")
                if not isinstance(metrics, dict) or set(metrics) != set(
                    operation_metrics
                ):
                    errors.append(f"{size}/{mode}: per-operation metrics are missing")
                    continue
                for operation, values in operation_metrics.items():
                    row = metrics.get(operation)
                    if not isinstance(row, dict) or row.get("count") != size:
                        errors.append(
                            f"{size}/{mode}/{operation}: operation count differs"
                        )
                        continue
                    percentiles = [
                        row.get(f"p{percentile}_ms") for percentile in (50, 95, 99)
                    ]
                    if not all(
                        isinstance(value, (int, float))
                        and math.isfinite(value)
                        and value >= 0
                        for value in percentiles
                    ) or percentiles != sorted(percentiles):
                        errors.append(
                            f"{size}/{mode}/{operation}: latency percentiles are invalid"
                        )
                        continue
                    if sample is not warmup:
                        values.append(
                            dict(zip(("p50", "p95", "p99"), percentiles, strict=True))
                        )
                seams = sample.get("seam_metrics")
                if (
                    not isinstance(seams, dict)
                    or seams.get("tool_dispatches") != size * 3
                    or seams.get("transport_requests") != size * 3
                    or not isinstance(seams.get("ffi_calls"), int)
                ):
                    errors.append(f"{size}/{mode}: seam counters are invalid")
                elif mode in {"http", "rust_native"} and seams["ffi_calls"] != 0:
                    errors.append(f"{size}/{mode}: pure mode reports FFI calls")

            reported_metrics = {
                "median_elapsed_ms": "elapsed_ms",
                "median_startup_ms": "startup_ms",
                "median_host_process_cpu_elapsed_ms": "cpu_ms",
                "median_host_process_rss_delta_bytes": "rss_delta_bytes",
                "median_throughput_ops_per_sec": "throughput_ops_per_sec",
            }
            for report_key, sample_key in reported_metrics.items():
                expected = (
                    statistics.median(row[sample_key] for row in measured_metrics)
                    if measured_metrics
                    else None
                )
                actual = result.get(report_key)
                if (
                    expected is None
                    or not isinstance(actual, (int, float))
                    or not math.isclose(
                        float(actual), expected, rel_tol=1e-9, abs_tol=1e-9
                    )
                ):
                    errors.append(f"{size}/{mode}: {report_key} disagrees with samples")
            if mode == "rust_native":
                native_samples = [
                    sample for sample in mode_samples if isinstance(sample, dict)
                ]
                for report_key, sample_key in (
                    ("median_native_cpu_elapsed_ms", "native_cpu_elapsed_ms"),
                    ("median_native_peak_rss_bytes", "native_peak_rss_bytes"),
                ):
                    values = [sample.get(sample_key) for sample in native_samples]
                    actual = result.get(report_key)
                    if (
                        len(values) != samples
                        or not all(
                            isinstance(value, (int, float)) and value > 0
                            for value in values
                        )
                        or not isinstance(actual, (int, float))
                        or not math.isclose(
                            float(actual),
                            statistics.median(values),
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                    ):
                        errors.append(
                            f"{size}/{mode}: {report_key} disagrees with samples"
                        )

            latency_summary.setdefault(str(size), {})[mode] = {
                operation: {
                    f"{percentile}_ms": statistics.mean(row[percentile] for row in rows)
                    for percentile in ("p50", "p95", "p99")
                }
                for operation, rows in operation_metrics.items()
                if rows
            }
            throughput = result.get("median_throughput_ops_per_sec")
            if isinstance(throughput, (int, float)) and math.isfinite(throughput):
                throughput_summary.setdefault(str(size), {})[mode] = float(throughput)
        if set(digest_by_mode) != set(modes) or len(set(digest_by_mode.values())) != 1:
            errors.append(f"{size}: canonical state digests differ or are incomplete")

    performance_gate: dict[str, Any] = {
        "applicable": "http" in modes and "rust_native" in modes,
        "thresholds": {
            "minimum_throughput_ratio": MIN_RUST_NATIVE_THROUGHPUT_RATIO,
            "maximum_per_operation_p95_latency_ratio": MAX_RUST_NATIVE_P95_LATENCY_RATIO,
        },
        "fixtures": {},
        "passed": None,
    }
    if performance_gate["applicable"]:
        performance_gate["passed"] = True
        for size in items:
            size_key = str(size)
            python_throughput = throughput_summary.get(size_key, {}).get("http")
            rust_throughput = throughput_summary.get(size_key, {}).get("rust_native")
            python_latency = latency_summary.get(size_key, {}).get("http", {})
            rust_latency = latency_summary.get(size_key, {}).get("rust_native", {})
            if (
                python_throughput is None
                or rust_throughput is None
                or set(python_latency) != set(rust_latency)
            ):
                errors.append(f"{size}: performance comparison metrics are incomplete")
                performance_gate["passed"] = False
                continue

            throughput_ratio = rust_throughput / python_throughput
            p95_ratios: dict[str, float | None] = {}
            if throughput_ratio < MIN_RUST_NATIVE_THROUGHPUT_RATIO:
                errors.append(
                    f"{size}/rust_native: throughput ratio {throughput_ratio:.3f} "
                    f"is below {MIN_RUST_NATIVE_THROUGHPUT_RATIO:.2f}"
                )
                performance_gate["passed"] = False
            for operation in python_latency:
                python_p95 = python_latency[operation]["p95_ms"]
                rust_p95 = rust_latency[operation]["p95_ms"]
                if python_p95 <= 0:
                    ratio = 1.0 if rust_p95 <= 0 else None
                else:
                    ratio = rust_p95 / python_p95
                p95_ratios[operation] = ratio
                if ratio is None or ratio > MAX_RUST_NATIVE_P95_LATENCY_RATIO:
                    ratio_label = "undefined" if ratio is None else f"{ratio:.3f}"
                    errors.append(
                        f"{size}/rust_native/{operation}: p95 latency ratio "
                        f"{ratio_label} exceeds {MAX_RUST_NATIVE_P95_LATENCY_RATIO:.2f}"
                    )
                    performance_gate["passed"] = False
            performance_gate["fixtures"][size_key] = {
                "throughput_ratio": throughput_ratio,
                "operation_p95_latency_ratios": p95_ratios,
            }

    return {
        "passed": not errors,
        "errors": errors,
        "validation_error_count": len(errors),
        "failed_samples": 0 if not errors else None,
        "latency_summary_ms": latency_summary,
        "performance_gate": performance_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, action="append", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--mode",
        action="append",
        choices=ENDPOINT_MODES,
        help="mode to include; defaults to Python HTTP, Python HTTP with Rust authority, and Rust-native HTTP",
    )
    parser.add_argument("--expected-cpus", type=float, required=True)
    parser.add_argument("--expected-memory-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.samples < 2 or any(item < 1 for item in args.items):
        parser.error("provide positive --items values and at least two samples")
    modes = (
        tuple(args.mode)
        if args.mode
        else (
            "http",
            "rust_authority_http",
            "rust_native",
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    limits = {
        "cpu_max": _cgroup_value("cpu.max"),
        "memory_max": _cgroup_value("memory.max"),
        "expected_cpus": args.expected_cpus,
        "expected_memory_bytes": args.expected_memory_bytes,
    }
    reports: list[dict[str, Any]] = []
    for size in args.items:
        with tempfile.TemporaryDirectory(prefix="kogwistar-endpoint-") as temp_dir:
            scratch = Path(temp_dir) / str(size)
            try:
                report = run_benchmark(
                    scratch,
                    items=size,
                    samples=args.samples,
                    modes=modes,
                    include_concurrent=False,
                )
            finally:
                _copy_json_artifacts(scratch, args.output / str(size))
        reports.append(report)

    verification = _verify_endpoint_reports(
        reports,
        output=args.output,
        items=args.items,
        samples=args.samples,
        modes=modes,
    )

    limits_verified = _verify_cgroup_limits(
        limits["cpu_max"],
        limits["memory_max"],
        args.expected_cpus,
        args.expected_memory_bytes,
    )
    result = {
        "schema": "kogwistar-resource-matched-endpoint-report/v1",
        "status": "complete",
        "modes": list(modes),
        "items": args.items,
        "samples_per_mode": args.samples,
        "state_digest_equal": all(row["state_digest_equal"] for row in reports),
        "verification": verification,
        "reports": reports,
        "runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "packages": _installed_packages(),
            "limits": limits,
            "limits_verified": limits_verified,
            "native_binary": os.environ.get("KOGWISTAR_MIXED_NATIVE_BINARY"),
        },
    }
    if not limits_verified:
        result["status"] = "invalid_container_limits"
    elif not verification["passed"]:
        result["status"] = "verification_failed"
    if not result["state_digest_equal"]:
        result["status"] = "failed_digest_parity"
    _write_json(args.output / "endpoint-report.v1.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
