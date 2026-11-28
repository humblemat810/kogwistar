"""Verification contracts for resource-matched native graph backend traces."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from time import perf_counter, process_time
from typing import Any, Protocol


MIN_RUST_THROUGHPUT_RATIO = 0.95
MAX_RUST_P95_LATENCY_RATIO = 1.10
BACKENDS = frozenset({"postgres_pgvector", "chroma_sqlite_metadata"})
MODES = frozenset({"python", "rust_native"})


class TraceBackendAdapter(Protocol):
    def write_record(self, operation: Mapping[str, Any]) -> None: ...

    def read_event(self, operation: Mapping[str, Any]) -> str: ...

    def read_projection(self, operation: Mapping[str, Any]) -> str: ...

    def snapshot(self) -> Mapping[str, Any]: ...


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)]


def run_trace_sample(
    adapter: TraceBackendAdapter,
    trace: list[Mapping[str, Any]],
    *,
    trace_id: str,
    trace_sha256: str,
    resource_limits_verified: bool,
    ffi_calls: int,
) -> dict[str, Any]:
    """Execute fixed read-after-write operations and collect measured evidence."""
    if not trace:
        raise ValueError("trace must contain operations")
    if not trace_id or not trace_sha256:
        raise ValueError("trace identity and digest are required")
    if not resource_limits_verified or ffi_calls < 0:
        raise ValueError("verified resource limits and non-negative FFI count are required")

    elapsed_by_operation: dict[str, list[float]] = {}
    wall_start = perf_counter()
    cpu_start = process_time()
    for operation in trace:
        name = operation.get("operation")
        operation_start = perf_counter()
        if name == "write_record":
            adapter.write_record(operation)
        elif name == "read_event":
            actual_id = adapter.read_event(operation)
            if actual_id != operation.get("expected_entity_id"):
                raise AssertionError(
                    f"read_event returned {actual_id!r}, expected "
                    f"{operation.get('expected_entity_id')!r}"
                )
        elif name == "read_projection":
            actual_id = adapter.read_projection(operation)
            if actual_id != operation.get("expected_entity_id"):
                raise AssertionError(
                    f"read_projection returned {actual_id!r}, expected "
                    f"{operation.get('expected_entity_id')!r}"
                )
        else:
            raise ValueError(f"unsupported trace operation: {name!r}")
        elapsed_by_operation.setdefault(str(name), []).append(
            (perf_counter() - operation_start) * 1000
        )

    snapshot = adapter.snapshot()
    elapsed_ms = (perf_counter() - wall_start) * 1000
    cpu_ms = (process_time() - cpu_start) * 1000
    state_digest = snapshot.get("state_digest")
    event_count = snapshot.get("canonical_event_count")
    if not isinstance(state_digest, str) or not state_digest:
        raise ValueError("backend snapshot must provide a canonical state digest")
    if not isinstance(event_count, int) or event_count < 0:
        raise ValueError("backend snapshot must provide a non-negative event count")
    projection_converged = snapshot.get("projection_converged")
    if not isinstance(projection_converged, bool):
        raise ValueError("backend snapshot must report projection convergence")

    return {
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "state_digest": state_digest,
        "canonical_event_count": event_count,
        "projection_converged": projection_converged,
        "elapsed_ms": elapsed_ms,
        "cpu_ms": cpu_ms,
        "throughput_ops_per_sec": len(trace) * 1000 / elapsed_ms,
        "operation_p95_ms": {
            name: _p95(values) for name, values in sorted(elapsed_by_operation.items())
        },
        "resource_limits_verified": resource_limits_verified,
        "ffi_calls": ffi_calls,
    }


def verify_backend_report(
    report: Mapping[str, Any],
    *,
    expected_backend: str,
    expected_samples: int,
) -> dict[str, Any]:
    """Fail closed unless both pure modes prove one equivalent backend trace."""

    errors: list[str] = []
    if expected_backend not in BACKENDS:
        errors.append(f"unsupported expected backend: {expected_backend}")
    if report.get("schema") != "kogwistar-native-graph-backend-benchmark/v1":
        errors.append("report schema is missing or unsupported")
    if report.get("backend") != expected_backend:
        errors.append("backend identity differs from requested backend")
    if report.get("samples_per_mode") != expected_samples or expected_samples < 2:
        errors.append("sample count differs from requested benchmark")
    if report.get("warmup_samples_per_mode") != 1:
        errors.append("exactly one warm-up per mode is required")

    trace = report.get("trace")
    if not isinstance(trace, Mapping):
        errors.append("trace identity is missing")
        trace_id = trace_sha = None
    else:
        trace_id = trace.get("trace_id")
        trace_sha = trace.get("trace_sha256")
        if not isinstance(trace_id, str) or not trace_id:
            errors.append("trace id is missing")
        if not isinstance(trace_sha, str) or not trace_sha:
            errors.append("trace digest is missing")

    modes = report.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != MODES:
        errors.append("report must contain exactly Python and Rust-native modes")
        modes = {}

    valid_samples: dict[str, list[Mapping[str, Any]]] = {}
    digests: set[str] = set()
    operation_p95: dict[str, dict[str, list[float]]] = {
        mode: {} for mode in MODES
    }
    median_throughput: dict[str, float] = {}
    for mode in sorted(MODES):
        mode_report = modes.get(mode)
        if not isinstance(mode_report, Mapping):
            errors.append(f"{mode}: mode report is missing")
            continue
        samples = mode_report.get("samples")
        warmup = mode_report.get("warmup")
        if not isinstance(samples, list) or len(samples) != expected_samples:
            errors.append(f"{mode}: measured sample count differs")
            samples = []
        if not isinstance(warmup, Mapping):
            errors.append(f"{mode}: warm-up sample is missing")
            warmup = {}
        rows = [warmup, *(row for row in samples if isinstance(row, Mapping))]
        if len(rows) != expected_samples + 1:
            errors.append(f"{mode}: one or more samples are malformed")
        valid_samples[mode] = [row for row in rows if isinstance(row, Mapping)]

        throughputs: list[float] = []
        for sample_index, sample in enumerate(rows):
            label = "warm-up" if sample_index == 0 else f"sample {sample_index}"
            if sample.get("trace_id") != trace_id or sample.get("trace_sha256") != trace_sha:
                errors.append(f"{mode}/{label}: trace identity differs")
            digest = sample.get("state_digest")
            if not isinstance(digest, str) or not digest:
                errors.append(f"{mode}/{label}: state digest is missing")
            else:
                digests.add(digest)
            event_count = sample.get("canonical_event_count")
            if not isinstance(event_count, int) or event_count < 0:
                errors.append(f"{mode}/{label}: canonical event count is invalid")
            if expected_backend == "chroma_sqlite_metadata" and sample.get(
                "projection_converged"
            ) is not True:
                errors.append(f"{mode}/{label}: Chroma projection did not converge")
            if sample.get("resource_limits_verified") is not True:
                errors.append(f"{mode}/{label}: resource limits are not verified")
            ffi_calls = sample.get("ffi_calls")
            if mode == "rust_native" and ffi_calls != 0:
                errors.append(f"{mode}/{label}: Rust-native path reports nonzero FFI calls")

            elapsed = sample.get("elapsed_ms")
            throughput = sample.get("throughput_ops_per_sec")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                errors.append(f"{mode}/{label}: elapsed time is invalid")
            if not isinstance(throughput, (int, float)) or throughput <= 0:
                errors.append(f"{mode}/{label}: throughput is invalid")
            elif sample_index > 0:
                throughputs.append(float(throughput))

            latencies = sample.get("operation_p95_ms")
            if not isinstance(latencies, Mapping) or not latencies:
                errors.append(f"{mode}/{label}: operation p95 metrics are missing")
                continue
            for operation, value in latencies.items():
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"{mode}/{label}/{operation}: p95 latency is invalid")
                    continue
                if sample_index > 0:
                    operation_p95[mode].setdefault(str(operation), []).append(float(value))
        if throughputs:
            median_throughput[mode] = statistics.median(throughputs)

    if len(digests) != 1:
        errors.append("Python/Rust trace state digests differ or are incomplete")
    if set(valid_samples) == MODES:
        python_counts = {
            row.get("canonical_event_count") for row in valid_samples["python"]
        }
        rust_counts = {
            row.get("canonical_event_count") for row in valid_samples["rust_native"]
        }
        if python_counts != rust_counts:
            errors.append("canonical event counts differ between modes")

    performance_errors: list[str] = []
    throughput_ratio: float | None = None
    python_rate = median_throughput.get("python")
    rust_rate = median_throughput.get("rust_native")
    if python_rate is None or rust_rate is None or python_rate <= 0:
        performance_errors.append("throughput comparison is incomplete")
    else:
        throughput_ratio = rust_rate / python_rate
        if throughput_ratio < MIN_RUST_THROUGHPUT_RATIO:
            performance_errors.append(
                f"throughput ratio {throughput_ratio:.3f} is below {MIN_RUST_THROUGHPUT_RATIO:.2f}"
            )

    python_operations = operation_p95.get("python", {})
    rust_operations = operation_p95.get("rust_native", {})
    if not python_operations or set(python_operations) != set(rust_operations):
        performance_errors.append("operation p95 comparison is incomplete")
    p95_ratios: dict[str, float | None] = {}
    for operation in sorted(set(python_operations) & set(rust_operations)):
        python_values = python_operations[operation]
        rust_values = rust_operations[operation]
        if len(python_values) != expected_samples or len(rust_values) != expected_samples:
            performance_errors.append(f"{operation}: operation p95 samples are incomplete")
            continue
        python_p95 = statistics.median(python_values)
        rust_p95 = statistics.median(rust_values)
        ratio = (
            1.0 if python_p95 == 0 and rust_p95 == 0
            else None if python_p95 == 0
            else rust_p95 / python_p95
        )
        p95_ratios[operation] = ratio
        if ratio is None or ratio > MAX_RUST_P95_LATENCY_RATIO:
            performance_errors.append(
                f"{operation}: p95 latency ratio exceeds {MAX_RUST_P95_LATENCY_RATIO:.2f}"
            )

    errors.extend(performance_errors)
    return {
        "passed": not errors,
        "errors": errors,
        "performance_gate": {
            "passed": not performance_errors,
            "throughput_ratio": throughput_ratio,
            "operation_p95_latency_ratios": p95_ratios,
        },
    }
