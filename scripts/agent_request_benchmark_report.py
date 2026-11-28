"""Validate comparable Python/Rust authenticated-agent benchmark samples."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

PYTHON_MARKER = "PYTHON_AGENT_BENCH_SAMPLE="
RUST_MARKER = "AGENT_BENCH_SAMPLE="
MIN_MEASURED_SAMPLES = 3
EXPECTED_POLL_INTERVAL_MS = 100
MAX_RUST_THROUGHPUT_RATIO = 0.95
MAX_RUST_P95_LATENCY_RATIO = 1.10
BACKEND_PROFILES = {
    "sqlite_event_projection",
    "chroma_sqlite_metadata",
    "postgres_pgvector",
}


def _parse_samples(output: str, marker: str, expected_mode: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if marker not in line:
            continue
        encoded = line.split(marker, 1)[1].strip()
        try:
            sample = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {marker} JSON at line {line_number}") from exc
        if not isinstance(sample, dict):
            raise ValueError(f"{marker} payload must be a JSON object")
        samples.append(sample)

    if not samples:
        raise ValueError(f"no {marker} samples found")
    if any(sample.get("execution_mode") != expected_mode for sample in samples):
        raise ValueError(f"execution mode mismatch: expected {expected_mode}")
    if any(
        not isinstance(sample.get("backend_profile"), str)
        or not sample["backend_profile"]
        for sample in samples
    ):
        raise ValueError("backend profile missing")
    identifiers = [sample.get("sample_id") for sample in samples]
    if identifiers.count("warmup") != 1:
        raise ValueError("requires exactly one warm-up sample")
    measured = [sample_id for sample_id in identifiers if sample_id != "warmup"]
    if len(measured) < MIN_MEASURED_SAMPLES:
        raise ValueError(
            "requires one warm-up and at least three measured samples"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be unique")

    for sample in samples:
        elapsed = sample.get("tcp_request_ms")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("tcp_request_ms must be finite and positive")
        if sample.get("run_status_poll_interval_ms") != EXPECTED_POLL_INTERVAL_MS:
            raise ValueError("poll interval mismatch: expected 100 ms")
    return samples


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def _mode_summary(samples: list[dict[str, Any]], warmup: dict[str, Any]) -> dict[str, Any]:
    measured = [sample for sample in samples if sample["sample_id"] != "warmup"]
    values = [float(sample["tcp_request_ms"]) for sample in measured]
    median_ms = statistics.median(values)
    p95_ms = _p95(values)
    return {
        "warmup_ms": float(warmup["tcp_request_ms"]),
        "measured_samples_ms": values,
        "measured_median_ms": median_ms,
        "measured_p95_ms": p95_ms,
        "throughput_requests_per_second": 1000.0 / median_ms,
    }


def _resource_errors(evidence: dict[str, Any]) -> list[str]:
    if evidence.get("verified") is not True or not evidence.get("source"):
        return ["resource limits lack verified Docker/cgroup evidence"]
    profiles = [evidence.get(name) for name in ("python_client", "rust_client", "postgres")]
    if any(not isinstance(profile, dict) for profile in profiles):
        return ["resource evidence must include Python, Rust, and PostgreSQL limits"]
    for name, profile in zip(("python_client", "rust_client", "postgres"), profiles):
        cpus = profile.get("cpus")
        memory_bytes = profile.get("memory_bytes")
        if (
            not isinstance(cpus, (int, float))
            or not math.isfinite(cpus)
            or cpus <= 0
            or not isinstance(memory_bytes, int)
            or memory_bytes <= 0
        ):
            return [f"invalid resource limits for {name}"]
    if profiles[0] != profiles[1]:
        return ["Python and Rust client resource limits differ"]
    return []


def summarize_agent_samples(
    *,
    python_output: str,
    rust_output: str,
    expected_backend_profile: str,
    expected_fixture_nodes: int,
    expected_digest: str,
    resource_evidence: dict[str, Any],
    rust_ffi_calls: int,
    rust_python_worker_count: int,
) -> dict[str, Any]:
    """Compare full authenticated HTTP traces; never infer missing isolation evidence."""
    if expected_backend_profile not in BACKEND_PROFILES:
        raise ValueError(f"unsupported backend profile: {expected_backend_profile}")
    python_samples = _parse_samples(python_output, PYTHON_MARKER, "python-http")
    rust_samples = _parse_samples(rust_output, RUST_MARKER, "rust-native-http")
    errors: list[str] = []
    python_ids = {sample["sample_id"] for sample in python_samples}
    rust_ids = {sample["sample_id"] for sample in rust_samples}
    if python_ids != rust_ids:
        raise ValueError("sample ID mismatch between runtimes")
    expected_ids = {"warmup", *(f"sample-{index}" for index in range(1, len(python_ids)))}
    if python_ids != expected_ids:
        raise ValueError("sample IDs must be warmup and contiguous sample-1..N")
    if {sample["backend_profile"] for sample in python_samples} != {
        sample["backend_profile"] for sample in rust_samples
    }:
        raise ValueError("backend profile mismatch between runtimes")
    combined = python_samples + rust_samples
    for sample in combined:
        if sample["backend_profile"] != expected_backend_profile:
            raise ValueError("backend profile mismatch between samples and requested profile")
        if sample.get("fixture_nodes") != expected_fixture_nodes:
            raise ValueError("fixture size mismatch")
        if sample.get("canonical_state_sha256") != expected_digest:
            raise ValueError("digest mismatch")
        if sample.get("run_status_poll_interval_ms") != EXPECTED_POLL_INTERVAL_MS:
            raise ValueError("poll interval mismatch: expected 100 ms")
    intervals = {sample["run_status_poll_interval_ms"] for sample in combined}
    if len(intervals) != 1:
        raise ValueError("poll interval mismatch between runtimes")

    python_warmup = next(sample for sample in python_samples if sample["sample_id"] == "warmup")
    rust_warmup = next(sample for sample in rust_samples if sample["sample_id"] == "warmup")
    python_summary = _mode_summary(python_samples, python_warmup)
    rust_summary = _mode_summary(rust_samples, rust_warmup)
    rust_summary["throughput_ratio"] = (
        rust_summary["throughput_requests_per_second"]
        / python_summary["throughput_requests_per_second"]
    )
    rust_summary["p95_latency_ratio"] = (
        rust_summary["measured_p95_ms"] / python_summary["measured_p95_ms"]
    )

    errors.extend(_resource_errors(resource_evidence))
    if rust_ffi_calls != 0:
        errors.append("Rust-native path invoked FFI")
    if rust_python_worker_count != 0:
        errors.append("Rust-native path used a Python worker")
    evidence_missing = bool(errors)
    performance_errors = []
    if rust_summary["throughput_ratio"] < MAX_RUST_THROUGHPUT_RATIO:
        performance_errors.append("Rust throughput is below the 0.95x gate")
    if rust_summary["p95_latency_ratio"] > MAX_RUST_P95_LATENCY_RATIO:
        performance_errors.append("Rust p95 latency exceeds the 1.10x gate")
    errors.extend(performance_errors)
    status = "not_assessed" if evidence_missing else "fail" if performance_errors else "pass"
    return {
        "schema": "kogwistar-agent-request-benchmark/v2",
        "status": status,
        "backend_profile": expected_backend_profile,
        "samples_per_mode": len(python_samples) - 1,
        "fixture_nodes": expected_fixture_nodes,
        "canonical_state_sha256": expected_digest,
        "poll_interval_ms": EXPECTED_POLL_INTERVAL_MS,
        "resource_evidence": resource_evidence,
        "rust_ffi_calls": rust_ffi_calls,
        "rust_python_worker_count": rust_python_worker_count,
        "python": python_summary,
        "rust_native": rust_summary,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--rust-log", type=Path, required=True)
    parser.add_argument("--fixture-nodes", type=int, required=True)
    parser.add_argument("--backend-profile", choices=sorted(BACKEND_PROFILES), required=True)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--resource-evidence", type=Path, required=True)
    parser.add_argument("--rust-ffi-calls", type=int, required=True)
    parser.add_argument("--rust-python-worker-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_agent_samples(
        python_output=args.python_log.read_text(encoding="utf-8"),
        rust_output=args.rust_log.read_text(encoding="utf-8"),
        expected_backend_profile=args.backend_profile,
        expected_fixture_nodes=args.fixture_nodes,
        expected_digest=args.digest,
        resource_evidence=json.loads(args.resource_evidence.read_text(encoding="utf-8")),
        rust_ffi_calls=args.rust_ffi_calls,
        rust_python_worker_count=args.rust_python_worker_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
