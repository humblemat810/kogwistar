from __future__ import annotations

import copy

import pytest

from scripts.native_graph_backend_benchmark import verify_backend_report


@pytest.fixture
def valid_report() -> dict[str, object]:
    sample = {
        "trace_id": "trace-01",
        "trace_sha256": "trace-sha",
        "state_digest": "graph-sha",
        "canonical_event_count": 6,
        "projection_converged": True,
        "elapsed_ms": 100.0,
        "throughput_ops_per_sec": 60.0,
        "operation_p95_ms": {"write": 2.0, "read": 1.0},
        "resource_limits_verified": True,
        "ffi_calls": 0,
    }
    return {
        "schema": "kogwistar-native-graph-backend-benchmark/v1",
        "backend": "postgres_pgvector",
        "samples_per_mode": 2,
        "warmup_samples_per_mode": 1,
        "trace": {"trace_id": "trace-01", "trace_sha256": "trace-sha"},
        "modes": {
            mode: {
                "warmup": copy.deepcopy(sample),
                "samples": [copy.deepcopy(sample), copy.deepcopy(sample)],
            }
            for mode in ("python", "rust_native")
        },
    }


def test_backend_report_accepts_equivalent_python_and_rust_traces(valid_report) -> None:
    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is True
    assert result["performance_gate"]["passed"] is True


@pytest.mark.parametrize("backend", ["sqlite_event_projection", "chroma_sqlite_metadata"])
def test_backend_report_rejects_backend_identity_mismatch(valid_report, backend) -> None:
    result = verify_backend_report(
        valid_report,
        expected_backend=backend,
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("backend" in error for error in result["errors"])


def test_backend_report_rejects_different_trace_identity(valid_report) -> None:
    valid_report["modes"]["rust_native"]["samples"][0]["trace_sha256"] = "other"

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("trace" in error for error in result["errors"])


def test_chroma_report_requires_eventual_projection_convergence(valid_report) -> None:
    valid_report["backend"] = "chroma_sqlite_metadata"
    valid_report["modes"]["rust_native"]["samples"][0]["projection_converged"] = False

    result = verify_backend_report(
        valid_report,
        expected_backend="chroma_sqlite_metadata",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("converg" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ffi_calls", 1, "ffi"),
        ("resource_limits_verified", False, "resource"),
        ("state_digest", "different", "digest"),
    ],
)
def test_backend_report_rejects_invalid_native_or_parity_evidence(
    valid_report, field, value, message
) -> None:
    valid_report["modes"]["rust_native"]["samples"][0][field] = value

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any(message in error.lower() for error in result["errors"])


def test_backend_report_applies_declared_throughput_and_tail_thresholds(valid_report) -> None:
    for sample in valid_report["modes"]["rust_native"]["samples"]:
        sample["throughput_ops_per_sec"] = 50.0
        sample["operation_p95_ms"]["write"] = 2.3

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert result["performance_gate"]["passed"] is False
    assert any("throughput" in error for error in result["errors"])
    assert any("p95" in error for error in result["errors"])
