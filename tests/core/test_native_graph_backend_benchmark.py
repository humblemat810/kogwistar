from __future__ import annotations

import copy

import pytest

from scripts.native_graph_backend_benchmark import run_trace_sample, verify_backend_report


class TraceAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def write_record(self, operation: dict[str, object]) -> None:
        self.calls.append("write_record")

    def read_event(self, operation: dict[str, object]) -> str:
        self.calls.append("read_event")
        return str(operation["expected_entity_id"])

    def read_projection(self, operation: dict[str, object]) -> str:
        self.calls.append("read_projection")
        return str(operation["expected_entity_id"])

    def snapshot(self) -> dict[str, object]:
        return {
            "state_digest": "same-state",
            "canonical_event_count": 1,
            "projection_converged": True,
        }


def test_run_trace_sample_executes_each_operation_and_records_native_evidence() -> None:
    adapter = TraceAdapter()
    trace = [
        {"operation": "write_record", "index": 0, "expected_entity_id": "node-0"},
        {"operation": "read_event", "index": 0, "expected_entity_id": "node-0"},
        {"operation": "read_projection", "index": 0, "expected_entity_id": "node-0"},
    ]

    sample = run_trace_sample(
        adapter,
        trace,
        trace_id="trace-01",
        trace_sha256="trace-sha",
        resource_limits_verified=True,
        ffi_calls=0,
    )

    assert adapter.calls == ["write_record", "read_event", "read_projection"]
    assert sample["trace_id"] == "trace-01"
    assert sample["trace_sha256"] == "trace-sha"
    assert sample["canonical_event_count"] == 1
    assert sample["state_digest"] == "same-state"
    assert sample["resource_limits_verified"] is True
    assert sample["ffi_calls"] == 0
    assert set(sample["operation_p95_ms"]) == {"write_record", "read_event", "read_projection"}
    assert sample["elapsed_ms"] > 0
    assert sample["throughput_ops_per_sec"] > 0


def test_run_trace_sample_fails_when_read_after_write_result_differs() -> None:
    class WrongReadAdapter(TraceAdapter):
        def read_event(self, operation: dict[str, object]) -> str:
            self.calls.append("read_event")
            return "wrong-node"

    with pytest.raises(AssertionError, match="read_event"):
        run_trace_sample(
            WrongReadAdapter(),
            [
                {"operation": "write_record", "index": 0, "expected_entity_id": "node-0"},
                {"operation": "read_event", "index": 0, "expected_entity_id": "node-0"},
            ],
            trace_id="trace-01",
            trace_sha256="trace-sha",
            resource_limits_verified=True,
            ffi_calls=0,
        )


def test_run_trace_sample_rejects_unknown_operation_before_backend_call() -> None:
    adapter = TraceAdapter()
    with pytest.raises(ValueError, match="unsupported"):
        run_trace_sample(
            adapter,
            [{"operation": "skip_write", "index": 0, "expected_entity_id": "node-0"}],
            trace_id="trace-01",
            trace_sha256="trace-sha",
            resource_limits_verified=True,
            ffi_calls=0,
        )
    assert adapter.calls == []


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
