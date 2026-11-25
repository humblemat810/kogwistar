from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.agent_request_benchmark_report import summarize_agent_samples

ROOT = Path(__file__).resolve().parents[2]


def _resource_evidence(*, rust_cpus: int = 3) -> dict[str, object]:
    return {
        "verified": True,
        "source": "docker-inspect+cgroup",
        "python_client": {"cpus": 3, "memory_bytes": 6 * 1024**3},
        "rust_client": {"cpus": rust_cpus, "memory_bytes": 6 * 1024**3},
        "postgres": {"cpus": 1, "memory_bytes": 2 * 1024**3},
    }


def _samples(prefix: str, *, mode: str, request_ms: list[float]) -> str:
    digest = "a" * 64
    rows = []
    for index, elapsed in enumerate(request_ms):
        sample_id = "warmup" if index == 0 else f"sample-{index}"
        rows.append(
            prefix
            + json.dumps(
                {
                    "sample_id": sample_id,
                    "execution_mode": mode,
                    "request_ms": elapsed * 10,
                    "tcp_request_ms": elapsed,
                    "run_status_poll_interval_ms": 100,
                    "fixture_nodes": 2001,
                    "canonical_state_sha256": digest,
                },
                sort_keys=True,
            )
        )
    return "\n".join(rows)


def test_agent_report_compares_same_trace_and_computes_scoped_gate() -> None:
    result = summarize_agent_samples(
        python_output=_samples(
            "PYTHON_AGENT_BENCH_SAMPLE=",
            mode="python-http",
            request_ms=[1000, 1100, 1200, 1300],
        ),
        rust_output=_samples(
            "AGENT_BENCH_SAMPLE=",
            mode="rust-native-http",
            request_ms=[800, 810, 820, 830],
        ),
        expected_fixture_nodes=2001,
        expected_digest="a" * 64,
        resource_evidence=_resource_evidence(),
        rust_ffi_calls=0,
        rust_python_worker_count=0,
    )

    assert result["status"] == "pass"
    assert result["samples_per_mode"] == 3
    assert result["python"]["warmup_ms"] == 1000
    assert result["python"]["measured_median_ms"] == 1200
    assert result["rust_native"]["measured_median_ms"] == 820
    assert result["rust_native"]["throughput_ratio"] > 0.95
    assert result["rust_native"]["p95_latency_ratio"] <= 1.10


def test_agent_report_requires_same_measured_sample_ids() -> None:
    python_output = _samples(
        "PYTHON_AGENT_BENCH_SAMPLE=",
        mode="python-http",
        request_ms=[1000, 1100, 1200, 1300],
    )
    rust_output = _samples(
        "AGENT_BENCH_SAMPLE=",
        mode="rust-native-http",
        request_ms=[800, 810, 820, 830],
    ).replace('"sample_id": "sample-3"', '"sample_id": "sample-4"')
    with pytest.raises(ValueError, match="sample ID mismatch between runtimes"):
        summarize_agent_samples(
            python_output=python_output,
            rust_output=rust_output,
            expected_fixture_nodes=2001,
            expected_digest="a" * 64,
            resource_evidence=_resource_evidence(),
            rust_ffi_calls=0,
            rust_python_worker_count=0,
        )


def test_agent_report_marks_measured_performance_regression_as_fail() -> None:
    result = summarize_agent_samples(
        python_output=_samples(
            "PYTHON_AGENT_BENCH_SAMPLE=",
            mode="python-http",
            request_ms=[1000, 1100, 1200, 1300],
        ),
        rust_output=_samples(
            "AGENT_BENCH_SAMPLE=",
            mode="rust-native-http",
            request_ms=[1400, 1500, 1600, 1700],
        ),
        expected_fixture_nodes=2001,
        expected_digest="a" * 64,
        resource_evidence=_resource_evidence(),
        rust_ffi_calls=0,
        rust_python_worker_count=0,
    )
    assert result["status"] == "fail"
    assert "Rust throughput is below the 0.95x gate" in result["errors"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("digest", "digest mismatch"),
        ("poll", "poll interval mismatch"),
        ("mode", "execution mode mismatch"),
        ("sample_count", "requires one warm-up and at least three measured samples"),
    ],
)
def test_agent_report_fails_closed_on_incomparable_or_incomplete_samples(
    change: str, message: str
) -> None:
    python_output = _samples(
        "PYTHON_AGENT_BENCH_SAMPLE=",
        mode="python-http",
        request_ms=[1000, 1100, 1200, 1300],
    )
    rust_output = _samples(
        "AGENT_BENCH_SAMPLE=",
        mode="rust-native-http",
        request_ms=[800, 810, 820, 830],
    )
    if change == "digest":
        rust_output = rust_output.replace("a" * 64, "b" * 64, 1)
    elif change == "poll":
        rust_output = rust_output.replace('"run_status_poll_interval_ms": 100', '"run_status_poll_interval_ms": 200', 1)
    elif change == "mode":
        rust_output = rust_output.replace("rust-native-http", "python-http", 1)
    else:
        rust_output = "\n".join(rust_output.splitlines()[:-1])

    with pytest.raises(ValueError, match=message):
        summarize_agent_samples(
            python_output=python_output,
            rust_output=rust_output,
            expected_fixture_nodes=2001,
            expected_digest="a" * 64,
            resource_evidence=_resource_evidence(),
            rust_ffi_calls=0,
            rust_python_worker_count=0,
        )


def test_agent_report_does_not_pass_without_resource_and_native_isolation_evidence() -> None:
    kwargs = {
        "python_output": _samples(
            "PYTHON_AGENT_BENCH_SAMPLE=",
            mode="python-http",
            request_ms=[1000, 1100, 1200, 1300],
        ),
        "rust_output": _samples(
            "AGENT_BENCH_SAMPLE=",
            mode="rust-native-http",
            request_ms=[800, 810, 820, 830],
        ),
        "expected_fixture_nodes": 2001,
        "expected_digest": "a" * 64,
        "resource_evidence": _resource_evidence(rust_cpus=2),
        "rust_ffi_calls": 0,
        "rust_python_worker_count": 0,
    }
    result = summarize_agent_samples(**kwargs)
    assert result["status"] == "not_assessed"
    assert "Python and Rust client resource limits differ" in result["errors"]

    kwargs["resource_evidence"] = _resource_evidence()
    kwargs["rust_ffi_calls"] = 1
    result = summarize_agent_samples(**kwargs)
    assert result["status"] == "not_assessed"
    assert "Rust-native path invoked FFI" in result["errors"]


def test_python_and_rust_benchmark_hooks_emit_comparable_sample_contract() -> None:
    python_source = (ROOT / "tests/workflow/test_answer_workflow_v2.py").read_text(
        encoding="utf-8"
    )
    rust_source = (
        ROOT
        / "rust/crates/kogwistar-api/src/tests/postgres_agentic_answer.rs"
    ).read_text(encoding="utf-8")

    for sample_id in ('"warmup"', '"sample-1"', '"sample-2"', '"sample-3"'):
        assert sample_id in python_source
    for field in (
        '"sample_id": sample_id',
        '"execution_mode": "python-http"',
        '"fixture_nodes": fixture_size + 1',
        '"tcp_request_ms": round(http_request_ms, 3)',
        '"canonical_state_sha256": digest',
    ):
        assert field in python_source

    assert '"AGENT_BENCH_SAMPLE={}"' in rust_source
    for field in (
        '"sample_id": benchmark_sample_id',
        '"execution_mode": "rust-native-http"',
        '"fixture_nodes": benchmark_fixture_size + 1',
        '"tcp_request_ms": agent_request_ms',
        '"canonical_state_sha256": digest',
    ):
        assert field in rust_source
    assert "KOGWISTAR_AGENT_BENCH_SAMPLE_ID" in rust_source
