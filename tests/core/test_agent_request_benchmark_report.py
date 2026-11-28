from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.agent_request_benchmark_report import main, summarize_agent_samples

ROOT = Path(__file__).resolve().parents[2]


def _summarize_agent_samples(**kwargs: Any) -> dict[str, Any]:
    return summarize_agent_samples(
        expected_backend_profile="postgres_pgvector",
        **kwargs,
    )


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
                    "backend_profile": "postgres_pgvector",
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
    result = _summarize_agent_samples(
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
    assert result["backend_profile"] == "postgres_pgvector"
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
        _summarize_agent_samples(
            python_output=python_output,
            rust_output=rust_output,
            expected_fixture_nodes=2001,
            expected_digest="a" * 64,
            resource_evidence=_resource_evidence(),
            rust_ffi_calls=0,
            rust_python_worker_count=0,
        )


def test_agent_report_cli_persists_backend_labelled_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_log = tmp_path / "python.log"
    rust_log = tmp_path / "rust.log"
    resource_file = tmp_path / "resources.json"
    report_file = tmp_path / "report.json"
    python_log.write_text(
        _samples("PYTHON_AGENT_BENCH_SAMPLE=", mode="python-http", request_ms=[1000, 1100, 1200, 1300]),
        encoding="utf-8",
    )
    rust_log.write_text(
        _samples("AGENT_BENCH_SAMPLE=", mode="rust-native-http", request_ms=[800, 810, 820, 830]),
        encoding="utf-8",
    )
    resource_file.write_text(json.dumps(_resource_evidence()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_request_benchmark_report",
            "--python-log", str(python_log),
            "--rust-log", str(rust_log),
            "--fixture-nodes", "2001",
            "--backend-profile", "postgres_pgvector",
            "--digest", "a" * 64,
            "--resource-evidence", str(resource_file),
            "--rust-ffi-calls", "0",
            "--rust-python-worker-count", "0",
            "--output", str(report_file),
        ],
    )

    assert main() == 0
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["schema"] == "kogwistar-agent-request-benchmark/v2"
    assert report["backend_profile"] == "postgres_pgvector"
    assert report["status"] == "pass"


@pytest.mark.parametrize("change", ["missing", "different"])
def test_agent_report_requires_matching_backend_profile(change: str) -> None:
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
    if change == "missing":
        rust_output = rust_output.replace('"backend_profile": "postgres_pgvector",', "", 1)
        expected_error = "backend profile missing"
    else:
        rust_output = rust_output.replace(
            "postgres_pgvector", "sqlite_event_projection", 1
        )
        expected_error = "backend profile mismatch between runtimes"

    with pytest.raises(ValueError, match=expected_error):
        _summarize_agent_samples(
            python_output=python_output,
            rust_output=rust_output,
            expected_fixture_nodes=2001,
            expected_digest="a" * 64,
            resource_evidence=_resource_evidence(),
            rust_ffi_calls=0,
            rust_python_worker_count=0,
        )


def test_agent_report_marks_measured_performance_regression_as_fail() -> None:
    result = _summarize_agent_samples(
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
        _summarize_agent_samples(
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
    result = _summarize_agent_samples(**kwargs)
    assert result["status"] == "not_assessed"
    assert "Python and Rust client resource limits differ" in result["errors"]

    kwargs["resource_evidence"] = _resource_evidence()
    kwargs["rust_ffi_calls"] = 1
    result = _summarize_agent_samples(**kwargs)
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
        '"backend_profile": "postgres_pgvector"',
        '"fixture_nodes": fixture_size + 1',
        '"tcp_request_ms": round(http_request_ms, 3)',
        '"canonical_state_sha256": digest',
    ):
        assert field in python_source

    assert '"AGENT_BENCH_SAMPLE={}"' in rust_source
    for field in (
        '"sample_id": benchmark_sample_id',
        '"execution_mode": "rust-native-http"',
        '"backend_profile": "postgres_pgvector"',
        '"fixture_nodes": benchmark_fixture_size + 1',
        '"tcp_request_ms": agent_request_ms',
        '"canonical_state_sha256": digest',
    ):
        assert field in rust_source
    assert "KOGWISTAR_AGENT_BENCH_SAMPLE_ID" in rust_source
