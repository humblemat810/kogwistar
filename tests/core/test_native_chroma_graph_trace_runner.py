from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_native_chroma_graph_benchmark import (
    _fixed_container_limits,
    collect_chroma_trace_report,
)


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (
            {
                "/sys/fs/cgroup/cpu.max": "200000 100000",
                "/sys/fs/cgroup/memory.max": "4294967296",
            },
            {"cpu_quota_micros": 200000, "cpu_period_micros": 100000, "memory_bytes": 4294967296},
        ),
        (
            {
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "200000",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": "4294967296",
            },
            {"cpu_quota_micros": 200000, "cpu_period_micros": 100000, "memory_bytes": 4294967296},
        ),
    ],
)
def test_fixed_container_limits_accept_exact_cgroup_v1_or_v2(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    expected: dict[str, int],
) -> None:
    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        del args, kwargs
        try:
            return files[path.as_posix()]
        except KeyError as error:
            raise FileNotFoundError(path.as_posix()) from error

    monkeypatch.setattr(Path, "read_text", read_text)

    assert _fixed_container_limits() == expected


def _sample(trace: dict[str, object]) -> dict[str, object]:
    return {
        "trace_id": trace["trace_id"],
        "trace_sha256": trace["trace_sha256"],
        "state_digest": "canonical-digest",
        "canonical_event_count": 5,
        "projection_converged": True,
        "peak_rss_bytes": 1024,
        "elapsed_ms": 100.0,
        "throughput_ops_per_sec": 30.0,
        "operation_p95_ms": {"write_record": 5.0, "read_event": 1.0},
        "resource_limits_verified": True,
        "ffi_calls": 0,
    }


def test_collector_runs_one_warmup_and_fresh_samples_for_both_modes() -> None:
    trace = {"trace_id": "t1", "trace_sha256": "sha1", "operations": []}
    calls: list[tuple[str, str]] = []

    def run_sample(mode: str, sample_id: str) -> dict[str, object]:
        calls.append((mode, sample_id))
        return _sample(trace)

    report, result = collect_chroma_trace_report(
        trace,
        run_sample,
        expected_samples=3,
    )

    assert result["passed"] is True
    assert report["backend"] == "chroma_sqlite_metadata"
    assert len(calls) == 8
    assert len({sample_id for _, sample_id in calls}) == 8
    assert {mode for mode, _ in calls} == {"python", "rust_native"}
    assert report["modes"]["python"]["warmup"] is not None
    assert len(report["modes"]["rust_native"]["samples"]) == 3


def test_collector_fails_gate_when_rust_native_reports_ffi() -> None:
    trace = {"trace_id": "t1", "trace_sha256": "sha1", "operations": []}

    def run_sample(mode: str, sample_id: str) -> dict[str, object]:
        del sample_id
        sample = _sample(trace)
        if mode == "rust_native":
            sample["ffi_calls"] = 1
        return sample

    _, result = collect_chroma_trace_report(trace, run_sample, expected_samples=2)

    assert result["passed"] is False
    assert any("nonzero FFI" in error for error in result["errors"])
