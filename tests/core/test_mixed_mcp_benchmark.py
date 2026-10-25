from __future__ import annotations

import asyncio
import json
import sys

import pytest

from scripts import mixed_mcp_benchmark
from scripts import mixed_concurrent_benchmark
from scripts.mixed_mcp_benchmark import (
    _run_concurrent_mcp_profiles,
    _run_one,
    _run_one_sync,
)
from scripts.mixed_workload import build_read_after_write_trace


@pytest.mark.ci_full
def test_mcp_registry_and_local_http_run_the_same_trace(tmp_path) -> None:
    trace = build_read_after_write_trace(2)
    in_process = _run_one_sync(tmp_path / "in-process.sqlite3", trace, "in_process")
    rust_authority = _run_one_sync(
        tmp_path / "rust-authority.sqlite3", trace, "rust_authority"
    )
    http = asyncio.run(_run_one(tmp_path / "http.sqlite3", trace, "http"))

    assert in_process["state_digest"] == rust_authority["state_digest"]
    assert in_process["state_digest"] == http["state_digest"]
    assert in_process["operations"] == http["operations"] == len(trace)
    assert in_process["per_operation"].keys() == http["per_operation"].keys()
    assert http["startup_ms"] > 0
    assert http["elapsed_ms"] > 0
    assert in_process["host_process_cpu_elapsed_ms"] > 0
    assert in_process["host_process_rss_start_bytes"] > 0
    assert rust_authority["host_process_cpu_elapsed_ms"] > 0
    assert http["host_process_rss_end_bytes"] > 0


def test_sync_mcp_modes_preserve_the_same_trace_digest(tmp_path) -> None:
    """Keep the restricted Windows path covered without requiring loopback."""

    trace = build_read_after_write_trace(2)
    python_result = _run_one_sync(
        tmp_path / "python.sqlite3", trace, "in_process"
    )
    rust_result = _run_one_sync(
        tmp_path / "rust-authority.sqlite3", trace, "rust_authority"
    )

    assert python_result["state_digest"] == rust_result["state_digest"]
    assert python_result["operations"] == rust_result["operations"] == len(trace)
    assert python_result["seam_metrics"]["tool_dispatches"] == len(trace)
    assert python_result["seam_metrics"]["ffi_calls"] == 0
    assert python_result["seam_metrics"]["serialized_bytes"] > 0
    assert rust_result["seam_metrics"]["tool_dispatches"] == len(trace)
    assert rust_result["seam_metrics"]["ffi_calls"] > 0
    assert rust_result["seam_metrics"]["transaction_boundaries"] > 0
    assert rust_result["seam_metrics"]["serialized_bytes"] > 0


@pytest.mark.ci_full
def test_concurrent_mcp_transports_preserve_exact_visibility(tmp_path) -> None:
    report = _run_concurrent_mcp_profiles(tmp_path, items=2, samples=2)

    assert report["workers"] == {"writers": 1, "readers": 4}
    assert set(report["mixes"]) == {"balanced", "read_heavy"}
    for mix_name, mix in report["mixes"].items():
        assert mix["state_digest_equal"] is True
        assert set(mix["modes"]) == {
            "python_in_process",
            "python_http",
            "rust_in_process",
            "rust_http",
            "rust_native_http",
        }
        assert all(
            len(mode["samples"]) == 2 for mode in mix["modes"].values()
        )
        for mode_name, mode in mix["modes"].items():
            assert mode["median_throughput_ops_per_sec"] > 0
            assert mode["median_elapsed_ms"] > 0
            assert mode["median_startup_ms"] >= 0
            assert set(mode["throughput_sample_variation"]) == {
                "mean",
                "stdev",
                "coefficient_of_variation_pct",
                "normal_approx_95pct_ci_low",
                "normal_approx_95pct_ci_high",
                "min",
                "max",
            }
            if mode_name == "rust_native_http":
                assert mode["median_native_cpu_elapsed_ms"] > 0
                assert mode["median_native_peak_rss_bytes"] > 0
            else:
                assert mode["median_host_process_cpu_elapsed_ms"] > 0
                assert mode["median_host_process_rss_delta_bytes"] >= 0
            for sample in mode["samples"]:
                assert sample["throughput_ops_per_sec"] > 0
                assert sample["sqlite_database_footprint_bytes"] > 0
                if mode_name == "rust_native_http":
                    assert sample["native_cpu_elapsed_ms"] > 0
                    assert sample["native_peak_rss_bytes"] > 0
                    assert sample["seam_metrics"]["complete"] is True
                    assert sample["seam_metrics"]["ffi_calls"] == 0
                    assert sample["seam_metrics"]["tool_dispatches"] > 0
                else:
                    assert sample["host_process_cpu_elapsed_ms"] > 0
                    assert sample["host_process_rss_start_bytes"] > 0
                    assert sample["host_process_rss_end_bytes"] > 0
                assert sample["per_operation"]["write_record"]["count"] == 2
                assert sample["per_operation"]["read_head"]["count"] == (
                    8 if mix_name == "balanced" else 16
                )
                for latency in sample["per_operation"].values():
                    assert latency["p50_ms"] >= 0
                    assert latency["p95_ms"] >= latency["p50_ms"]
                    assert latency["p99_ms"] >= latency["p95_ms"]


def test_cli_passes_requested_concurrent_fixture_size(monkeypatch, tmp_path) -> None:
    observed: dict[str, int] = {}

    def fake_run_benchmark(
        output, *, items: int, samples: int, concurrent_items: int, modes
    ) -> dict[str, int]:
        observed.update(
            items=items,
            samples=samples,
            concurrent_items=concurrent_items,
            modes=list(modes),
        )
        output.mkdir(parents=True, exist_ok=True)
        return observed.copy()

    monkeypatch.setattr(mixed_mcp_benchmark, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mixed_mcp_benchmark.py",
            "--output",
            str(tmp_path),
            "--items",
            "12",
            "--samples",
            "3",
            "--concurrent-items",
            "7",
        ],
    )

    assert mixed_mcp_benchmark.main() == 0
    assert observed == {
        "items": 12,
        "samples": 3,
        "concurrent_items": 7,
        "modes": list(mixed_mcp_benchmark.DEFAULT_MODES),
    }
    assert json.loads((tmp_path / "report.v3.json").read_text()) == observed


def test_benchmark_accepts_a_resource_matched_mode_subset(tmp_path, monkeypatch) -> None:
    native_binary = tmp_path / "mixed_mcp_authority_benchmark"
    native_binary.write_bytes(b"native-placeholder")
    monkeypatch.setenv("KOGWISTAR_MIXED_NATIVE_BINARY", str(native_binary))
    async def fake_run_one(*args, **kwargs):
        return {
            "state_digest": "same-state",
            "elapsed_ms": 1.0,
            "startup_ms": 1.0,
            "throughput_ops_per_sec": 1.0,
            "host_process_cpu_elapsed_ms": 1.0,
            "host_process_rss_delta_bytes": 1,
            "native_cpu_elapsed_ms": 1.0,
            "native_peak_rss_bytes": 1,
        }

    monkeypatch.setattr(mixed_mcp_benchmark, "_run_one", fake_run_one)
    def fake_asyncio_run(coroutine):
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        raise AssertionError("fake MCP operation unexpectedly suspended")

    monkeypatch.setattr(mixed_mcp_benchmark, "_run_async", fake_asyncio_run)
    monkeypatch.setattr(
        mixed_mcp_benchmark,
        "_trace_tool_arguments",
        lambda step: {"index": step["index"]},
    )
    monkeypatch.setattr(
        mixed_mcp_benchmark,
        "_run_concurrent_mcp_profiles",
        lambda *args, **kwargs: {"state_digest_equal": True},
    )

    report = mixed_mcp_benchmark.run_benchmark(
        tmp_path / "report",
        items=1,
        samples=2,
        modes=("http", "rust_native"),
    )

    assert set(report["modes"]) == {"http", "rust_native"}


def test_benchmark_rejects_unknown_mode_before_creating_output(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported MCP benchmark mode"):
        mixed_mcp_benchmark.run_benchmark(
            tmp_path / "report", items=1, samples=2, modes=("python",)
        )
    assert not (tmp_path / "report").exists()


def test_progress_json_checkpoint_replaces_atomically(tmp_path) -> None:
    report_path = tmp_path / "progress.v2.json"

    mixed_mcp_benchmark._write_json_atomic(report_path, {"completed_modes": ["python"]})
    concurrent_path = tmp_path / "concurrent-progress.v2.json"
    mixed_concurrent_benchmark._write_json_atomic(
        concurrent_path, {"completed_mix": "balanced"}
    )

    assert json.loads(report_path.read_text()) == {"completed_modes": ["python"]}
    assert not report_path.with_suffix(".json.tmp").exists()
    assert json.loads(concurrent_path.read_text()) == {"completed_mix": "balanced"}
    assert not concurrent_path.with_suffix(".json.tmp").exists()


def test_mcp_run_prepares_and_reuses_tool_arguments_once(monkeypatch, tmp_path) -> None:
    prepared_indexes: list[int] = []
    argument_list_ids: list[int] = []

    def prepare_argument(step):
        prepared_indexes.append(step["index"])
        return {"index": step["index"]}

    async def fake_run_one(path, trace, mode, *, trace_arguments):
        argument_list_ids.append(id(trace_arguments))
        return {
            "state_digest": "same-state",
            "elapsed_ms": 100.0,
            "startup_ms": 1.0,
            "throughput_ops_per_sec": 30.0,
            "host_process_cpu_elapsed_ms": 20.0,
            "host_process_rss_delta_bytes": 0,
            "native_cpu_elapsed_ms": 10.0,
            "native_peak_rss_bytes": 1024,
        }

    def fake_run_one_sync(path, trace, mode, *, trace_arguments):
        argument_list_ids.append(id(trace_arguments))
        return {
            "state_digest": "same-state",
            "elapsed_ms": 100.0,
            "startup_ms": 1.0,
            "throughput_ops_per_sec": 30.0,
            "host_process_cpu_elapsed_ms": 20.0,
            "host_process_rss_delta_bytes": 0,
            "native_cpu_elapsed_ms": 10.0,
            "native_peak_rss_bytes": 1024,
        }

    def fake_asyncio_run(coroutine):
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        raise AssertionError("fake MCP operation unexpectedly suspended")

    monkeypatch.setattr(mixed_mcp_benchmark, "_ensure_native_benchmark_binary", lambda: None)
    monkeypatch.setattr(mixed_mcp_benchmark, "_trace_tool_arguments", prepare_argument)
    monkeypatch.setattr(mixed_mcp_benchmark, "_run_one", fake_run_one)
    monkeypatch.setattr(mixed_mcp_benchmark, "_run_one_sync", fake_run_one_sync)
    monkeypatch.setattr(mixed_mcp_benchmark, "_run_async", fake_asyncio_run)
    monkeypatch.setattr(
        mixed_mcp_benchmark,
        "_run_concurrent_mcp_profiles",
        lambda *args, **kwargs: {"state_digest_equal": True},
    )

    report = mixed_mcp_benchmark.run_benchmark(
        tmp_path, items=2, samples=2, concurrent_items=2
    )

    assert prepared_indexes == [0, 0, 0, 1, 1, 1]
    assert len(argument_list_ids) == 12
    assert len(set(argument_list_ids)) == 1
    assert report["tool_argument_preparation_ms"] >= 0
    assert report["state_digest_equal"] is True
