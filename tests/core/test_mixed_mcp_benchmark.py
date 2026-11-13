from __future__ import annotations

import asyncio
import json
import sys

import pytest

from scripts import mixed_mcp_benchmark
from scripts import mixed_concurrent_benchmark
from scripts.mixed_mcp_benchmark import (
    _bind_trace_identity,
    _run_concurrent_mcp_profiles,
    _run_one,
    _run_one_sync,
)


def test_sample_metrics_are_bound_to_trace_and_reject_duplicate_counts() -> None:
    result = {
        "operations": 2,
        "per_operation": {"write": {"count": 1}, "read": {"count": 1}},
        "seam_metrics": {"tool_dispatches": 2},
    }

    bound = _bind_trace_identity(
        result, run_id="run-a", trace_sha256="trace-a", expected_operations=2
    )

    assert bound["run_id"] == "run-a"
    assert bound["trace_sha256"] == "trace-a"
    with pytest.raises(RuntimeError, match="operation counters"):
        _bind_trace_identity(
            {**result, "per_operation": {"write": {"count": 3}}},
            run_id="run-a",
            trace_sha256="trace-a",
            expected_operations=2,
        )
    with pytest.raises(RuntimeError, match="dispatch counters"):
        _bind_trace_identity(
            {**result, "seam_metrics": {"tool_dispatches": 1}},
            run_id="run-a",
            trace_sha256="trace-a",
            expected_operations=2,
        )


def _joined_rust_http_sample() -> dict[str, object]:
    operations = {
        "begin_transaction": 1,
        "commit_transaction": 1,
        "raw_append": 2,
    }
    return {
        "operations": 2,
        "per_operation": {"write": {"count": 1}, "read": {"count": 1}},
        "seam_metrics": {
            "scope": "python_host_boundary",
            "complete": True,
            "tool_dispatches": 2,
            "transport_requests": 2,
            "ffi_calls": 4,
            "ffi_by_operation": operations,
            "serialized_bytes": 100,
            "transaction_boundaries": 2,
        },
        "native_bridge_trace": {
            "measurement": "untimed_isolated_database_replay",
            "state_digest_matches": True,
            "ffi_calls": 4,
            "calls_by_operation": operations,
            "transaction_boundary_calls": 2,
            "transaction_active_calls": 1,
            "input_bytes": 90,
            "output_bytes": 60,
        },
    }


def test_rust_trace_join_allows_session_lifecycle_count_differences() -> None:
    result = _joined_rust_http_sample()
    seam = result["seam_metrics"]
    bridge = result["native_bridge_trace"]
    assert isinstance(seam, dict)
    assert isinstance(bridge, dict)
    seam["ffi_calls"] = 105
    seam["ffi_by_operation"] = {
        "open_init": 1,
        "begin_transaction": 1,
        "commit_transaction": 1,
        "raw_append": 2,
        "close": 100,
    }
    bridge["ffi_calls"] = 6
    bridge["calls_by_operation"] = {
        "open_init": 1,
        "begin_transaction": 1,
        "commit_transaction": 1,
        "raw_append": 2,
        "close": 1,
    }

    joined = _bind_trace_identity(
        result,
        run_id="run-a",
        trace_sha256="trace-a",
        expected_operations=2,
        mode="rust_authority_http",
    )

    assert joined["trace_join"]["ffi_calls"] == 105
    assert joined["trace_join"]["observed_bridge_replay_ffi_calls"] == 6


def test_rust_trace_join_rejects_graph_call_count_difference() -> None:
    result = _joined_rust_http_sample()
    bridge = result["native_bridge_trace"]
    assert isinstance(bridge, dict)
    bridge["ffi_calls"] = 3
    bridge["calls_by_operation"] = {
        "begin_transaction": 1,
        "commit_transaction": 1,
        "raw_append": 1,
    }

    with pytest.raises(RuntimeError, match="operation counters disagree"):
        _bind_trace_identity(
            result,
            run_id="run-a",
            trace_sha256="trace-a",
            expected_operations=2,
            mode="rust_authority_http",
        )


def test_rust_trace_join_rejects_inconsistent_host_ffi_total() -> None:
    result = _joined_rust_http_sample()
    seam = result["seam_metrics"]
    assert isinstance(seam, dict)
    seam["ffi_calls"] = 5

    with pytest.raises(RuntimeError, match="host FFI total"):
        _bind_trace_identity(
            result,
            run_id="run-a",
            trace_sha256="trace-a",
            expected_operations=2,
            mode="rust_authority_http",
        )


def _fake_benchmark_sample(trace: list[dict[str, object]], mode: str) -> dict[str, object]:
    operations = {"begin_transaction": 1, "commit_transaction": 1}
    sample: dict[str, object] = {
        "state_digest": "same-state",
        "operations": len(trace),
        "per_operation": {"write_record": {"count": len(trace)}},
        "seam_metrics": {
            "scope": "python_host_boundary",
            "complete": True,
            "tool_dispatches": len(trace),
            "transport_requests": len(trace)
            if mode in {"http", "rust_authority_http", "rust_native"}
            else 0,
        },
        "elapsed_ms": 1.0,
        "startup_ms": 1.0,
        "throughput_ops_per_sec": 1.0,
        "host_process_cpu_elapsed_ms": 1.0,
        "host_process_rss_delta_bytes": 1,
        "native_cpu_elapsed_ms": 1.0,
        "native_peak_rss_bytes": 1,
    }
    if mode in {"rust_authority", "rust_authority_http"}:
        seam = sample["seam_metrics"]
        assert isinstance(seam, dict)
        seam.update(
            ffi_calls=2,
            ffi_by_operation=operations,
            serialized_bytes=10,
            transaction_boundaries=2,
        )
        sample["native_bridge_trace"] = {
            "measurement": "untimed_isolated_database_replay",
            "state_digest_matches": True,
            "ffi_calls": 2,
            "calls_by_operation": operations,
            "transaction_boundary_calls": 2,
            "transaction_active_calls": 1,
            "input_bytes": 10,
            "output_bytes": 10,
        }
    return sample
def test_trace_attribution_joins_http_ingress_and_rust_transaction_owner() -> None:
    result = _bind_trace_identity(
        _joined_rust_http_sample(),
        run_id="run-a",
        trace_sha256="trace-a",
        expected_operations=2,
        mode="rust_authority_http",
    )

    assert result["trace_join"] == {
        "mode": "rust_authority_http",
        "run_id": "run-a",
        "trace_sha256": "trace-a",
        "expected_operations": 2,
        "ingress_requests": 2,
        "tool_dispatches": 2,
        "ffi_calls": 4,
        "serialized_bytes": 100,
        "transaction_boundaries": 2,
        "transaction_owner": "rust_meta_store",
        "observed_bridge_input_bytes": 90,
        "observed_bridge_output_bytes": 60,
        "observed_bridge_replay_ffi_calls": 4,
        "transaction_active_calls": 1,
        "observed_transaction_begins": 1,
        "observed_transaction_commits": 1,
        "bridge_replay_state_matches": True,
    }


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("seam_metrics", "transport_requests", 1, "ingress"),
        ("seam_metrics", "ffi_calls", 3, "FFI"),
        ("seam_metrics", "transaction_boundaries", 1, "transaction boundaries"),
        ("native_bridge_trace", "transaction_active_calls", 0, "active Rust transaction calls"),
    ],
)
def test_trace_attribution_rejects_unjoined_or_missing_counters(
    section: str, key: str, value: int, message: str
) -> None:
    result = _joined_rust_http_sample()
    nested = result[section]
    assert isinstance(nested, dict)
    nested[key] = value

    with pytest.raises(RuntimeError, match=message):
        _bind_trace_identity(
            result,
            run_id="run-a",
            trace_sha256="trace-a",
            expected_operations=2,
            mode="rust_authority_http",
        )


def _batch_comparison_mode(
    *, elapsed_ms: float, ffi_calls: int, serialized_bytes: int
) -> dict[str, object]:
    return {
        "state_digest": "same-state",
        "samples": [
            {
                "state_digest": "same-state",
                "run_id": "run-a",
                "trace_sha256": "trace-a",
                "operations": 600,
                "elapsed_ms": elapsed_ms,
                "throughput_ops_per_sec": 600_000 / elapsed_ms,
                "seam_metrics": {
                    "transaction_boundaries": 202,
                    "ffi_calls": ffi_calls,
                    "serialized_bytes": serialized_bytes,
                },
            }
            for _ in range(2)
        ],
    }


def test_batch_comparison_enforces_seam_budgets_and_defers_noisy_latency() -> None:
    per_item = _batch_comparison_mode(
        elapsed_ms=1000.0, ffi_calls=900, serialized_bytes=100_000
    )
    batch = _batch_comparison_mode(
        elapsed_ms=1100.0, ffi_calls=400, serialized_bytes=100_500
    )

    result = mixed_mcp_benchmark._compare_batch_write_modes(per_item, batch)

    assert result["transaction_boundaries_equal"] is True
    assert result["ffi_ratio"] == pytest.approx(400 / 900)
    assert result["serialized_byte_ratio"] == pytest.approx(1.005)
    assert result["latency_status"] == "inconclusive_insufficient_samples"


def test_batch_comparison_rejects_different_trace_identity() -> None:
    per_item = _batch_comparison_mode(
        elapsed_ms=1000.0, ffi_calls=900, serialized_bytes=100_000
    )
    batch = _batch_comparison_mode(
        elapsed_ms=1000.0, ffi_calls=400, serialized_bytes=100_500
    )
    batch["samples"] = [
        {**sample, "trace_sha256": "different-trace"}
        for sample in batch["samples"]
    ]

    with pytest.raises(RuntimeError, match="run and trace identity"):
        mixed_mcp_benchmark._compare_batch_write_modes(per_item, batch)


@pytest.mark.parametrize(
    ("batch_kwargs", "error"),
    [
        ({"elapsed_ms": 1000.0, "ffi_calls": 550, "serialized_bytes": 100_000}, "FFI ratio"),
        ({"elapsed_ms": 1000.0, "ffi_calls": 400, "serialized_bytes": 102_000}, "serialized-byte ratio"),
    ],
)
def test_batch_comparison_fails_when_deterministic_budgets_are_exceeded(
    batch_kwargs: dict[str, float | int], error: str
) -> None:
    per_item = _batch_comparison_mode(
        elapsed_ms=1000.0, ffi_calls=900, serialized_bytes=100_000
    )
    batch = _batch_comparison_mode(**batch_kwargs)

    with pytest.raises(RuntimeError, match=error):
        mixed_mcp_benchmark._compare_batch_write_modes(per_item, batch)


def test_batch_comparison_enforces_latency_budget_only_with_stable_five_sample_sets() -> None:
    per_item = _batch_comparison_mode(
        elapsed_ms=1000.0, ffi_calls=900, serialized_bytes=100_000
    )
    batch = _batch_comparison_mode(
        elapsed_ms=1250.0, ffi_calls=400, serialized_bytes=100_500
    )
    per_item["samples"] = [
        {**sample, "elapsed_ms": 1000.0} for sample in per_item["samples"]
    ] + [{**per_item["samples"][0], "elapsed_ms": 1000.0} for _ in range(3)]
    batch["samples"] = [
        {**sample, "elapsed_ms": 1250.0} for sample in batch["samples"]
    ] + [{**batch["samples"][0], "elapsed_ms": 1250.0} for _ in range(3)]

    with pytest.raises(RuntimeError, match="latency ratio"):
        mixed_mcp_benchmark._compare_batch_write_modes(per_item, batch)
from scripts.mixed_workload import build_read_after_write_trace


@pytest.mark.ci_full
def test_mcp_registry_and_local_http_run_the_same_trace(tmp_path) -> None:
    trace = build_read_after_write_trace(2)
    in_process = _run_one_sync(tmp_path / "in-process.sqlite3", trace, "in_process")
    rust_authority = _run_one_sync(
        tmp_path / "rust-authority.sqlite3", trace, "rust_authority"
    )
    http = asyncio.run(_run_one(tmp_path / "http.sqlite3", trace, "http"))
    rust_authority_http = asyncio.run(
        _run_one(
            tmp_path / "rust-authority-http.sqlite3", trace, "rust_authority_http"
        )
    )
    rust_http_joined = _bind_trace_identity(
        rust_authority_http,
        run_id="integration-run",
        trace_sha256="integration-trace",
        expected_operations=len(trace),
        mode="rust_authority_http",
    )

    assert in_process["state_digest"] == rust_authority["state_digest"]
    assert in_process["state_digest"] == http["state_digest"]
    assert in_process["state_digest"] == rust_authority_http["state_digest"]
    assert in_process["operations"] == http["operations"] == len(trace)
    assert in_process["per_operation"].keys() == http["per_operation"].keys()
    for operation, metrics in http["per_operation"].items():
        assert metrics["count"] == sum(
            step["operation"] == operation for step in trace
        )
        assert 0 <= metrics["p50_ms"] <= metrics["p95_ms"] <= metrics["p99_ms"]
    assert http["startup_ms"] > 0
    assert http["elapsed_ms"] > 0
    assert in_process["host_process_cpu_elapsed_ms"] > 0
    assert in_process["host_process_rss_start_bytes"] > 0
    assert rust_authority["host_process_cpu_elapsed_ms"] > 0
    assert http["host_process_rss_end_bytes"] > 0
    assert rust_authority_http["seam_metrics"]["ffi_calls"] > 0
    assert rust_authority_http["seam_metrics"]["tool_dispatches"] == len(trace)
    assert rust_authority_http["seam_metrics"]["transport_requests"] == len(trace)
    assert rust_http_joined["trace_join"]["transaction_owner"] == "rust_meta_store"
    assert rust_http_joined["trace_join"]["ingress_requests"] == len(trace)
    assert rust_http_joined["trace_join"]["bridge_replay_state_matches"] is True


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
    native_trace = rust_result["native_bridge_trace"]
    assert native_trace["measurement"] == "untimed_isolated_database_replay"
    assert native_trace["state_digest_matches"] is True
    assert native_trace["ffi_calls"] == rust_result["seam_metrics"]["ffi_calls"]


@pytest.mark.ci_full
def test_rust_native_http_transport_attribution_is_complete(tmp_path) -> None:
    trace = build_read_after_write_trace(2)
    result = asyncio.run(
        mixed_mcp_benchmark._run_one(
            tmp_path / "rust-native-attribution.sqlite3", trace, "rust_native"
        )
    )

    attribution = result["transport_attribution"]
    assert attribution["scope"] == "rust_native_mcp_benchmark"
    for stage in (
        "client_request_json_encode",
        "client_http_request_response_body_round_trip",
        "client_response_json_decode",
    ):
        assert attribution[stage]["count"] == len(trace)
        assert attribution[stage]["mean_ms"] >= 0
    server_stage = attribution["server_response_json_encode"]
    assert server_stage["count"] == len(trace)
    assert server_stage["total_ms"] >= 0
    assert server_stage["mean_ms"] >= 0
    assert result["seam_metrics"]["response_serialization_count"] == len(trace)
    assert result["state_digest"]


def test_rust_batched_write_seam_preserves_trace_and_commit_boundaries(
    tmp_path,
) -> None:
    trace = build_read_after_write_trace(2)
    ordinary = _run_one_sync(
        tmp_path / "rust-ordinary.sqlite3", trace, "rust_authority"
    )
    batched = _run_one_sync(
        tmp_path / "rust-batched.sqlite3", trace, "rust_authority_batch_write"
    )

    assert ordinary["state_digest"] == batched["state_digest"]
    assert ordinary["operations"] == batched["operations"] == len(trace)
    assert ordinary["per_operation"].keys() == batched["per_operation"].keys()
    assert {
        name: values["count"]
        for name, values in ordinary["per_operation"].items()
    } == {
        name: values["count"]
        for name, values in batched["per_operation"].items()
    }
    assert ordinary["seam_metrics"]["tool_dispatches"] == len(trace)
    assert batched["seam_metrics"]["tool_dispatches"] == len(trace)
    assert batched["seam_metrics"]["ffi_calls"] < ordinary["seam_metrics"]["ffi_calls"]
    assert (
        batched["seam_metrics"]["transaction_boundaries"]
        == ordinary["seam_metrics"]["transaction_boundaries"]
    )
    assert batched["native_bridge_trace"]["ffi_calls"] == batched["seam_metrics"]["ffi_calls"]


def test_native_seam_metrics_require_complete_zero_ffi_attestation() -> None:
    metrics = {
        "scope": "rust_native_server",
        "complete": True,
        "tool_dispatches": 3,
        "ffi_calls": 0,
        "serialized_bytes": 42,
        "transaction_boundaries": 1,
        "transport_requests": 3,
    }

    assert mixed_mcp_benchmark._validate_native_seam_metrics(
        metrics, expected_operations=3, label="test"
    ) == metrics


def test_native_graph_storage_attribution_matches_operation_trace_counts() -> None:
    metrics = {
        "scope": "rust_native_server",
        "complete": True,
        "tool_dispatches": 3,
        "ffi_calls": 0,
        "serialized_bytes": 42,
        "transaction_boundaries": 1,
        "transport_requests": 3,
        "graph_storage": {
            "write_record": {"calls": 1, "duration_ns": 100},
            "read_event": {"calls": 1, "duration_ns": 20},
            "read_projection": {"calls": 2, "duration_ns": 30},
        },
    }

    assert mixed_mcp_benchmark._validate_native_seam_metrics(
        metrics,
        expected_operations=3,
        label="test",
        expected_graph_storage_calls={
            "write_record": 1,
            "read_event": 1,
            "read_projection": 2,
        },
    ) == metrics


@pytest.mark.parametrize(
    "storage",
    [
        None,
        {"write_record": {"calls": 0, "duration_ns": 0}},
        {
            "write_record": {"calls": 1, "duration_ns": 0},
            "read_event": {"calls": 1, "duration_ns": 20},
            "read_projection": {"calls": 2, "duration_ns": 30},
        },
    ],
)
def test_native_graph_storage_attribution_rejects_missing_or_mismatched_counts(
    storage: object,
) -> None:
    metrics: dict[str, object] = {
        "scope": "rust_native_server",
        "complete": True,
        "tool_dispatches": 3,
        "ffi_calls": 0,
        "serialized_bytes": 42,
        "transaction_boundaries": 1,
        "transport_requests": 3,
        "graph_storage": storage,
    }

    with pytest.raises(RuntimeError, match="graph-storage|graph-storage attribution"):
        mixed_mcp_benchmark._validate_native_seam_metrics(
            metrics,
            expected_operations=3,
            label="test",
            expected_graph_storage_calls={
                "write_record": 1,
                "read_event": 1,
                "read_projection": 2,
            },
        )


def test_transport_attribution_summarizes_separate_client_and_server_stages() -> None:
    assert mixed_mcp_benchmark._latency_summary_ms([1.0, 2.0, 3.0]) == {
        "count": 3,
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 3.0,
        "p99_ms": 3.0,
    }
    assert mixed_mcp_benchmark._response_serialization_attribution(
        {"response_serialization_ns": 9000, "response_serialization_count": 3},
        expected_operations=3,
    ) == {"count": 3, "total_ms": 0.009, "mean_ms": 0.003}


@pytest.mark.parametrize(
    "metrics",
    [
        {"response_serialization_ns": 1, "response_serialization_count": 2},
        {"response_serialization_ns": True, "response_serialization_count": 3},
        {"response_serialization_ns": 0, "response_serialization_count": False},
    ],
)
def test_response_serialization_attribution_rejects_incomplete_metrics(metrics) -> None:
    with pytest.raises(RuntimeError, match="serialization attribution is incomplete"):
        mixed_mcp_benchmark._response_serialization_attribution(
            metrics, expected_operations=3
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scope", "python_host_boundary", "scope"),
        ("complete", False, "incomplete"),
        ("ffi_calls", 1, "FFI"),
        ("tool_dispatches", 2, "tool_dispatches"),
        ("transport_requests", 4, "transport_requests"),
    ],
)
def test_native_seam_metrics_reject_false_or_incomplete_attestation(
    field: str, value: object, message: str
) -> None:
    metrics: dict[str, object] = {
        "scope": "rust_native_server",
        "complete": True,
        "tool_dispatches": 3,
        "ffi_calls": 0,
        "serialized_bytes": 42,
        "transaction_boundaries": 1,
        "transport_requests": 3,
    }
    metrics[field] = value

    with pytest.raises(RuntimeError, match=message):
        mixed_mcp_benchmark._validate_native_seam_metrics(
            metrics, expected_operations=3, label="test"
        )


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


@pytest.mark.parametrize("skip_concurrent", [False, True])
def test_cli_passes_requested_concurrent_fixture_size(
    monkeypatch, tmp_path, skip_concurrent: bool
) -> None:
    observed: dict[str, object] = {}

    def fake_run_benchmark(
        output,
        *,
        items: int,
        samples: int,
        concurrent_items: int,
        modes,
        include_concurrent: bool,
    ) -> dict[str, object]:
        observed.update(
            items=items,
            samples=samples,
            concurrent_items=concurrent_items,
            modes=list(modes),
            include_concurrent=include_concurrent,
        )
        output.mkdir(parents=True, exist_ok=True)
        return observed.copy()

    monkeypatch.setattr(mixed_mcp_benchmark, "run_benchmark", fake_run_benchmark)
    argv = [
        "mixed_mcp_benchmark.py",
        "--output",
        str(tmp_path),
        "--items",
        "12",
        "--samples",
        "3",
        "--concurrent-items",
        "7",
    ]
    if skip_concurrent:
        argv.append("--skip-concurrent")
    monkeypatch.setattr(sys, "argv", argv)

    assert mixed_mcp_benchmark.main() == 0
    assert observed == {
        "items": 12,
        "samples": 3,
        "concurrent_items": 7,
        "modes": list(mixed_mcp_benchmark.DEFAULT_MODES),
        "include_concurrent": not skip_concurrent,
    }
    assert json.loads((tmp_path / "report.v3.json").read_text()) == observed


def test_benchmark_accepts_a_resource_matched_mode_subset(tmp_path, monkeypatch) -> None:
    native_binary = tmp_path / "mixed_mcp_authority_benchmark"
    native_binary.write_bytes(b"native-placeholder")
    monkeypatch.setenv("KOGWISTAR_MIXED_NATIVE_BINARY", str(native_binary))
    async def fake_run_one(*args, **kwargs):
        trace = args[1]
        return _fake_benchmark_sample(trace, args[2])

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
        modes=("http", "rust_authority_http", "rust_native"),
    )

    assert set(report["modes"]) == {"http", "rust_authority_http", "rust_native"}


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


@pytest.mark.parametrize(
    "writer",
    [
        mixed_mcp_benchmark._write_json_atomic,
        mixed_concurrent_benchmark._write_json_atomic,
    ],
)
def test_progress_json_checkpoint_retries_transient_replace_lock(
    monkeypatch, tmp_path, writer
) -> None:
    report_path = tmp_path / "progress.v3.json"
    original_replace = mixed_mcp_benchmark.os.replace
    attempts = 0

    def replace_with_transient_lock(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "transient Windows replace lock")
        original_replace(source, target)

    monkeypatch.setattr(mixed_mcp_benchmark.os, "replace", replace_with_transient_lock)

    writer(report_path, {"completed_modes": ["python"]})

    assert attempts == 3
    assert json.loads(report_path.read_text()) == {"completed_modes": ["python"]}
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]


def test_mcp_run_prepares_and_reuses_tool_arguments_once(monkeypatch, tmp_path) -> None:
    prepared_indexes: list[int] = []
    argument_list_ids: list[int] = []

    def prepare_argument(step):
        prepared_indexes.append(step["index"])
        return {"index": step["index"]}

    async def fake_run_one(path, trace, mode, *, trace_arguments):
        argument_list_ids.append(id(trace_arguments))
        result = _fake_benchmark_sample(trace, mode)
        result.update(
            elapsed_ms=100.0,
            startup_ms=1.0,
            throughput_ops_per_sec=30.0,
            host_process_cpu_elapsed_ms=20.0,
            host_process_rss_delta_bytes=0,
            native_cpu_elapsed_ms=10.0,
            native_peak_rss_bytes=1024,
        )
        return result

    def fake_run_one_sync(path, trace, mode, *, trace_arguments):
        argument_list_ids.append(id(trace_arguments))
        result = _fake_benchmark_sample(trace, mode)
        result.update(
            elapsed_ms=100.0,
            startup_ms=1.0,
            throughput_ops_per_sec=30.0,
            host_process_cpu_elapsed_ms=20.0,
            host_process_rss_delta_bytes=0,
            native_cpu_elapsed_ms=10.0,
            native_peak_rss_bytes=1024,
        )
        return result

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
    assert len(report["trace_sha256"]) == 64
    assert all(
        sample["run_id"] == report["run_id"]
        and sample["trace_sha256"] == report["trace_sha256"]
        and sample["operations"] == 6
        for mode in report["modes"].values()
        for sample in [mode["warmup"], *mode["samples"]]
    )
