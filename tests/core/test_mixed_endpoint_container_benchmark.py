from __future__ import annotations

import hashlib
import json
import sys

import pytest

from scripts import mixed_endpoint_container_benchmark


def _fake_endpoint_run(output, *, items: int, samples: int, modes: tuple[str, ...]):
    run_id = "fixture-run"
    trace = {
        "schema": "kogwistar-mixed-rewrite-trace/v2",
        "profile": "read_after_write",
        "operations": [
            {"operation": name}
            for _ in range(items)
            for name in ("write_record", "read_event", "read_projection")
        ],
    }
    run_dir = output / run_id
    run_dir.mkdir(parents=True)
    trace_path = run_dir / "mcp_trace.v1.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    expected_operations = items * 3
    digest = "canonical-state"
    mode_reports = {}
    for mode in modes:
        sample = {
            "run_id": run_id,
            "trace_sha256": trace_sha,
            "state_digest": digest,
            "operations": expected_operations,
            "elapsed_ms": 100.0,
            "startup_ms": 5.0,
            "host_process_cpu_elapsed_ms": 80.0,
            "host_process_rss_delta_bytes": 128,
            "throughput_ops_per_sec": expected_operations * 10.0,
            "per_operation": {
                name: {"count": items, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0}
                for name in ("write_record", "read_event", "read_projection")
            },
            "seam_metrics": {
                "tool_dispatches": expected_operations,
                "transport_requests": expected_operations,
                "ffi_calls": 0 if mode != "rust_authority_http" else expected_operations,
            },
            "trace_join": {
                "run_id": run_id,
                "trace_sha256": trace_sha,
                "expected_operations": expected_operations,
                "mode": mode,
            },
        }
        if mode == "rust_native":
            sample["native_cpu_elapsed_ms"] = 60.0
            sample["native_peak_rss_bytes"] = 4096
        mode_reports[mode] = {
            "state_digest": digest,
            "warmup": sample.copy(),
            "samples": [sample.copy() for _ in range(samples)],
            "median_elapsed_ms": 100.0,
            "median_startup_ms": 5.0,
            "median_host_process_cpu_elapsed_ms": 80.0,
            "median_host_process_rss_delta_bytes": 128,
            "median_throughput_ops_per_sec": expected_operations * 10.0,
            "elapsed_sample_variation": {"coefficient_of_variation_pct": 0.0},
        }
        if mode == "rust_native":
            mode_reports[mode]["median_native_cpu_elapsed_ms"] = 60.0
            mode_reports[mode]["median_native_peak_rss_bytes"] = 4096
    (output / "progress.v3.json").write_text("{}", encoding="utf-8")
    (output / "sample.sqlite3").write_bytes(b"not copied")
    return {
        "run_id": run_id,
        "trace_sha256": trace_sha,
        "items": items,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "state_digest_equal": True,
        "modes": mode_reports,
    }


def _fake_concurrent_report(output, *, items: int, samples: int):
    run_id = "concurrent-fixture"
    mix = "balanced"
    trace = {
        "schema": "kogwistar-mixed-rewrite-concurrent-trace/v3",
        "profile": "concurrent_balanced",
        "readers": 4,
        "phases": [],
    }
    trace_dir = output / "fixture-run" / "concurrent" / run_id
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "concurrent_balanced.trace.v3.json"
    trace_path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    digest = "same-concurrent-state"
    modes = {}
    for mode, throughput in (("python_http", 100.0), ("rust_native_http", 105.0)):
        sample = {
            "authority": "python" if mode == "python_http" else "rust_native",
            "transport": "http",
            "operations": items * 5,
            "elapsed_ms": 100.0,
            "startup_ms": 5.0,
            "throughput_ops_per_sec": throughput,
            "host_process_cpu_elapsed_ms": 80.0,
            "host_process_rss_delta_bytes": 128,
            "per_operation": {
                "write_record": {
                    "count": items,
                    "p50_ms": 1.0,
                    "p95_ms": 2.0,
                    "p99_ms": 3.0,
                },
                "read_head": {
                    "count": items * 4,
                    "p50_ms": 1.0,
                    "p95_ms": 2.0,
                    "p99_ms": 3.0,
                },
            },
        }
        if mode == "rust_native_http":
            sample["native_cpu_elapsed_ms"] = 60.0
            sample["native_peak_rss_bytes"] = 4096
            sample["seam_metrics"] = {"ffi_calls": 0}
        modes[mode] = {
            "state_digest": digest,
            "warmup": {**sample, "state_digest": digest},
            "samples": [{**sample, "state_digest": digest} for _ in range(samples)],
            "median_throughput_ops_per_sec": throughput,
        }
    return {
        "schema": "kogwistar-mixed-mcp-concurrent-report/v3",
        "run_id": run_id,
        "items": items,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "workers": {"writers": 1, "readers": 4},
        "mixes": {
            mix: {
                "trace_sha256": trace_sha,
                "state_digest_equal": True,
                "modes": modes,
            }
        },
    }


def test_concurrent_endpoint_report_verifier_checks_digest_and_performance(tmp_path) -> None:
    output = tmp_path / "results"
    report = {
        "run_id": "fixture-run",
        "concurrent": _fake_concurrent_report(output / "2", items=2, samples=2),
    }
    verification = mixed_endpoint_container_benchmark._verify_concurrent_endpoint_reports(
        [report], output=output, items=[2], samples=2, concurrent_items=2
    )

    assert verification["passed"] is True
    assert verification["performance_gate"]["passed"] is True
    assert verification["performance_gate"]["fixtures"]["2"]["throughput_ratio"] == 1.05


def test_concurrent_endpoint_report_verifier_rejects_mismatched_state(tmp_path) -> None:
    output = tmp_path / "results"
    concurrent = _fake_concurrent_report(output / "2", items=2, samples=2)
    concurrent["mixes"]["balanced"]["modes"]["rust_native_http"]["state_digest"] = "wrong"
    verification = mixed_endpoint_container_benchmark._verify_concurrent_endpoint_reports(
        [{"run_id": "fixture-run", "concurrent": concurrent}],
        output=output,
        items=[2],
        samples=2,
        concurrent_items=2,
    )

    assert verification["passed"] is False
    assert any("digest" in error for error in verification["errors"])


@pytest.mark.ci
def test_endpoint_runner_compares_python_mixed_and_rust_http_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_benchmark(output, **kwargs):
        calls.append({"output": output, **kwargs})
        report = _fake_endpoint_run(
            output,
            items=kwargs["items"],
            samples=kwargs["samples"],
            modes=kwargs["modes"],
        )
        if kwargs["include_concurrent"]:
            report["concurrent"] = _fake_concurrent_report(
                output,
                items=kwargs["concurrent_items"],
                samples=kwargs["samples"],
            )
        return report

    monkeypatch.setattr(
        mixed_endpoint_container_benchmark, "run_benchmark", fake_run_benchmark
    )
    monkeypatch.setattr(
        mixed_endpoint_container_benchmark,
        "_cgroup_value",
        lambda name: "400000 100000" if name == "cpu.max" else "8589934592",
    )
    monkeypatch.setattr(
        mixed_endpoint_container_benchmark,
        "_verify_cgroup_limits",
        lambda cpu, memory, expected_cpu, expected_memory: (
            cpu == "400000 100000"
            and memory == "8589934592"
            and expected_cpu == 4
            and expected_memory == 8589934592
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mixed_endpoint_container_benchmark.py",
            "--output",
            str(tmp_path),
            "--items",
            "2",
            "--samples",
            "2",
            "--expected-cpus",
            "4",
            "--expected-memory-bytes",
            "8589934592",
            "--concurrent-items",
            "2",
        ],
    )

    assert mixed_endpoint_container_benchmark.main() == 0
    assert calls[0]["modes"] == (
        "http",
        "rust_authority_http",
        "rust_native",
    )
    assert calls[0]["include_concurrent"] is True
    assert calls[0]["concurrent_modes"] == ("python_http",)
    assert calls[0]["concurrent_mixes"] == ("balanced",)
    assert calls[0]["output"] != tmp_path / "2"
    assert (tmp_path / "2" / "progress.v3.json").is_file()
    assert (tmp_path / "2" / "fixture-run" / "mcp_trace.v1.json").is_file()
    assert not list(tmp_path.rglob("*.sqlite3"))
    report = json.loads((tmp_path / "endpoint-report.v1.json").read_text())
    assert report["status"] == "complete"
    assert report["modes"] == ["http", "rust_authority_http", "rust_native"]
    assert report["state_digest_equal"] is True
    assert report["verification"]["passed"] is True
    assert report["verification"]["concurrent"]["passed"] is True
    assert report["verification"]["failed_samples"] == 0
    assert (
        report["verification"]["latency_summary_ms"]["2"]["rust_native"][
            "write_record"
        ]["p99_ms"]
        == 3.0
    )


def test_endpoint_report_verifier_rejects_cross_mode_digest_mismatch(tmp_path) -> None:
    modes = ("http", "rust_authority_http", "rust_native")
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    mixed = source / "fixture-run"
    target = destination / "2" / "fixture-run"
    target.mkdir(parents=True)
    (target / "mcp_trace.v1.json").write_bytes(
        (mixed / "mcp_trace.v1.json").read_bytes()
    )
    report["modes"]["rust_native"]["state_digest"] = "wrong-state"

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is False
    assert (
        "2: canonical state digests differ or are incomplete" in verification["errors"]
    )


def test_endpoint_performance_gate_rejects_rust_native_throughput_regression(
    tmp_path,
) -> None:
    modes = ("http", "rust_authority_http", "rust_native")
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    trace_target = destination / "2" / "fixture-run" / "mcp_trace.v1.json"
    trace_target.parent.mkdir(parents=True)
    trace_target.write_bytes(
        (source / "fixture-run" / "mcp_trace.v1.json").read_bytes()
    )
    rust_native = report["modes"]["rust_native"]
    for sample in rust_native["samples"]:
        sample["throughput_ops_per_sec"] *= 0.9
    rust_native["median_throughput_ops_per_sec"] *= 0.9

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is False
    assert any(
        "rust_native: throughput ratio" in error for error in verification["errors"]
    )


def test_endpoint_performance_gate_rejects_rust_native_operation_tail_regression(
    tmp_path,
) -> None:
    modes = ("http", "rust_authority_http", "rust_native")
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    trace_target = destination / "2" / "fixture-run" / "mcp_trace.v1.json"
    trace_target.parent.mkdir(parents=True)
    trace_target.write_bytes(
        (source / "fixture-run" / "mcp_trace.v1.json").read_bytes()
    )
    rust_native = report["modes"]["rust_native"]
    for sample in rust_native["samples"]:
        sample["per_operation"]["write_record"]["p95_ms"] = 2.3

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is False
    assert any(
        "rust_native/write_record: p95 latency ratio" in error
        for error in verification["errors"]
    )


def test_endpoint_performance_gate_accepts_values_at_declared_limits(tmp_path) -> None:
    modes = ("http", "rust_authority_http", "rust_native")
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    trace_target = destination / "2" / "fixture-run" / "mcp_trace.v1.json"
    trace_target.parent.mkdir(parents=True)
    trace_target.write_bytes(
        (source / "fixture-run" / "mcp_trace.v1.json").read_bytes()
    )
    rust_native = report["modes"]["rust_native"]
    for sample in rust_native["samples"]:
        sample["throughput_ops_per_sec"] *= 0.95
        for metrics in sample["per_operation"].values():
            metrics["p95_ms"] = 2.2
    rust_native["median_throughput_ops_per_sec"] *= 0.95

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is True
    assert verification["performance_gate"]["passed"] is True


def test_endpoint_performance_gate_is_unassessed_without_both_pure_modes(
    tmp_path,
) -> None:
    modes = ("http",)
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    trace_target = destination / "2" / "fixture-run" / "mcp_trace.v1.json"
    trace_target.parent.mkdir(parents=True)
    trace_target.write_bytes(
        (source / "fixture-run" / "mcp_trace.v1.json").read_bytes()
    )

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is True
    assert verification["performance_gate"]["applicable"] is False
    assert verification["performance_gate"]["passed"] is None


@pytest.mark.ci
@pytest.mark.parametrize("pure_mode", ("http", "rust_native"))
def test_endpoint_report_verifier_rejects_ffi_in_pure_mode(
    tmp_path, pure_mode: str
) -> None:
    modes = ("http", "rust_authority_http", "rust_native")
    source = tmp_path / "source"
    report = _fake_endpoint_run(source, items=2, samples=2, modes=modes)
    destination = tmp_path / "results"
    trace_source = source / "fixture-run" / "mcp_trace.v1.json"
    trace_target = destination / "2" / "fixture-run" / "mcp_trace.v1.json"
    trace_target.parent.mkdir(parents=True)
    trace_target.write_bytes(trace_source.read_bytes())

    for sample in [
        report["modes"][pure_mode]["warmup"],
        *report["modes"][pure_mode]["samples"],
    ]:
        sample["seam_metrics"]["ffi_calls"] = 1

    verification = mixed_endpoint_container_benchmark._verify_endpoint_reports(
        [report], output=destination, items=[2], samples=2, modes=modes
    )

    assert verification["passed"] is False
    assert any(
        f"2/{pure_mode}: pure mode reports FFI calls" in error
        for error in verification["errors"]
    )


def test_endpoint_runner_preserves_json_evidence_when_mode_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def failed_run(output, **kwargs):
        output.mkdir(parents=True)
        (output / "progress.v3.json").write_text(
            '{"status":"running"}', encoding="utf-8"
        )
        (output / "mcp_trace.v1.json").write_text("{}", encoding="utf-8")
        (output / "sample.sqlite3").write_bytes(b"scratch only")
        raise RuntimeError("simulated MCP failure")

    monkeypatch.setattr(mixed_endpoint_container_benchmark, "run_benchmark", failed_run)
    monkeypatch.setattr(
        mixed_endpoint_container_benchmark,
        "_cgroup_value",
        lambda name: "400000 100000" if name == "cpu.max" else "8589934592",
    )
    monkeypatch.setattr(
        mixed_endpoint_container_benchmark, "_verify_cgroup_limits", lambda *args: True
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mixed_endpoint_container_benchmark.py",
            "--output",
            str(tmp_path),
            "--items",
            "2",
            "--mode",
            "http",
            "--expected-cpus",
            "4",
            "--expected-memory-bytes",
            "8589934592",
        ],
    )

    with pytest.raises(RuntimeError, match="simulated MCP failure"):
        mixed_endpoint_container_benchmark.main()

    assert (tmp_path / "2" / "progress.v3.json").is_file()
    assert (tmp_path / "2" / "mcp_trace.v1.json").is_file()
    assert not list(tmp_path.rglob("*.sqlite3"))
