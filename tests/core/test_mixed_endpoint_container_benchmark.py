from __future__ import annotations

import json
import sys

import pytest

from scripts import mixed_endpoint_container_benchmark


@pytest.mark.ci
def test_endpoint_runner_compares_python_mixed_and_rust_http_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_benchmark(output, **kwargs):
        calls.append({"output": output, **kwargs})
        output.mkdir(parents=True)
        (output / "progress.v3.json").write_text("{}", encoding="utf-8")
        (output / "mcp_trace.v1.json").write_text("{}", encoding="utf-8")
        (output / "sample.sqlite3").write_bytes(b"not copied to host mount")
        return {
            "state_digest_equal": True,
            "modes": {
                "http": {"state_digest": "digest"},
                "rust_authority_http": {"state_digest": "digest"},
                "rust_native": {"state_digest": "digest"},
            },
        }

    monkeypatch.setattr(mixed_endpoint_container_benchmark, "run_benchmark", fake_run_benchmark)
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
        ],
    )

    assert mixed_endpoint_container_benchmark.main() == 0
    assert calls[0]["modes"] == (
        "http",
        "rust_authority_http",
        "rust_native",
    )
    assert calls[0]["include_concurrent"] is False
    assert calls[0]["output"] != tmp_path / "2"
    assert (tmp_path / "2" / "progress.v3.json").is_file()
    assert (tmp_path / "2" / "mcp_trace.v1.json").is_file()
    assert not list(tmp_path.rglob("*.sqlite3"))
    report = json.loads((tmp_path / "endpoint-report.v1.json").read_text())
    assert report["status"] == "complete"
    assert report["modes"] == ["http", "rust_authority_http", "rust_native"]
    assert report["state_digest_equal"] is True


def test_endpoint_runner_preserves_json_evidence_when_mode_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def failed_run(output, **kwargs):
        output.mkdir(parents=True)
        (output / "progress.v3.json").write_text('{"status":"running"}', encoding="utf-8")
        (output / "mcp_trace.v1.json").write_text("{}", encoding="utf-8")
        (output / "sample.sqlite3").write_bytes(b"scratch only")
        raise RuntimeError("simulated MCP failure")

    monkeypatch.setattr(mixed_endpoint_container_benchmark, "run_benchmark", failed_run)
    monkeypatch.setattr(
        mixed_endpoint_container_benchmark,
        "_cgroup_value",
        lambda name: "400000 100000" if name == "cpu.max" else "8589934592",
    )
    monkeypatch.setattr(mixed_endpoint_container_benchmark, "_verify_cgroup_limits", lambda *args: True)
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
