from __future__ import annotations

import json
import sys

import pytest

from scripts import mixed_endpoint_container_benchmark


@pytest.mark.ci
def test_endpoint_runner_uses_only_http_modes_and_records_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_benchmark(output, **kwargs):
        calls.append({"output": output, **kwargs})
        return {
            "state_digest_equal": True,
            "modes": {
                "http": {"state_digest": "digest"},
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
    assert calls[0]["modes"] == ("http", "rust_native")
    assert calls[0]["include_concurrent"] is False
    report = json.loads((tmp_path / "endpoint-report.v1.json").read_text())
    assert report["status"] == "complete"
    assert report["state_digest_equal"] is True
