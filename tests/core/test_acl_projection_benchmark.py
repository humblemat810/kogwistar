from __future__ import annotations

from scripts import acl_projection_benchmark as benchmark


def test_acl_workload_is_deterministic_and_python_projection_matches_count() -> None:
    workload = benchmark._workload(10)
    assert workload == benchmark._workload(10)

    first = benchmark._python_sample(workload, iterations=2)
    second = benchmark._python_sample(workload, iterations=2)
    assert first["visible_count"] == 7
    assert first["visible_digest"] == second["visible_digest"]


def test_resource_limit_verification_reads_cgroup_v2(monkeypatch) -> None:
    values = {
        "cpu.max": "400000 100000",
        "memory.max": "8589934592",
        "cpu/cpu.cfs_quota_us": None,
        "cpu/cpu.cfs_period_us": None,
        "memory/memory.limit_in_bytes": None,
    }
    monkeypatch.setattr(benchmark, "_cgroup_value", values.get)

    result = benchmark._verify_limits(4, 8 * 1024**3)

    assert result["verified"] is True
    assert result["actual_cpus"] == 4
    assert result["actual_memory_bytes"] == 8 * 1024**3


def test_resource_limit_verification_reads_cgroup_v1(monkeypatch) -> None:
    values = {
        "cpu.max": None,
        "memory.max": None,
        "cpu/cpu.cfs_quota_us": "400000",
        "cpu/cpu.cfs_period_us": "100000",
        "memory/memory.limit_in_bytes": "8589934592",
    }
    monkeypatch.setattr(benchmark, "_cgroup_value", values.get)

    result = benchmark._verify_limits(4, 8 * 1024**3)

    assert result["verified"] is True
    assert result["actual_cpus"] == 4
    assert result["actual_memory_bytes"] == 8 * 1024**3
