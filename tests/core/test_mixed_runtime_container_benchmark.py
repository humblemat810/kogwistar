from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mixed_runtime_container_benchmark.py"
_SPEC = importlib.util.spec_from_file_location("mixed_runtime_container_benchmark", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_runtime_probe_requires_actual_free_threaded_execution() -> None:
    assert _MODULE._verify_runtime("cpython315-nogil", "CPython", "3.15.0", False, False)
    assert not _MODULE._verify_runtime("cpython315-nogil", "CPython", "3.15.0", False, True)
    assert not _MODULE._verify_runtime("cpython315-nogil", "CPython", "3.15.0", True, False)
    assert not _MODULE._verify_runtime("cpython315-nogil", "CPython", "3.15.0", None, None)
    assert not _MODULE._verify_runtime("cpython315-nogil", "CPython", "3.14.9", False, False)


def test_runtime_probe_distinguishes_gil_cpython_from_pypy() -> None:
    assert _MODULE._verify_runtime("cpython314-gil", "CPython", "3.14.9", True, True)
    assert not _MODULE._verify_runtime("cpython314-gil", "CPython", "3.14.9", True, False)
    assert not _MODULE._verify_runtime("cpython314-gil", "CPython", "3.13.9", True, True)
    assert not _MODULE._verify_runtime("cpython314-gil", "CPython", "3.15.0", True, True)
    assert _MODULE._verify_runtime("cpython315-gil", "CPython", "3.15.0", True, True)
    assert not _MODULE._verify_runtime("cpython315-gil", "CPython", "3.15.0", True, False)
    assert _MODULE._verify_runtime("pypy311", "PyPy", "3.11.13", None, None)
    assert not _MODULE._verify_runtime("pypy311", "PyPy", "3.10.18", None, None)
    assert not _MODULE._verify_runtime("pypy311", "CPython", "3.15.0", True, True)
    assert not _MODULE._verify_runtime("unknown", "CPython", "3.15.0", True, True)


def test_runtime_docker_matrix_contains_all_requested_interpreters() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "scripts" / "runtime_matrix" / "Dockerfile").read_text()
    launcher = (root / "scripts" / "run_mixed_runtime_matrix.ps1").read_text()

    assert "AS cpython314-gil" in dockerfile
    assert "ARG PYTHON_VERSION=3.14.9" in dockerfile
    assert "Name = \"cpython314-gil\"; Target = \"cpython314-gil\"" in launcher
    assert "--cpus $CpuLimit --memory $MemoryLimit --memory-swap $MemoryLimit" in launcher
    assert 'Join-Path $repo "kogwistar"' in launcher
    assert "Copy-Item -Recurse (Join-Path $repo \"kogwistar\")" not in launcher
    assert "COPY kogwistar /src/kogwistar" in dockerfile
    assert "COPY scripts /src/scripts" in dockerfile
    assert 'target=/src,readonly' not in launcher


def test_fixed_container_limits_are_verified_from_normalized_cgroup_values() -> None:
    assert _MODULE._verify_cgroup_limits("400000 100000", "8589934592", 4.0, 8 * 1024**3)
    assert not _MODULE._verify_cgroup_limits("max 100000", "8589934592", 4.0, 8 * 1024**3)
    assert not _MODULE._verify_cgroup_limits("400000 100000", "max", 4.0, 8 * 1024**3)


def test_cgroup_v1_files_are_normalized_for_limit_verification() -> None:
    values = {
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": str(8 * 1024**3),
    }

    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def __truediv__(self, other: str) -> FakePath:
            return FakePath(f"{self.value.rstrip('/')}/{other}")

        def read_text(self, *, encoding: str | None = None) -> str:
            del encoding
            value = values.get(self.value)
            if value is None:
                raise OSError("cgroup v2 file is unavailable")
            return value

    original_path = _MODULE.Path
    _MODULE.Path = FakePath
    try:
        cpu_max = _MODULE._cgroup_value("cpu.max")
        memory_max = _MODULE._cgroup_value("memory.max")
    finally:
        _MODULE.Path = original_path

    assert _MODULE._verify_cgroup_limits(cpu_max, memory_max, 4.0, 8 * 1024**3)
