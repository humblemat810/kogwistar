from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "runtime_authority_gate.py"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runtime_authority_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase4_gate_is_persisted_and_declares_async_rust_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "test_rust_runtime_authority.py" in source
    assert "test_rust_runtime_python_worker.py" in source
    assert "test_runtime_parity_bridge_contract.py" in source
    assert "test_rust_runtime_postgres.py" in source
    assert "async-v2-worker-protocol" in source
    assert "live_postgres_required" in source
    assert 'counts["skipped"] == 0' in source
    assert "not slow and not manual" in source
    assert "no:cacheprovider" in source
    assert "GROUP_TIMEOUT_SECONDS = 600" in source


def test_phase4_junit_counts_handles_suite_root_and_nested_suites(tmp_path: Path) -> None:
    gate = _gate()
    direct = tmp_path / "direct.xml"
    direct.write_text(
        "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>",
        encoding="utf-8",
    )
    nested = tmp_path / "nested.xml"
    nested.write_text(
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )

    assert gate._junit_counts(direct) == {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "passed": 1,
    }
    assert gate._junit_counts(nested) == {
        "tests": 3,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
        "passed": 1,
    }


def test_phase4_live_postgres_groups_reject_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def fake_run(command, **_kwargs):
        junit_arg = next(value for value in command if str(value).startswith("--junitxml="))
        junit = Path(str(junit_arg).split("=", 1)[1])
        junit.write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="1"/>',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    postgres = gate._run_group(
        python=python,
        name="postgres",
        tests=gate.GROUPS["postgres"][0],
        output_root=tmp_path,
        live_postgres=True,
    )
    worker = gate._run_group(
        python=python,
        name="durable-worker",
        tests=gate.GROUPS["durable-worker"][0],
        output_root=tmp_path,
        live_postgres=True,
    )

    assert postgres["status"] == "failed"
    assert worker["status"] == "failed"


def test_phase4_group_timeout_is_recorded_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def fake_run(command, **kwargs):
        assert "no:cacheprovider" in command
        assert "faulthandler_timeout=30" in command
        assert kwargs["timeout"] == gate.GROUP_TIMEOUT_SECONDS
        assert kwargs["stdout"].writable()
        assert kwargs["stderr"].writable()
        kwargs["stdout"].write("partial stdout")
        kwargs["stderr"].write("partial stderr")
        raise gate.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate._run_group(
        python=ROOT / ".venv" / "Scripts" / "python.exe",
        name="durable-worker",
        tests=gate.GROUPS["durable-worker"][0],
        output_root=tmp_path,
        live_postgres=True,
    )

    assert result["status"] == "failed"
    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 600
    assert "partial stdout" in (tmp_path / "durable-worker.out.txt").read_text(
        encoding="utf-8"
    )
    stderr = (tmp_path / "durable-worker.err.txt").read_text(encoding="utf-8")
    assert "partial stderr" in stderr
    assert "exceeded 600-second timeout" in stderr
