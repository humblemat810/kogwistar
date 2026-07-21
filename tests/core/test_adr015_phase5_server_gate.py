from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "adr015_phase5_server_gate.py"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("adr015_phase5_server_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase5_gate_is_persisted_and_keeps_production_upgrade_separate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "test_chat_rest_submit_and_sse" in source
    assert "test_chat_rest_sse_reconnects_cleanly_after_terminal_run" in source
    assert "test_mcp_tool_registry_contract.py" in source
    assert "test_committed_openapi_golden_has_no_drift" in source
    assert "test_versioned_server_and_syscall_deferrals_match_rust_frozen_inventory" in source
    assert "not-proven-local" in source
    assert "live_async_sse_required" in source
    assert 'counts["skipped"] == 0' in source


def test_phase5_junit_counts_handles_direct_testcases(tmp_path: Path) -> None:
    gate = _gate()
    junit = tmp_path / "direct.xml"
    junit.write_text(
        "<testsuite><testcase/><testcase><skipped/></testcase>"
        "<testcase><failure/></testcase><testcase><error/></testcase></testsuite>",
        encoding="utf-8",
    )

    assert gate._junit_counts(junit) == {
        "tests": 4,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
    }


def test_phase5_async_sse_group_rejects_skip_but_baseline_group_allows_it(
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
    async_sse = gate._run_group(
        python=python,
        name="async-sse",
        tests=gate.GROUPS["async-sse"][0],
        output_root=tmp_path,
        live_async_sse=True,
    )
    baseline = gate._run_group(
        python=python,
        name="rest-sse",
        tests=gate.GROUPS["rest-sse"][0],
        output_root=tmp_path,
        live_async_sse=False,
    )

    assert async_sse["status"] == "failed"
    assert baseline["status"] == "passed"
