from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "server_contract_gate.py"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("server_contract_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase5_gate_is_persisted_and_keeps_production_upgrade_separate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "test_chat_rest_submit_and_sse" in source
    assert (
        "tests/server/test_chat_server_api_run_inspection.py::"
        "test_chat_rest_sse_reconnects_cleanly_after_terminal_run"
    ) in source
    assert (
        "tests/server/test_chat_server_api_runtime_workflow.py::"
        "test_workflow_run_lane_message_lifecycle_streams_through_registry_sse"
    ) in source
    assert (
        "tests/server/test_agent_event_api.py::"
        "test_terminal_sse_stream_stops_when_service_only_returns_stale_events"
    ) in source
    for module in (
        "test_chat_mcp_conversation_tools.py",
        "test_chat_mcp_graph_reads.py",
        "test_chat_mcp_graph_tools.py",
        "test_chat_mcp_operator_tools.py",
        "test_chat_mcp_workflow_authority.py",
        "test_chat_mcp_workflow_run_access.py",
        "test_chat_mcp_workflow_runtime_design.py",
        "test_chat_mcp_workflow_services.py",
    ):
        assert f'"tests/mcp/{module}"' in source
    assert "test_mcp_tool_registry_contract.py" in source
    assert "test_committed_openapi_golden_has_no_drift" in source
    assert "test_closed_server_cutover_inventory_matches_rust_frozen_inventory" in source
    assert "not-proven-local" in source
    assert "live_async_sse_required" in source
    assert 'counts["skipped"] == 0' in source
    assert "RUST_API_GROUP = \"rust-api\"" in source
    assert "--include-ignored" in source
    assert "adr015-phase5-server-gate/v4" in source


def test_rust_api_gate_requires_live_services_and_records_cargo_test_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    monkeypatch.setattr(gate.shutil, "which", lambda _name: "cargo.exe")
    monkeypatch.setenv("KOGWISTAR_TEST_PG_DSN", "postgresql://test/db")
    monkeypatch.setenv("KOGWISTAR_TEST_CHROMA_URL", "http://127.0.0.1:18000")

    def fake_run(command, **kwargs):
        assert "--include-ignored" in command
        assert "--release" not in command
        assert kwargs["env"]["KOGWISTAR_TEST_PG_DSN"] == "postgresql://test/db"
        kwargs["stdout"].write("running 455 tests\n")
        for index in range(455):
            kwargs["stdout"].write(f"test tests::fixture_{index} ... ok\n")
        kwargs["stdout"].write(
            "test result: ok. 455 passed; 0 failed; 0 ignored; "
            "0 measured; 1 filtered out; finished in 42.02s\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate._run_rust_api_group(output_root=tmp_path)

    assert result["status"] == "passed"
    assert result["counts"] == {
        "tests": 455,
        "passed": 455,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "filtered_out": 1,
        "named_test_results": 455,
    }
    assert len(result["test_results"]) == 455
    assert result["required_services"] == ["PostgreSQL", "Chroma"]
    assert result["test_profile"] == "debug-ci"


def test_rust_api_gate_fails_closed_without_live_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    monkeypatch.setattr(gate.shutil, "which", lambda _name: "cargo.exe")
    monkeypatch.delenv("KOGWISTAR_TEST_PG_DSN", raising=False)
    monkeypatch.delenv("GKE_PG_DSN", raising=False)
    monkeypatch.delenv("PG_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KOGWISTAR_TEST_CHROMA_URL", raising=False)
    monkeypatch.delenv("KOGWISTAR_TEST_CHROMA_ENDPOINT", raising=False)

    result = gate._run_rust_api_group(output_root=tmp_path)

    assert result["status"] == "failed"
    assert result["returncode"] is None
    assert "PostgreSQL DSN, Chroma URL" in (
        tmp_path / "rust-api.err.txt"
    ).read_text(encoding="utf-8")


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
        "passed": 1,
        "failures": 1,
        "errors": 1,
        "skipped": 1,
    }


def test_phase5_async_sse_group_rejects_skip_but_baseline_group_allows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def fake_run(command, **_kwargs):
        assert "capture_output" not in _kwargs
        _kwargs["stdout"].write("pytest stdout")
        _kwargs["stderr"].write("pytest stderr")
        junit_arg = next(value for value in command if str(value).startswith("--junitxml="))
        junit = Path(str(junit_arg).split("=", 1)[1])
        junit.write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="1"/>',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

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
    assert (tmp_path / "rest-sse.out.txt").read_text(encoding="utf-8") == "pytest stdout"
    assert (tmp_path / "rest-sse.err.txt").read_text(encoding="utf-8") == "pytest stderr"


def test_phase5_group_timeout_is_reported_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def timed_out(command, **_kwargs):
        raise gate.subprocess.TimeoutExpired(command, timeout=gate.GROUP_TIMEOUT_SECONDS)

    monkeypatch.setattr(gate.subprocess, "run", timed_out)
    result = gate._run_group(
        python=ROOT / ".venv" / "Scripts" / "python.exe",
        name="rest-sse",
        tests=gate.GROUPS["rest-sse"][0],
        output_root=tmp_path,
        live_async_sse=False,
    )

    assert result["status"] == "failed"
    assert result["timed_out"] is True
    assert result["returncode"] == 124
    assert str(gate.GROUP_TIMEOUT_SECONDS) in (tmp_path / "rest-sse.err.txt").read_text(
        encoding="utf-8"
    )
