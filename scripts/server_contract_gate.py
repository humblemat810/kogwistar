"""Run reproducible ADR-015 Phase-5 local server-contract evidence.

This proves the current Python-owned REST, auth, SSE, syscall, and MCP serving
baseline before a later Rust server authority promotion.  It deliberately does
not claim a mixed-version rolling upgrade or a production canary: those need a
real deployment and remain separate release gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from xml.etree import ElementTree

try:
    from source_fingerprint import candidate_source_fingerprint
except ModuleNotFoundError:  # imported as a repository module in tests
    from scripts.source_fingerprint import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / ".codex" / "adr015-phase5-server-gate.json"
NON_MODEL_MARKERS = "not slow and not manual and not llm_real and not requires_ollama"

GROUPS: dict[str, tuple[tuple[str, ...], bool]] = {
    "rest-sse": (
        (
            "tests/server/test_chat_server_api.py::test_chat_rest_submit_and_sse",
            "tests/server/test_chat_server_api.py::test_chat_rest_sse_reconnects_cleanly_after_terminal_run",
            "tests/server/test_chat_server_api.py::test_runtime_rest_submit_and_sse",
            "tests/server/test_chat_server_api.py::test_runtime_rest_cancel",
        ),
        False,
    ),
    "auth-syscall": (
        (
            "tests/server/test_auth_integration.py",
            "tests/server/test_mcp_role_middleware.py",
            "tests/server/test_syscall_api.py",
            "tests/server/test_capability_kernel_contract.py",
        ),
        False,
    ),
    "mcp": (
        (
            "tests/mcp/test_chat_mcp_tools.py",
            "tests/server/test_mcp_tool_registry_contract.py",
        ),
        False,
    ),
    "frozen-contracts": (
        (
            "tests/core/test_rust_port_contract_manifest.py::test_committed_openapi_golden_has_no_drift",
            "tests/server/test_mcp_tool_registry_contract.py::test_committed_mcp_tool_schema_matches_live_registry",
            "tests/core/test_rust_port_contract_manifest.py::test_versioned_server_and_syscall_deferrals_match_rust_frozen_inventory",
        ),
        False,
    ),
    "async-sse": (
        (
            "tests/server/test_chat_server_async_events.py::test_chat_rest_events_poll_sees_live_updates_for_async_backends",
            "tests/server/test_chat_server_async_events.py::test_mcp_run_events_sees_live_updates_for_async_backends",
            "tests/server/test_chat_server_async_events.py::test_workflow_runtime_sse_cancel_after_sleep_ticks_for_async_backends",
        ),
        True,
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--group", action="append", choices=tuple(GROUPS))
    return parser.parse_args()


def _junit_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        for name in totals:
            totals[name] += int(suite.attrib.get(name, 0))
    if totals["tests"] == 0:
        for case in root.iter("testcase"):
            totals["tests"] += 1
            totals["failures"] += case.find("failure") is not None
            totals["errors"] += case.find("error") is not None
            totals["skipped"] += case.find("skipped") is not None
    return totals


def _run_group(
    *,
    python: Path,
    name: str,
    tests: tuple[str, ...],
    output_root: Path,
    live_async_sse: bool,
) -> dict[str, Any]:
    junit = output_root / f"{name}.junit.xml"
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-m",
        NON_MODEL_MARKERS,
        f"--junitxml={junit}",
        *tests,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    (output_root / f"{name}.out.txt").write_text(completed.stdout, encoding="utf-8")
    (output_root / f"{name}.err.txt").write_text(completed.stderr, encoding="utf-8")
    counts = _junit_counts(junit) if junit.is_file() else None
    passed = (
        completed.returncode == 0
        and counts is not None
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and (not live_async_sse or counts["skipped"] == 0)
    )
    return {
        "name": name,
        "tests": list(tests),
        "live_async_sse_required": live_async_sse,
        "command": command,
        "returncode": completed.returncode,
        "junit": str(junit),
        "counts": counts,
        "status": "passed" if passed else "failed",
    }


def run_gate(
    *, python: Path, output_root: Path, groups: tuple[str, ...]
) -> dict[str, Any]:
    source_sha256, source_file_count = candidate_source_fingerprint(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = [
        _run_group(
            python=python,
            name=name,
            tests=GROUPS[name][0],
            output_root=output_root,
            live_async_sse=GROUPS[name][1],
        )
        for name in groups
    ]
    return {
        "schema": "adr015-phase5-server-gate/v1",
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
        "server_authority": "python-baseline-before-rust-cutover",
        "production_rolling_upgrade": "not-proven-local",
        "groups": reports,
        "status": "passed" if all(group["status"] == "passed" for group in reports) else "failed",
    }


def main() -> int:
    args = _args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")
    report_path = args.report.expanduser().resolve()
    result = run_gate(
        python=python,
        output_root=report_path.parent / "adr015-phase5-server-gate",
        groups=tuple(args.group or tuple(GROUPS)),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
