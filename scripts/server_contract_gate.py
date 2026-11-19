"""Run reproducible ADR-015 Phase-5 local server-contract evidence.

This proves the current Python-owned REST, auth, SSE, syscall, and MCP serving
baseline before a later Rust server authority promotion.  It deliberately does
not claim a mixed-version rolling upgrade or a production canary: those need a
real deployment and remain separate release gates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

try:
    from gate_test_results import read_junit_test_results
    from source_fingerprint import candidate_source_fingerprint
except ModuleNotFoundError:  # imported as a repository module in tests
    from scripts.gate_test_results import read_junit_test_results
    from scripts.source_fingerprint import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / ".codex" / "adr015-phase5-server-gate.json"
NON_MODEL_MARKERS = "not slow and not manual and not llm_real and not requires_ollama"
GROUP_TIMEOUT_SECONDS = 600
RUST_API_GROUP = "rust-api"
RUST_API_EXCLUDED_TESTS = (
    "runtime_serving_lookup_scale_probe",
    "postgres_queue_projects_to_chroma_without_cross_store_transaction",
    "python_mcp_sdk_receives_rust_server_initiated_notification",
)

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
    parser.add_argument(
        "--group", action="append", choices=(*GROUPS, RUST_API_GROUP)
    )
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
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
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
    stdout_path = output_root / f"{name}.out.txt"
    stderr_path = output_root / f"{name}.err.txt"
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
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=stdout,
                stderr=stderr,
                timeout=GROUP_TIMEOUT_SECONDS,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            stderr.write(
                f"pytest group exceeded {GROUP_TIMEOUT_SECONDS}-second timeout\n"
            )
    counts = _junit_counts(junit) if junit.is_file() else None
    test_results = read_junit_test_results(junit)
    passed = (
        returncode == 0
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
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": GROUP_TIMEOUT_SECONDS,
        "junit": str(junit),
        "counts": counts,
        "test_results": test_results,
        "status": "passed" if passed else "failed",
    }


def _run_rust_api_group(*, output_root: Path) -> dict[str, Any]:
    cargo = shutil.which("cargo")
    pg_dsn = next(
        (os.environ[name] for name in ("KOGWISTAR_TEST_PG_DSN", "GKE_PG_DSN", "PG_DSN", "DATABASE_URL") if os.environ.get(name)),
        None,
    )
    chroma_url = next(
        (os.environ[name] for name in ("KOGWISTAR_TEST_CHROMA_URL", "KOGWISTAR_TEST_CHROMA_ENDPOINT") if os.environ.get(name)),
        None,
    )
    missing = [
        label
        for label, value in (
            ("cargo", cargo),
            ("PostgreSQL DSN", pg_dsn),
            ("Chroma URL", chroma_url),
        )
        if not value
    ]
    command = [
        cargo or "cargo",
        "test",
        "--manifest-path",
        "rust/Cargo.toml",
        "--release",
        "--locked",
        "-p",
        "kogwistar-api",
        "--lib",
        "--",
        "--include-ignored",
    ]
    for test_name in RUST_API_EXCLUDED_TESTS:
        command.extend(("--skip", test_name))

    stdout_path = output_root / f"{RUST_API_GROUP}.out.txt"
    stderr_path = output_root / f"{RUST_API_GROUP}.err.txt"
    if missing:
        stderr_path.write_text(
            "missing required Rust API test environment: " + ", ".join(missing) + "\n",
            encoding="utf-8",
        )
        return {
            "name": RUST_API_GROUP,
            "command": command,
            "returncode": None,
            "timed_out": False,
            "timeout_seconds": GROUP_TIMEOUT_SECONDS,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "counts": None,
            "excluded_test_filters": sorted(RUST_API_EXCLUDED_TESTS),
            "required_services": ["PostgreSQL", "Chroma"],
            "status": "failed",
        }

    child_env = os.environ.copy()
    child_env.setdefault("KOGWISTAR_TEST_PG_DSN", pg_dsn)
    child_env.setdefault("KOGWISTAR_TEST_CHROMA_URL", chroma_url)
    timed_out = False
    returncode: int | None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=child_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=stdout,
                stderr=stderr,
                timeout=GROUP_TIMEOUT_SECONDS,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            stderr.write(
                f"Rust API test group exceeded {GROUP_TIMEOUT_SECONDS}-second timeout\n"
            )

    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    summary = re.search(
        r"test result: (?:ok|FAILED)\.\s+(?P<passed>\d+) passed;\s+"
        r"(?P<failures>\d+) failed;\s+(?P<skipped>\d+) ignored;\s+"
        r"\d+ measured;\s+(?P<filtered>\d+) filtered out",
        output,
    )
    test_results = [
        {
            "source": "rust",
            "name": match.group("name"),
            "function": match.group("name").rsplit("::", 1)[-1],
            "status": {
                "ok": "passed",
                "FAILED": "failed",
                "ignored": "skipped",
            }[match.group("status")],
        }
        for match in re.finditer(
            r"(?m)^test\s+(?P<name>\S+)\s+\.\.\.\s+(?P<status>ok|FAILED|ignored)\s*$",
            output,
        )
    ]
    counts = None
    if summary is not None:
        passed = int(summary.group("passed"))
        failures = int(summary.group("failures"))
        counts = {
            "tests": passed + failures,
            "passed": passed,
            "failures": failures,
            "errors": 0,
            "skipped": int(summary.group("skipped")),
            "filtered_out": int(summary.group("filtered")),
        }
        observed = {
            "passed": sum(row["status"] == "passed" for row in test_results),
            "failures": sum(row["status"] == "failed" for row in test_results),
            "skipped": sum(row["status"] == "skipped" for row in test_results),
        }
        counts["named_test_results"] = len(test_results)
        if observed != {
            "passed": counts["passed"],
            "failures": counts["failures"],
            "skipped": counts["skipped"],
        }:
            counts["errors"] = 1
    passed = (
        returncode == 0
        and not timed_out
        and counts is not None
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["skipped"] == 0
        and counts["filtered_out"] == 1
        and counts["errors"] == 0
        and counts["named_test_results"] == counts["tests"]
    )
    return {
        "name": RUST_API_GROUP,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": GROUP_TIMEOUT_SECONDS,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "counts": counts,
        "test_results": test_results,
        "excluded_test_filters": sorted(RUST_API_EXCLUDED_TESTS),
        "required_services": ["PostgreSQL", "Chroma"],
        "status": "passed" if passed else "failed",
    }


def run_gate(
    *, python: Path, output_root: Path, groups: tuple[str, ...]
) -> dict[str, Any]:
    source_sha256, source_file_count = candidate_source_fingerprint(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in groups:
        if name == RUST_API_GROUP:
            reports.append(_run_rust_api_group(output_root=output_root))
        else:
            reports.append(
                _run_group(
                    python=python,
                    name=name,
                    tests=GROUPS[name][0],
                    output_root=output_root,
                    live_async_sse=GROUPS[name][1],
                )
            )
    return {
        "schema": "adr015-phase5-server-gate/v3",
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
        groups=tuple(args.group or (*GROUPS, RUST_API_GROUP)),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
