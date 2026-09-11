"""Run ADR-015 Phase-4 runtime evidence against current native source.

Async callbacks use the explicit ``async-v2`` Python worker protocol. The
durable reducer/effect DTO remains shared with sync-v1; unsupported callback
features still fail closed. This gate rejects a skipped or silently downgraded
async path rather than treating Python fallback as Rust authority.
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
DEFAULT_NATIVE_REPORT = ROOT / ".codex" / "adr015-host-native-report.json"
DEFAULT_REPORT = ROOT / ".codex" / "adr015-phase4-runtime-gate.json"
NON_SLOW_MARKERS = "not slow and not manual and not llm_real and not requires_ollama"

GROUPS: dict[str, tuple[tuple[str, ...], bool]] = {
    "durable-worker": (
        (
            "tests/runtime/test_rust_runtime_authority.py",
            "tests/runtime/test_rust_runtime_python_worker.py",
            "tests/runtime/test_rust_runtime_worker_handoff.py",
            "tests/runtime/test_rust_runtime_checkpoint_slice.py",
        ),
        False,
    ),
    "bijection": (
        (
            "tests/runtime/test_sync_runtime_bijection_contract.py",
            "tests/runtime/test_async_runtime_bijection_contract.py",
        ),
        False,
    ),
    "bridge": (("tests/runtime/test_runtime_parity_bridge_contract.py",), False),
    "suspend-terminal": (
        (
            "tests/runtime/test_workflow_suspend_resume.py",
            "tests/runtime/test_workflow_terminal_status.py",
        ),
        False,
    ),
    "postgres": (("tests/runtime/test_rust_runtime_postgres.py",), True),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-report", type=Path, default=DEFAULT_NATIVE_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--group", action="append", choices=tuple(GROUPS))
    return parser.parse_args()


def _load_native_report(path: Path, source_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "missing current native report; run scripts/host_native_wheel_builder.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "adr015-host-native/v1":
        raise RuntimeError("native report schema is unsupported")
    if payload.get("candidate_source_sha256") != source_sha256:
        raise RuntimeError("native report does not match current candidate source")
    extension = payload.get("extension")
    if not isinstance(extension, str) or not Path(extension).is_file():
        raise RuntimeError("native report extension does not exist")
    if payload.get("smoke", {}).get("sqlite_transaction_id_abi") is not True:
        raise RuntimeError("native report lacks current SQLite transaction ABI smoke")
    return payload


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
            if case.find("failure") is not None:
                totals["failures"] += 1
            if case.find("error") is not None:
                totals["errors"] += 1
            if case.find("skipped") is not None:
                totals["skipped"] += 1
    return totals


def _run_group(
    *,
    python: Path,
    name: str,
    tests: tuple[str, ...],
    output_root: Path,
    live_postgres: bool,
) -> dict[str, Any]:
    junit = output_root / f"{name}.junit.xml"
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "-m",
        NON_SLOW_MARKERS,
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
        and (not live_postgres or counts["skipped"] == 0)
    )
    return {
        "name": name,
        "tests": list(tests),
        "live_postgres_required": live_postgres,
        "command": command,
        "returncode": completed.returncode,
        "junit": str(junit),
        "counts": counts,
        "status": "passed" if passed else "failed",
    }


def run_gate(
    *,
    python: Path,
    native_report: Path,
    output_root: Path,
    groups: tuple[str, ...],
) -> dict[str, Any]:
    source_sha256, source_file_count = candidate_source_fingerprint(ROOT)
    native = _load_native_report(native_report, source_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    reports = [
        _run_group(
            python=python,
            name=name,
            tests=GROUPS[name][0],
            output_root=output_root,
            live_postgres=GROUPS[name][1],
        )
        for name in groups
    ]
    return {
        "schema": "adr015-phase4-runtime-gate/v1",
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
        "native_extension": native["extension"],
        "native_extension_sha256": native["extension_sha256"],
        "async_rust_authority": "async-v2-worker-protocol",
        "groups": reports,
        "status": "passed" if all(group["status"] == "passed" for group in reports) else "failed",
    }


def main() -> int:
    args = _args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")
    selected = tuple(args.group or tuple(GROUPS))
    report_path = args.report.expanduser().resolve()
    result = run_gate(
        python=python,
        native_report=args.native_report.expanduser().resolve(),
        output_root=report_path.parent / "adr015-phase4-runtime-gate",
        groups=selected,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
