"""Prove each pending ADR-015 durable-store capability separately.

This gate is intentionally narrower than the Phase-3 aggregate gate.  A green
event-log test cannot authorize projections, queues, or graph/pgvector.  Every
group requires a live PostgreSQL/pgvector result with no skips.  It is local
evidence only; production canary evidence remains required before ownership is
promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    import adr015_phase3_store_gate as store_gate
    from adr015_source_identity import candidate_source_fingerprint
except ModuleNotFoundError:  # imported as a repository module in tests
    from scripts import adr015_phase3_store_gate as store_gate
    from scripts.adr015_source_identity import candidate_source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_REPORT = ROOT / ".codex" / "adr015-host-native-report.json"
DEFAULT_REPORT = ROOT / ".codex" / "adr015-phase3-capability-gate.json"

# Keep each capability's admission proof explicit.  The shared transaction
# suite is deliberately present in graph-pgvector: it verifies public Rust
# graph/meta authority keeps graph records, event history, and queue effects in
# one PostgreSQL transaction.
CAPABILITY_GROUPS: dict[str, tuple[str, ...]] = {
    "postgres-sequence-event-log": (
        "tests/core/test_rust_store_postgres_differential.py",
    ),
    "projections-snapshots-run-registry": (
        "tests/core/test_rust_store_projection_differential.py::test_postgres_named_projections_python_and_rust_interoperate",
        "tests/core/test_rust_store_projection_differential.py::test_postgres_rust_created_projection_schema_python_reads_and_atomic_batch",
        "tests/core/test_rust_store_run_registry_differential.py::test_postgres_run_registry_bidirectional_isolated_and_atomic",
        "tests/core/test_rust_store_workflow_history_differential.py::test_postgres_workflow_history_python_created_schema_and_atomic_batch",
        "tests/core/test_rust_store_workflow_history_differential.py::test_postgres_rust_created_schema_python_reads_history",
        "tests/core/test_rust_store_recovery_differential.py::test_postgres_atomic_bounded_recovery_rebuild_and_python_rust_interop",
    ),
    "queues-leases-lanes": (
        "tests/core/test_rust_store_index_jobs_differential.py::test_postgres_python_rust_queue_differential_live",
        "tests/core/test_rust_store_index_jobs_differential.py::test_postgres_queue_concurrency_reclaim_and_terminal_retry_live",
        "tests/core/test_rust_store_lane_messages_differential.py::test_postgres_lane_python_rust_and_concurrency",
    ),
    "graph-pgvector": (
        "tests/core/test_rust_store_graph_mutation_differential.py",
        "tests/core/test_rust_postgres_transaction_session.py",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-report", type=Path, default=DEFAULT_NATIVE_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--capability", action="append", choices=tuple(CAPABILITY_GROUPS))
    return parser.parse_args()


def run_gate(
    *,
    python: Path,
    native_report: Path,
    output_root: Path,
    capabilities: tuple[str, ...],
) -> dict[str, Any]:
    source_sha256, source_file_count = candidate_source_fingerprint(ROOT)
    native = store_gate._load_native_report(native_report, source_sha256)
    output_root.mkdir(parents=True, exist_ok=True)
    groups = [
        store_gate._run_group(
            python=python,
            name=capability,
            tests=CAPABILITY_GROUPS[capability],
            output_root=output_root,
            live_postgres=True,
        )
        for capability in capabilities
    ]
    return {
        "schema": "adr015-phase3-capability-gate/v1",
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
        "native_extension": native["extension"],
        "native_extension_sha256": native["extension_sha256"],
        "groups": groups,
        "production_authority": "not-promoted-local-evidence-only",
        "status": "passed" if all(group["status"] == "passed" for group in groups) else "failed",
    }


def main() -> int:
    args = _args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")
    report_path = args.report.expanduser().resolve()
    result = run_gate(
        python=python,
        native_report=args.native_report.expanduser().resolve(),
        output_root=report_path.parent / "adr015-phase3-capability-gate",
        capabilities=tuple(args.capability or tuple(CAPABILITY_GROUPS)),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
