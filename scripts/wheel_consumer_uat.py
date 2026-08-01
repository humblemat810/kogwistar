"""Run bounded ADR-015 consumer UAT against an installed native wheel.

The parent process deliberately creates a new interpreter for each ownership
step.  It therefore proves clean-restart persisted SQLite compatibility, not
simultaneous ownership or a live handoff.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-step", choices=("rust-write", "python-write", "rust-read"))
    parser.add_argument("--database", type=Path)
    return parser.parse_args()


def _store(mode: str, directory: Path):
    os.environ["KOGWISTAR_IMPL_META_STORE"] = mode
    from kogwistar.engine_core.rust_meta_sqlite import build_sqlite_meta_store

    return build_sqlite_meta_store(directory, "consumer.sqlite")


def _identity(step: str, observed: str) -> dict[str, Any]:
    import kogwistar
    import kogwistar._rust as native

    return {
        "step": step,
        "observed": observed,
        "package_file": kogwistar.__file__,
        "native_file": native.__file__,
        "contract_version": native.CONTRACT_VERSION,
    }


def _worker(step: str, database: Path) -> int:
    directory = database.parent
    if step == "rust-write":
        from kogwistar.engine_core.rust_meta_sqlite import (
            RustEngineSQLite,
            RustSQLiteConnectionUnavailable,
        )

        store = _store("rust", directory)
        if not isinstance(store, RustEngineSQLite):
            raise RuntimeError("public Rust selector did not choose RustEngineSQLite")
        store.ensure_initialized()
        try:
            store.connect()
        except RustSQLiteConnectionUnavailable:
            pass
        else:
            raise RuntimeError("Rust owner exposed raw Python SQLite writer")
        try:
            with store.transaction():
                store.next_global_seq()
                raise RuntimeError("intentional rollback")
        except RuntimeError as error:
            if str(error) != "intentional rollback":
                raise
        if store.current_global_seq() != 0:
            raise RuntimeError("rolled-back transaction changed global sequence")
        store.set_index_applied_fingerprint(
            namespace="adr015-consumer-uat",
            coalesce_key="consumer.sqlite",
            applied_fingerprint="rust-first",
        )
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-consumer-uat", coalesce_key="consumer.sqlite"
        )
    elif step == "python-write":
        from kogwistar.engine_core.engine_sqlite import EngineSQLite

        store = _store("python", directory)
        if not isinstance(store, EngineSQLite):
            raise RuntimeError("public Python selector did not choose EngineSQLite")
        store.ensure_initialized()
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-consumer-uat", coalesce_key="consumer.sqlite"
        )
        if observed != "rust-first":
            raise RuntimeError(f"Python restart read {observed!r}, expected 'rust-first'")
        store.set_index_applied_fingerprint(
            namespace="adr015-consumer-uat",
            coalesce_key="consumer.sqlite",
            applied_fingerprint="python-second",
        )
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-consumer-uat", coalesce_key="consumer.sqlite"
        )
    else:
        from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite

        store = _store("rust", directory)
        if not isinstance(store, RustEngineSQLite):
            raise RuntimeError("public Rust selector changed after restart")
        store.ensure_initialized()
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-consumer-uat", coalesce_key="consumer.sqlite"
        )
        if observed != "python-second":
            raise RuntimeError(f"Rust restart read {observed!r}, expected 'python-second'")
    print(json.dumps(_identity(step, str(observed)), sort_keys=True))
    return 0


def _parent(args: argparse.Namespace) -> int:
    if args.workdir is None or args.report is None:
        raise SystemExit("--workdir and --report are required")
    workdir = args.workdir.expanduser().resolve()
    database = workdir / "consumer.sqlite"
    if database.exists():
        raise SystemExit(f"refusing existing database: {database}")
    workdir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    for step in ("rust-write", "python-write", "rust-read"):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-step", step,
             "--database", str(database)],
            check=False, capture_output=True, text=True,
        )
        steps.append({"step": step, "returncode": result.returncode,
                      "stdout": result.stdout.strip(), "stderr": result.stderr.strip()})
        if result.returncode:
            break
    report = {
        "schema": "adr015-consumer-uat/v1",
        "scope": "clean-wheel public selector, raw-writer closure, rollback, and Rust/Python/Rust restart compatibility; no live mixed ownership or HA",
        "python": str(Path(sys.executable).resolve()),
        "database": str(database),
        "steps": steps,
        "status": "passed" if len(steps) == 3 and all(item["returncode"] == 0 for item in steps) else "failed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    args = _args()
    if args.worker_step is not None:
        if args.database is None:
            raise SystemExit("--database is required with --worker-step")
        return _worker(args.worker_step, args.database.expanduser().resolve())
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
