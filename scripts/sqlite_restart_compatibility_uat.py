"""Verify SQLite owner changes only across clean process restarts.

This bounded UAT deliberately proves persisted-file compatibility without
allowing Python's sqlite3 and the bundled Rust SQLite to stay open together.
It is suitable for a single VM or workstation; it is not a multi-process
SQLite authority or HA test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker-step", choices=("rust-write", "python-write", "rust-read"))
    parser.add_argument("--database", type=Path)
    return parser.parse_args()


def _store_for(mode: str, directory: Path):
    import os

    os.environ["KOGWISTAR_IMPL_META_STORE"] = mode
    from kogwistar.engine_core.rust_meta_sqlite import build_sqlite_meta_store

    return build_sqlite_meta_store(directory, "owner.sqlite")


def _worker(*, step: str, database: Path) -> int:
    import kogwistar
    import kogwistar._rust as native

    directory = database.parent
    if step == "rust-write":
        store = _store_for("rust", directory)
        if type(store).__name__ != "RustEngineSQLite":
            raise RuntimeError("Rust SQLite owner was not selected")
        store.ensure_initialized()
        store.set_index_applied_fingerprint(
            namespace="adr015-owner-uat",
            coalesce_key="owner.sqlite",
            applied_fingerprint="rust-first",
        )
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-owner-uat", coalesce_key="owner.sqlite"
        )
    elif step == "python-write":
        store = _store_for("python", directory)
        if type(store).__name__ != "EngineSQLite":
            raise RuntimeError("Python SQLite owner was not selected")
        store.ensure_initialized()
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-owner-uat", coalesce_key="owner.sqlite"
        )
        if observed != "rust-first":
            raise RuntimeError(f"Python restart read {observed!r}, expected 'rust-first'")
        store.set_index_applied_fingerprint(
            namespace="adr015-owner-uat",
            coalesce_key="owner.sqlite",
            applied_fingerprint="python-second",
        )
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-owner-uat", coalesce_key="owner.sqlite"
        )
    else:
        store = _store_for("rust", directory)
        if type(store).__name__ != "RustEngineSQLite":
            raise RuntimeError("Rust SQLite owner was not selected")
        store.ensure_initialized()
        observed = store.get_index_applied_fingerprint(
            namespace="adr015-owner-uat", coalesce_key="owner.sqlite"
        )
        if observed != "python-second":
            raise RuntimeError(f"Rust restart read {observed!r}, expected 'python-second'")
    print(
        json.dumps(
            {
                "step": step,
                "observed": observed,
                "package_file": kogwistar.__file__,
                "native_file": native.__file__,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_parent(args: argparse.Namespace) -> int:
    if args.workdir is None or args.report is None:
        raise SystemExit("--workdir and --report are required for parent execution")
    workdir = args.workdir.expanduser().resolve()
    database = workdir / "owner.sqlite"
    if database.exists():
        raise SystemExit(f"refusing existing database: {database}")
    workdir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for step in ("rust-write", "python-write", "rust-read"):
        result = subprocess.run(
            [
                args.python,
                str(Path(__file__).resolve()),
                "--worker-step",
                step,
                "--database",
                str(database),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        record = {
            "step": step,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        records.append(record)
        if result.returncode:
            break

    report = {
        "schema": "adr015-sqlite-owner-uat/v1",
        "python": str(Path(args.python).resolve()),
        "database": str(database),
        "steps": records,
        "status": "passed" if len(records) == 3 and all(item["returncode"] == 0 for item in records) else "failed",
        "scope": "fresh-process Rust/Python/Rust SQLite owner restart compatibility; no live mixed ownership",
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
        return _worker(step=args.worker_step, database=args.database.expanduser().resolve())
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
