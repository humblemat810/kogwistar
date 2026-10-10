"""Install one built wheel, then verify its native contract in a clean process."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, default=Path("wheelhouse"))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _verify() -> int:
    import json
    import sqlite3
    import tempfile

    import kogwistar
    from kogwistar import _rust
    from kogwistar.engine_core.sqlite_context import (
        acquire_sqlite_database,
        sqlite_execution_context,
    )

    assert _rust.CONTRACT_VERSION == "1.0.0"
    assert _rust.stable_id_json('["node","golden"]')

    with tempfile.TemporaryDirectory(prefix="kogwistar-sqlite-contract-") as root:
        base = Path(root)
        rust_only = base / "rust" / "engine.db"
        with sqlite_execution_context("rust"):
            result = _rust.store_sqlite_json(
                json.dumps(
                    {"path": str(rust_only), "operation": {"kind": "open_init"}}
                )
            )
        assert json.loads(result) == {"initialized": True}

        same_db = base / "shared" / "engine.db"
        same_db.parent.mkdir(parents=True)
        python_lease = acquire_sqlite_database(same_db, "python")
        python_connection = sqlite3.connect(same_db)
        try:
            with sqlite_execution_context("rust"):
                try:
                    _rust.store_sqlite_json(
                        json.dumps(
                            {"path": str(same_db), "operation": {"kind": "open_init"}}
                        )
                    )
                except Exception as error:
                    assert getattr(error, "code", None) == (
                        "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT"
                    )
                else:
                    raise AssertionError("cross-engine overlapping DB use was accepted")
        finally:
            python_connection.close()
            python_lease.release()

        with sqlite_execution_context("python"):
            try:
                _rust.store_sqlite_json(
                    json.dumps(
                        {"path": str(rust_only), "operation": {"kind": "open_init"}}
                    )
                )
            except Exception as error:
                assert getattr(error, "code", None) == "KOGWISTAR_SQLITE_ENGINE_MISMATCH"
            else:
                raise AssertionError("Python-bound context invoked native Rust SQLite")

        # Closing the Python handle permits a safe sequential native handoff.
        with sqlite_execution_context("rust"):
            _rust.store_sqlite_json(
                json.dumps(
                    {"path": str(same_db), "operation": {"kind": "open_init"}}
                )
            )
    print(kogwistar.__file__, _rust.__file__, f"SQLite {sqlite3.sqlite_version}")
    return 0


def main() -> int:
    args = _args()
    if args.verify:
        return _verify()
    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {args.wheelhouse}, found {len(wheels)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[0])],
        check=True,
    )
    return subprocess.run(
        [sys.executable, "-P", str(Path(__file__).resolve()), "--verify"],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
