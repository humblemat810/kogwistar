from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "golden" / "database-ddl.json"


class _ClosingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connection

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _digest(values: Any) -> str:
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_baseline() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kogwistar-ddl-") as raw_root:
        root = Path(raw_root)
        sqlite_store = EngineSQLite(root, filename="baseline.sqlite")
        original_connect = sqlite_store.connect
        sqlite_store.connect = lambda: _ClosingConnection(original_connect())  # type: ignore[method-assign]
        sqlite_store.ensure_initialized()
        connection = sqlite3.connect(sqlite_store.db_path)
        try:
            rows = connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
                ORDER BY type, name
                """
            ).fetchall()
        finally:
            connection.close()
    sqlite_objects = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": _normalize_sql(str(row[3])),
        }
        for row in rows
    ]

    postgres_store = EnginePostgresMetaStore(engine=object(), schema="kogwistar")
    postgres_statements = [
        _normalize_sql(statement) for statement in postgres_store._bootstrap_statements()
    ]
    return {
        "baseline_version": 1,
        "sqlite": {
            "objects": sqlite_objects,
            "sha256": _digest(sqlite_objects),
        },
        "postgresql": {
            "schema_placeholder": "kogwistar",
            "statements": postgres_statements,
            "sha256": _digest(postgres_statements),
        },
    }


def _encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze ADR-015 database DDL baseline.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = _encoded(build_baseline())
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != encoded:
            print(f"database DDL baseline drift: {OUTPUT}")
            return 2
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
