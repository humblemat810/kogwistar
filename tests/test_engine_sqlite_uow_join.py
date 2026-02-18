import sqlite3
from pathlib import Path

import pytest

from graph_knowledge_engine.engine_sqlite import EngineSQLite


def test_sqlite_transaction_joins_active_conn_no_extra_connect(tmp_path: Path):
    persist_dir = tmp_path / "sqlite_meta"
    persist_dir.mkdir(parents=True, exist_ok=True)

    db = EngineSQLite(persistent_directory=persist_dir)

    # Count connect() calls
    orig_connect = db.connect
    calls = {"n": 0}

    def counted_connect() -> sqlite3.Connection:
        calls["n"] += 1
        return orig_connect()

    db.connect = counted_connect  # type: ignore[assignment]

    with db.transaction() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR REPLACE INTO t (id) VALUES ('outer')")

        # Nested transaction should reuse same conn; no extra db.connect()
        with db.transaction() as conn2:
            assert conn2 is conn
            conn2.execute("INSERT OR REPLACE INTO t (id) VALUES ('inner')")

    # Only ONE connect() call should have happened (outer transaction)
    assert calls["n"] == 1

    # Both inserts committed
    with db.transaction(immediate=False) as conn:
        rows = conn.execute("SELECT id FROM t ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["inner", "outer"]


def test_sqlite_nested_transaction_rolls_back_with_outer(tmp_path: Path):
    persist_dir = tmp_path / "sqlite_meta"
    persist_dir.mkdir(parents=True, exist_ok=True)

    db = EngineSQLite(persistent_directory=persist_dir)

    # Create schema outside the test transaction
    with db.transaction() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY)")

    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO t (id) VALUES ('outer')")

            # Nested join scope
            with db.transaction() as conn2:
                assert conn2 is conn
                conn2.execute("INSERT OR REPLACE INTO t (id) VALUES ('inner')")

            raise RuntimeError("boom")

    # After rollback, nothing should be persisted
    with db.transaction(immediate=False) as conn:
        rows = conn.execute("SELECT id FROM t").fetchall()
    assert rows == []