from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class EngineSQLite:
    """
    Lightweight SQLite helper for engine persistence.

    Responsibilities:
    - ensure persistent DB exists
    - allocate monotonic sequence numbers:
        * global counter (single row, no growth)
        * per-user counter (one row per user, no growth per increment)
    - provide safe transactional context

    Seq semantics:
    - Global counter is stored in table: global_seq(value)
        * current_global_seq() returns the last issued value (0 if never issued).
    - Per-user counters are stored in table: user_seq(user_id, value)
        * current_user_seq(user_id) returns last issued value for that user (0 if none).
    """

    def __init__(
        self,
        persistent_directory: Path,
        filename: str = "engine.db",
    ) -> None:
        self.persistent_directory = persistent_directory
        self.db_path = persistent_directory / filename

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """
        Create persistent directory and required tables if they do not exist.
        Safe to call multiple times.
        """
        self.persistent_directory.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_seq (
                    value INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO global_seq(rowid, value) VALUES (1, 0)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_seq (
                    user_id TEXT PRIMARY KEY,
                    value   INTEGER NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Connection / transaction helpers
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """
        Create a SQLite connection with sane defaults.
        """
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """
        Context manager for a short, safe transaction.

        Default: BEGIN IMMEDIATE (prevents writer starvation).

        Example:
            with db.transaction() as conn:
                ...
        """
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Global sequence
    # ------------------------------------------------------------------

    def next_global_seq(self) -> int:
        """
        Allocate the next global monotonic sequence number.
        """
        with self.transaction() as conn:
            return self.next_global_seq_conn(conn)

    def next_global_seq_conn(self, conn: sqlite3.Connection) -> int:
        """
        Allocate the next global sequence using an existing transaction.
        """
        (value,) = conn.execute(
            "UPDATE global_seq SET value = value + 1 RETURNING value"
        ).fetchone()
        return int(value)

    def current_global_seq(self) -> int:
        """
        Return the current global sequence value (last issued).
        """
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM global_seq WHERE rowid = 1").fetchone()
            return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Per-user sequence
    # ------------------------------------------------------------------

    def next_user_seq(self, user_id: str) -> int:
        """
        Allocate the next sequence number scoped to user_id.
        """
        with self.transaction() as conn:
            return self.next_user_seq_conn(conn, user_id)

    def next_user_seq_conn(self, conn: sqlite3.Connection, user_id: str) -> int:
        """
        Allocate the next user sequence using an existing transaction.
        """
        (value,) = conn.execute(
            """
            INSERT INTO user_seq(user_id, value)
            VALUES (?, 1)
            ON CONFLICT(user_id)
            DO UPDATE SET value = user_seq.value + 1
            RETURNING value
            """,
            (user_id,),
        ).fetchone()
        return int(value)

    def current_user_seq(self, user_id: str) -> int:
        """
        Return the current sequence value for user_id.
        """
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM user_seq WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    def set_user_seq(self, user_id: str, value: int) -> None:
        """
        Hard reset: set the counter for user_id to an exact value.

        WARNING: this allows seq reuse, which can collide with previously-issued seq
        that may already exist in persisted logs / Chroma / nodes tables.
        """
        if value < 0:
            raise ValueError("value must be >= 0")

        with self.transaction() as conn:
            self.set_user_seq_conn(conn, user_id, value)

    def set_user_seq_conn(self, conn: sqlite3.Connection, user_id: str, value: int) -> None:
        conn.execute(
            """
            INSERT INTO user_seq(user_id, value)
            VALUES (?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET value = excluded.value
            """,
            (user_id, value),
        )
    # ------------------------------------------------------------------
    # Usage examples
    # ------------------------------------------------------------------

    """
    Usage examples
    ==============

    Engine startup
    --------------
    >>> db = EngineSQLite(Path("/var/lib/my_engine"))
    >>> db.ensure_initialized()

    Global sequence (engine-wide ordering)
    -------------------------------------
    >>> seq = db.next_global_seq()
    >>> seq
    1

    >>> db.current_global_seq()
    1

    Per-user sequence
    -----------------
    >>> u1 = db.next_user_seq("alice")
    >>> u2 = db.next_user_seq("alice")
    >>> u1, u2
    (1, 2)

    >>> db.current_user_seq("alice")
    2

    Atomic allocation + insert
    --------------------------
    >>> with db.transaction() as conn:
    ...     seq = db.next_global_seq_conn(conn)
    ...     conn.execute(
    ...         "INSERT INTO nodes (node_id, seq) VALUES (?, ?)",
    ...         ("n1", seq),
    ...     )

    Notes
    -----
    - Global sequence uses a single-row counter (constant storage).
    - Per-user sequence uses one row per user (constant storage per user).
    - All increments are atomic and safe under concurrency.
    - BEGIN IMMEDIATE prevents writer starvation under read load.
    db.set_user_seq("alice", 17)
    """
