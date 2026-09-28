# SQLite Python/Rust Interoperability Investigation

**Scope:** PR #42, baseline `c1b8649`
**Evidence preserved:** 2026-09-28

Raw instrumented Linux run remains in Docker volume
`kogwistar-linux-pytest-tmp:/tmp/kogwistar/sqlite-architectural-cpython312.log`;
this report preserves the CI failures, runtime versions/source ID, extended
SQLite error, WAL/SHM snapshots, and the shared-versus-bundled experiments.

## Architectural contract

Both Python `sqlite3` and Rust `rusqlite` may be loaded in one process. Each
execution context selects exactly one implementation (`python` or `rust`) and
cannot switch until it ends. Different implementations cannot hold overlapping
connections to the same database in one process. Unused imports are allowed;
this contract does not require shared-library linking.

## Original CI failures

The CPython 3.12, 3.13, and 3.14 CI reports contained three failures in the
Python/Rust SQLite differential tests:

- `test_sqlite_python_to_rust_and_rust_to_python_queue_contract`
- `test_python_sqlite_then_rust_reads_and_writes_actual_database`
- `test_rust_python_sqlite_handoff_repeated_commits_are_strictly_visible`

Reported exception was `sqlite3.OperationalError: disk I/O error`. A prior
related failure in `test_rust_sqlite_then_python_initializes_reads_writes_aliases_and_cursors`
reported stale retained-event sequence (`1`, expected `2`).

## Collected diagnostics

On Linux CPython 3.12, Python reported SQLite `3.46.1`. With the Rust extension
temporarily built against the system SQLite shared library, dynamic-link
inspection showed both Python and the extension resolving
`/lib/x86_64-linux-gnu/libsqlite3.so.0`; the three then-failing interop tests
passed in that experiment.

After restoring the repository's `rusqlite` `bundled` feature and rebuilding,
the same Linux CPython 3.12 tests reproduced failures. Instrumented Python
SQLite errors reported extended error code `522`, which is
`SQLITE_IOERR_SHORT_READ`. At observed failures, `engine.db-wal` existed with
size zero and `engine.db-shm` was absent. Failures occurred on subsequent
Python reads after Rust operations. The captured error output included the SQL
and database sidecar state; no payload data was logged.

The instrumented bundled build reports Rust SQLite `3.50.2`, source ID
`2025-06-28 14:00:48 2af157d77fb1304a74176eaee7fbc7c7e932d946bf25325e9c26c91db19e3079`,
while Python reports `3.46.1`. Thus the system-linked experiment matched the
Python runtime version, while the bundled experiment used a different version
and a distinct SQLite library instance. Existing evidence cannot distinguish
which difference was causal by itself.

After adding context ownership and deterministic handle cleanup, the same
CPython 3.12 Linux environment passed 31 context/interoperability tests with
both libraries still distinct (`3.46.1` / `3.50.2`). Instrumentation observed
Python close/checkpoint sidecar transitions and Rust opens/closes without any
`SQLITE_IOERR_SHORT_READ`. This validates the repair direction for that test
set; CPython 3.13 and 3.14 remain mandatory CI matrix checks.

These observations establish an architectural hole: before the repair,
supported paths did not enforce execution-context binding or arbitrate every
cross-implementation handle lifetime for a database. They also establish that
the reproduced failure involved a WAL database, separate SQLite builds, and
different SQLite versions. They do **not** prove which individual connection
close/overlap triggered each historical short read; the baseline did not record
enough per-handle events to isolate that low-level trigger. The original disk
I/O failures are therefore consistent with the unsafe WAL/SHM lifecycle, but
must not be attributed solely to bundled SQLite, library version, or cleanup
without that missing baseline evidence. The repair rejects overlapping
cross-implementation handles and allows sequential handoff while retaining
distinct SQLite builds.

## Code-level lifecycle findings

- `EngineSQLite.connect()` created Python connections. The standard sqlite3
  connection context manager commits/rolls back but does not close; callers
  retaining a connection beyond its transaction can keep it active.
- `EngineSQLite.transaction()` did close its own connection in `finally`.
- Rust SQLite operations outside explicit transactions open and drop a fresh
  `SqliteStore`; explicit transaction sessions are cached in the PyO3 layer.
- The Rust session cache was keyed only by database path, not execution
  context. It could conflate sessions from distinct contexts.
- Authority was selected while constructing backend facades, but direct
  bridge/native entrypoints did not previously enforce an immutable ambient
  selection or arbitrate cross-implementation open-connection lifetimes.
- Existing `_rust_bridge.py` rejected a Rust bridge call during an active
  Python transaction on the same path, but this did not cover all direct native
  calls, all connection lifetimes, or non-transactional overlap.

## Current repair direction and remaining proof

Implementation adds context-bound selection, native entrypoint checks,
process-local database ownership, explicit Python connection close behavior,
and context-scoped Rust cache keys. Diagnostics are opt-in through
`KOGWISTAR_SQLITE_DIAGNOSTICS=1`; they record implementation, context/database
identity, SQLite versions, handle lifecycle, transaction boundaries, error
codes, SQL statement fingerprints (not SQL text), and WAL/SHM metadata, never
row contents. CPython 3.12 passed locally;
mandatory native CI covers 3.12, 3.13, and 3.14.

## Validation record

- Linux CPython 3.12 Docker run with native extension: 31 context/interoperability
  tests passed; both SQLite builds remained distinct.
- Windows CPython 3.13 focused context/interoperability run: 32 passed.
- Provider-free CI-marked suite on Windows CPython 3.13: 952 passed, 1 skipped,
  1210 deselected. The first run exposed one stale test expectation for
  non-transaction Rust session reuse; expectation was corrected to require an
  ephemeral connection, then the full run passed.
- Rust SQLite and PyO3 crate tests: 22 SQLite-store tests passed (1 ignored),
  3 PyO3 tests passed. Rust formatting, Python Ruff, and native-wheel smoke
  passed.
- CPython 3.14 and cross-platform hosted matrix remain CI acceptance checks;
  no local result is claimed for them.
