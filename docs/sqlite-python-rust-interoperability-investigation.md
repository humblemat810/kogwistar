# SQLite Python/Rust Interoperability Investigation

**Scope:** PR #42, baseline `c1b8649`
**Evidence preserved:** 2026-09-28

Raw instrumented Linux run remains in Docker volume
`kogwistar-linux-pytest-tmp:/tmp/kogwistar/sqlite-architectural-cpython312.log`;
this report preserves the CI failures, runtime versions/source ID, extended
SQLite error, WAL/SHM snapshots, and the shared-versus-bundled experiments.

## Architectural contract

Both Python `sqlite3` and Rust `rusqlite` may be loaded in one process. Each
framework-managed execution context selects exactly one implementation
(`python` or `rust`) and cannot switch until it ends. Kogwistar-mediated
connections are leased so different implementations cannot hold overlapping
connections to the same database. Direct connections opened outside Kogwistar
are not visible to that lease mechanism and must be closed before switching
implementations. Unused imports are allowed; this contract does not require
shared-library linking.

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
while Python reports `3.46.1`. This version difference initially appeared
causal, but a later test-level diagnosis supersedes that hypothesis.

## Current diagnosis

The CI failure was reproduced on Linux CPython 3.12 with the current bundled
Rust SQLite build. The failing differential tests opened a raw connection via
`with sqlite3.connect(...) as conn` and then proceeded to Rust operations. The
SQLite connection context manager commits or rolls back; it does not close the
connection. The Python connection therefore remained open while Rust opened
the same database, and because it was not created through `EngineSQLite`, it
did not participate in Kogwistar's lease or emit ownership events. This was an
invalid test of the supported sequential-handoff contract.

The differential tests now explicitly close those raw handles while retaining
their transaction semantics. With the bundled Rust SQLite 3.50.2 and Python
SQLite 3.46.1 still distinct, the SQLite-filtered differential suites passed
on Linux CPython 3.12 and Windows CPython 3.12.13, 3.13.3, and 3.14.3 (**21
passed, 9 deselected on each**). Ruff passes. The captured failing database copy passes
`PRAGMA integrity_check` and contains both expected events, further indicating
that the failure was tied to overlapping live handles rather than corrupt
persisted data. The system-linked result is not evidence that matching SQLite
versions are required.

This closes the reproduced test failure, not the entire DL-07 gate. The PyPy
3.11 CI lane is Python-authority-only and excludes `requires_rust`; the native
extension currently targets `abi3-py312`, so PyPy 3.11 native SQLite handoff is
not a supported topology and is not claimed. Hosted CI evidence for the fixed
cross-language suites and the remaining platform matrix are still required.

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

### Hard-link aliases and shared file identity

The process authority lock was initially keyed only by the caller's database
path. On Windows, a Python owner could hold one path while a Rust process opened
a hard-link alias to the same SQLite file; both obtained different path mutexes
and the second implementation was admitted. The reverse direction was
reproducible too. This was a real bypass of the intended single-authority
invariant, not just a theoretical concern.

Both implementations now keep their path lock and additionally lock by stable
file identity after the database exists. POSIX uses device/inode; Windows uses
the Win32 volume serial and file index, derived identically by Python and Rust.
Python acquires this identity lock as part of acquiring a lease for an existing
file, and refreshes it when a previously absent database becomes available.
The lock is released with the lease, and Windows process-crash recovery remains
provided by named-mutex handle lifetime.

Rust also creates a missing database file while holding its path lock, attaches
the file-identity lock, and only then opens/configures the SQLite connection.
This closes the window where connection setup could run before identity
ownership was established. Python opens the connection and refreshes identity
before issuing its connection pragmas.

Windows CPython 3.13 evidence on 2026-10-26: both Python-owner/Rust-alias and
Rust-owner/Python-alias subprocess regressions passed (**2 passed**); the full
`test_sqlite_context_invariant.py` module passed (**29 passed**), Rust SQLite
store tests passed (**23 passed, 1 ignored**), and Ruff plus Rust formatting
checks passed. This closes the tested Windows hard-link case only. Linux/macOS,
new-file creation races, and the supported Python CI matrix still require
verification; do not infer complete SQLite interoperability from this result.
PyPy 3.11 on Windows separately passed a direct path/SQLite-URI identity and
lease-release smoke. Its local installation lacks pytest and Pydantic, so this
is not a PyPy test-suite or cross-language parity result.

The persisted-data contract was also rechecked on Windows CPython 3.13:
`test_python_sqlite_then_rust_reads_and_writes_actual_database` and
`test_rust_python_sqlite_handoff_repeated_commits_are_strictly_visible` both
passed (**2 passed**), confirming actual Python/Rust data visibility in both
directions in addition to process exclusion.

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
