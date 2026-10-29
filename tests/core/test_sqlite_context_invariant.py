from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from pathlib import Path
from typing import Any, TypeVar

import pytest

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.sqlite_context import (
    SQLiteContextError,
    acquire_sqlite_database,
    independent_sqlite_execution_context,
    native_sqlite_enter,
    native_sqlite_exit,
    select_sqlite_implementation,
    sqlite_execution_context,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]
T = TypeVar("T")


def _assert_code(error: BaseException, code: str) -> None:
    assert getattr(error, "code", None) == code


def _run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


def test_context_binding_is_immutable_and_nested_calls_inherit() -> None:
    with sqlite_execution_context("python"):
        assert select_sqlite_implementation("python").implementation == "python"
        with pytest.raises(SQLiteContextError) as mismatch:
            select_sqlite_implementation("rust")
        _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
        with pytest.raises(SQLiteContextError):
            with sqlite_execution_context("rust"):
                pass


def test_native_reused_session_can_own_one_transaction_at_a_time(
    tmp_path: Path,
) -> None:
    database = str(tmp_path / "session.db")
    with sqlite_execution_context("rust") as binding:
        context_id, lease = native_sqlite_enter(database, "open_init", True)
        assert context_id == binding.context_id
        assert lease is None
        native_sqlite_exit(database, context_id, "open_init", True, True)

        context_id, lease = native_sqlite_enter(
            database, "begin_transaction", True
        )
        assert context_id == binding.context_id
        assert lease is None
        with pytest.raises(SQLiteContextError) as duplicate:
            native_sqlite_enter(database, "begin_transaction", True)
        _assert_code(duplicate.value, "KOGWISTAR_SQLITE_TRANSACTION_ALREADY_ACTIVE")

        native_sqlite_exit(database, context_id, "commit_transaction", True, True)
        context_id, lease = native_sqlite_enter(
            database, "get_named_projection", True
        )
        assert context_id == binding.context_id
        assert lease is None

        native_sqlite_enter(database, "close", True)
        native_sqlite_exit(database, context_id, "close", True, True)

    with sqlite_execution_context("python"):
        lease = acquire_sqlite_database(database, "python")
        lease.release()


def test_python_connection_closes_and_releases_database_ownership(tmp_path: Path) -> None:
    engine = EngineSQLite(tmp_path)
    path = engine.db_path
    engine.ensure_initialized()
    with sqlite_execution_context("python"):
        with engine.connect() as connection:
            connection.execute("CREATE TABLE close_check (value INTEGER)")
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    with sqlite_execution_context("rust"):
        lease = acquire_sqlite_database(path, "rust")
        lease.release()


def test_overlapping_python_and_rust_database_ownership_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    python_lease = acquire_sqlite_database(path, "python")
    try:
        with sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(path, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        python_lease.release()


def test_sqlite_authority_lock_rejects_another_process(tmp_path: Path) -> None:
    path = tmp_path / "cross-process.sqlite"
    child_code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from kogwistar.engine_core.sqlite_context import acquire_sqlite_database\n"
        "lease = acquire_sqlite_database(Path(sys.argv[1]), 'python')\n"
        "print('ready', flush=True)\n"
        "sys.stdin.read()\n"
        "lease.release()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(SQLiteContextError) as conflict:
            acquire_sqlite_database(path, "python")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_PROCESS_AUTHORITY_CONFLICT")
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=10)


def test_sqlite_authority_recovers_after_owner_process_crash(tmp_path: Path) -> None:
    path = tmp_path / "crashed-owner.sqlite"
    child_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from kogwistar.engine_core.sqlite_context import acquire_sqlite_database\n"
        "lease = acquire_sqlite_database(Path(sys.argv[1]), 'python')\n"
        "print('ready', flush=True)\n"
        "sys.stdin.read()\n"
        "lease.release()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == "ready"
        child.kill()
        child.wait(timeout=10)
        assert child.returncode is not None

        with sqlite_execution_context("rust"):
            lease = acquire_sqlite_database(path, "rust")
            lease.release()
    finally:
        if child.poll() is None:
            child.kill()
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=10)


@pytest.mark.requires_rust
@pytest.mark.parametrize("owner_implementation", ["python", "rust"])
def test_other_store_recovers_after_owner_process_crash(
    tmp_path: Path,
    owner_implementation: str,
) -> None:
    database = tmp_path / "rust-owner-crash" / "engine.db"
    ready_path = tmp_path / "rust-owner-ready"
    child_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from kogwistar.engine_core.engine_sqlite import EngineSQLite\n"
        "from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite\n"
        "from kogwistar.engine_core.sqlite_context import independent_sqlite_execution_context\n"
        "implementation = sys.argv[1]\n"
        "database = Path(sys.argv[2])\n"
        "connection = None\n"
        "with independent_sqlite_execution_context(implementation):\n"
        "    engine = EngineSQLite(database.parent) if implementation == 'python' else RustEngineSQLite(database.parent)\n"
        "    engine.ensure_initialized()\n"
        "    if implementation == 'python':\n"
        "        connection = engine.connect()\n"
        "    assert engine.next_global_seq() == 1\n"
        "Path(sys.argv[3]).write_text('ready', encoding='utf-8')\n"
        "sys.stdin.read()\n"
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            owner_implementation,
            str(database),
            str(ready_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if ready_path.exists():
                break
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout else ""
                raise AssertionError(f"Rust owner exited before readiness: {output}")
            time.sleep(0.05)
        else:
            child.kill()
            output, _ = child.communicate(timeout=10)
            raise AssertionError(f"Rust owner did not become ready: {output}")

        child.kill()
        child.wait(timeout=10)
        assert child.returncode is not None

        successor_implementation = "rust" if owner_implementation == "python" else "python"
        successor = (
            RustEngineSQLite(database.parent)
            if successor_implementation == "rust"
            else EngineSQLite(database.parent)
        )
        with sqlite_execution_context(successor_implementation):
            successor.ensure_initialized()
            assert successor.current_global_seq() == 1
            assert successor.next_global_seq() == 2
            assert successor.current_global_seq() == 2
    finally:
        if child.poll() is None:
            child.kill()
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=10)


@pytest.mark.ci_full
def test_python_authority_rejects_standalone_rust_process(
    tmp_path: Path,
) -> None:
    """Exercise the shared lock-file contract across real language processes."""
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")

    path = tmp_path / "python-owned-rust-child.sqlite"
    ready_path = Path(f"{path}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(path),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
            "KOGWISTAR_SQLITE_LOCK_EXPECT_CONFLICT": "1",
        }
    )
    with sqlite_execution_context("python"):
        lease = acquire_sqlite_database(path, "python")
        try:
            child = subprocess.Popen(
                command,
                cwd=manifest.parent.parent,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for _ in range(120):
                    if ready_path.exists():
                        break
                    if child.poll() is not None:
                        output = child.stdout.read() if child.stdout else ""
                        raise AssertionError(
                            f"Rust lock child exited before readiness: {output}"
                        )
                    time.sleep(0.5)
                else:
                    child.kill()
                    output, _ = child.communicate(timeout=10)
                    raise AssertionError(
                        f"Rust lock child did not become ready: {output}"
                    )
            finally:
                if child.stdin is not None:
                    child.stdin.close()
                    child.stdin = None
                child.wait(timeout=30)
                output = child.stdout.read() if child.stdout else ""
                assert child.returncode == 0, output
        finally:
            lease.release()


@pytest.mark.ci_full
def test_python_authority_rejects_rust_process_through_hardlink_alias(
    tmp_path: Path,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")
    database = tmp_path / "python-owned-hardlink.sqlite"
    alias = tmp_path / "rust-hardlink.sqlite"
    with sqlite3.connect(database):
        pass
    try:
        os.link(database, alias)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    ready_path = Path(f"{database}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(alias),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
            "KOGWISTAR_SQLITE_LOCK_EXPECT_CONFLICT": "1",
        }
    )
    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    with sqlite_execution_context("python"):
        lease = acquire_sqlite_database(database, "python")
        try:
            child = subprocess.run(
                command,
                cwd=manifest.parent.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
                check=False,
            )
            assert child.returncode == 0, child.stdout
        finally:
            lease.release()


@pytest.mark.ci_full
def test_python_process_rejects_rust_authority_through_hardlink_alias(
    tmp_path: Path,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")
    database = tmp_path / "rust-owned-hardlink.sqlite"
    alias = tmp_path / "python-hardlink.sqlite"
    ready_path = Path(f"{database}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    sqlite3.connect(database).close()
    try:
        os.link(database, alias)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(database),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
        }
    )
    child = subprocess.Popen(
        command,
        cwd=manifest.parent.parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(120):
            if ready_path.exists():
                break
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout else ""
                raise AssertionError(
                    f"Rust lock child exited before readiness: {output}"
                )
            time.sleep(0.5)
        else:
            child.kill()
            output, _ = child.communicate(timeout=10)
            raise AssertionError(f"Rust lock child did not become ready: {output}")

        with sqlite_execution_context("python"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(alias, "python")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_PROCESS_AUTHORITY_CONFLICT")
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=30)
        output = child.stdout.read() if child.stdout else ""
        assert child.returncode == 0, output


@pytest.mark.ci_full
def test_rust_authority_rejects_python_process(tmp_path: Path) -> None:
    """Exercise the reverse direction of the shared authority contract."""
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")

    path = tmp_path / "rust-owned-python-child.sqlite"
    ready_path = Path(f"{path}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(path),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
        }
    )
    child = subprocess.Popen(
        command,
        cwd=manifest.parent.parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(120):
            if ready_path.exists():
                break
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout else ""
                raise AssertionError(f"Rust lock child exited before readiness: {output}")
            time.sleep(0.5)
        else:
            child.kill()
            output, _ = child.communicate(timeout=10)
            raise AssertionError(f"Rust lock child did not become ready: {output}")

        with sqlite_execution_context("python"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(path, "python")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_PROCESS_AUTHORITY_CONFLICT")
    finally:
        if child.stdin is not None:
            child.stdin.close()
            child.stdin = None
        child.wait(timeout=30)
        output = child.stdout.read() if child.stdout else ""
        assert child.returncode == 0, output


@pytest.mark.ci_full
@pytest.mark.skipif(os.name == "nt", reason="Windows may not replace an open SQLite file")
def test_python_authority_rejects_rust_process_after_database_path_replacement(
    tmp_path: Path,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")
    database = tmp_path / "python-owned-replaced.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE original_probe (value INTEGER)")
    replacement = tmp_path / "replacement.sqlite"
    with sqlite3.connect(replacement) as connection:
        connection.execute("CREATE TABLE replacement_probe (value INTEGER)")

    ready_path = Path(f"{database}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(database),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
            "KOGWISTAR_SQLITE_LOCK_EXPECT_CONFLICT": "1",
        }
    )

    with sqlite_execution_context("python"):
        owner = acquire_sqlite_database(database, "python")
        try:
            os.replace(replacement, database)
            child = subprocess.run(
                command,
                cwd=manifest.parent.parent,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
                check=False,
            )
            assert child.returncode == 0, child.stdout
        finally:
            owner.release()


@pytest.mark.ci_full
@pytest.mark.skipif(os.name == "nt", reason="Windows may not replace an open SQLite file")
def test_rust_authority_rejects_python_after_database_path_replacement(
    tmp_path: Path,
) -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo is required for the cross-language process check")
    database = tmp_path / "rust-owned-replaced.sqlite"
    sqlite3.connect(database).close()
    replacement = tmp_path / "replacement.sqlite"
    with sqlite3.connect(replacement) as connection:
        connection.execute("CREATE TABLE replacement_probe (value INTEGER)")

    ready_path = Path(f"{database}.ready")
    manifest = Path(__file__).resolve().parents[2] / "rust" / "Cargo.toml"
    command = [
        cargo,
        "test",
        "--manifest-path",
        str(manifest),
        "-p",
        "kogwistar-store-sqlite",
        "tests::authority::standalone_rust_process_cannot_open_an_authority_owned_database",
        "--",
        "--exact",
        "--nocapture",
    ]
    env = os.environ.copy()
    env.update(
        {
            "KOGWISTAR_SQLITE_LOCK_CHILD_PATH": str(database),
            "KOGWISTAR_SQLITE_LOCK_CHILD_READY": str(ready_path),
        }
    )
    child = subprocess.Popen(
        command,
        cwd=manifest.parent.parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(120):
            if ready_path.exists():
                break
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout else ""
                raise AssertionError(f"Rust lock child exited before readiness: {output}")
            time.sleep(0.5)
        else:
            child.kill()
            output, _ = child.communicate(timeout=10)
            raise AssertionError(f"Rust lock child did not become ready: {output}")

        os.replace(replacement, database)
        with independent_sqlite_execution_context("python"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(database, "python")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_PROCESS_AUTHORITY_CONFLICT")
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait(timeout=30)
        output = child.stdout.read() if child.stdout else ""
        assert child.returncode == 0, output


def test_cross_implementation_acquire_race_grants_one_owner_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "racing-engine.db"
    barrier = threading.Barrier(2)
    both_attempted = threading.Event()
    attempt_lock = threading.Lock()
    attempt_count = 0
    outcomes: list[tuple[str, str]] = []

    def acquire(implementation: str) -> None:
        nonlocal attempt_count
        lease = None
        with independent_sqlite_execution_context(implementation):
            barrier.wait(timeout=5)
            try:
                lease = acquire_sqlite_database(path, implementation)
                outcome = "acquired"
            except SQLiteContextError as error:
                _assert_code(
                    error, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT"
                )
                outcome = "conflict"
            finally:
                with attempt_lock:
                    attempt_count += 1
                    if attempt_count == 2:
                        both_attempted.set()
            try:
                assert both_attempted.wait(timeout=5)
                outcomes.append((implementation, outcome))
            finally:
                if lease is not None:
                    lease.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(acquire, implementation)
            for implementation in ("python", "rust")
        ]
        for future in futures:
            future.result(timeout=10)

    assert sorted(outcome for _, outcome in outcomes) == ["acquired", "conflict"]


@pytest.mark.requires_rust
def test_cross_implementation_store_open_race_on_new_database_has_one_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "first-open-race" / "engine.db"
    python_engine = EngineSQLite(path.parent)
    rust_engine = RustEngineSQLite(path.parent)
    barrier = threading.Barrier(2)
    attempted = threading.Event()
    lock = threading.Lock()
    count = 0
    outcomes: list[tuple[str, str, Exception | None]] = []

    def initialize(implementation: str) -> None:
        nonlocal count
        connection: sqlite3.Connection | None = None
        with independent_sqlite_execution_context(implementation):
            barrier.wait(timeout=5)
            try:
                if implementation == "python":
                    python_engine.ensure_initialized()
                    connection = python_engine.connect()
                else:
                    rust_engine.ensure_initialized()
                outcome = (implementation, "acquired", None)
            except Exception as error:
                outcome = (implementation, "failed", error)
            finally:
                with lock:
                    count += 1
                    if count == 2:
                        attempted.set()
            try:
                assert attempted.wait(timeout=5)
                outcomes.append(outcome)
            finally:
                if connection is not None:
                    connection.close()
                if implementation == "rust" and outcome[1] == "acquired":
                    rust_engine.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(initialize, implementation)
            for implementation in ("python", "rust")
        ]
        for future in futures:
            future.result(timeout=20)

    assert sorted(outcome for _, outcome, _ in outcomes) == ["acquired", "failed"]
    failure = next(error for _, outcome, error in outcomes if outcome == "failed")
    assert failure is not None
    if isinstance(failure, SQLiteContextError):
        _assert_code(failure, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    else:
        assert "authority" in str(failure).lower() or "another implementation" in str(failure).lower()


@pytest.mark.requires_rust
def test_cross_process_store_open_race_on_new_database_has_one_owner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "cross-process-first-open" / "engine.db"
    start_path = tmp_path / "start-open-race"
    release_path = tmp_path / "release-open-race"
    child_code = r"""
import json
import sys
import time
from pathlib import Path
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.engine_core.sqlite_context import independent_sqlite_execution_context

implementation, database_arg, ready_arg, start_arg, release_arg, status_arg = sys.argv[1:]
database = Path(database_arg)
ready = Path(ready_arg)
start = Path(start_arg)
release = Path(release_arg)
status = Path(status_arg)
ready.touch()
while not start.exists():
    time.sleep(0.005)
connection = None
engine = None
try:
    with independent_sqlite_execution_context(implementation):
        if implementation == "python":
            engine = EngineSQLite(database.parent)
            engine.ensure_initialized()
            connection = engine.connect()
        else:
            engine = RustEngineSQLite(database.parent)
            engine.ensure_initialized()
    result = {"state": "acquired"}
except Exception as error:
    result = {
        "state": "conflict",
        "type": type(error).__name__,
        "code": getattr(error, "code", None),
        "message": str(error),
    }
status.write_text(json.dumps(result), encoding="utf-8")
if result["state"] == "acquired":
    while not release.exists():
        time.sleep(0.005)
    if connection is not None:
        connection.close()
    if implementation == "rust":
        with independent_sqlite_execution_context("rust"):
            engine.close()
"""
    implementations = ("python", "rust")
    processes = {
        implementation: subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_code,
                implementation,
                str(database),
                str(tmp_path / f"{implementation}.ready"),
                str(start_path),
                str(release_path),
                str(tmp_path / f"{implementation}.status.json"),
            ],
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for implementation in implementations
    }
    status_paths = {
        implementation: tmp_path / f"{implementation}.status.json"
        for implementation in implementations
    }
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if all((tmp_path / f"{name}.ready").exists() for name in implementations):
                break
            time.sleep(0.01)
        else:
            pytest.fail("both SQLite authority processes did not reach the start barrier")

        start_path.touch()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if all(path.exists() for path in status_paths.values()):
                break
            if any(
                process.poll() is not None and not status_paths[name].exists()
                for name, process in processes.items()
            ):
                break
            time.sleep(0.01)

        assert all(path.exists() for path in status_paths.values()), {
            name: process.poll() for name, process in processes.items()
        }
        results = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in status_paths.items()
        }
    finally:
        start_path.touch()
        release_path.touch()
        for name, process in processes.items():
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, f"{name}: stdout={stdout!r}, stderr={stderr!r}"

    assert sorted(result["state"] for result in results.values()) == [
        "acquired",
        "conflict",
    ], results
    conflict = next(result for result in results.values() if result["state"] == "conflict")
    assert conflict.get("code") == "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT" or (
        "authority" in conflict["message"].lower()
        or "another implementation" in conflict["message"].lower()
    ), conflict


def test_same_implementation_can_concurrently_acquire_same_database(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    first = acquire_sqlite_database(path, "python")
    second = acquire_sqlite_database(path, "python")
    second.release()
    first.release()


@pytest.mark.requires_rust
def test_same_implementation_async_contexts_share_database_normally(tmp_path: Path) -> None:
    async def scenario(implementation: str) -> list[int]:
        database = tmp_path / implementation
        engine = EngineSQLite(database)
        engine.ensure_initialized()
        rust_engine = RustEngineSQLite(database)
        with sqlite_execution_context("rust"):
            rust_engine.ensure_initialized()
        barrier = asyncio.Barrier(2)

        async def allocate() -> int:
            with independent_sqlite_execution_context(implementation):
                await barrier.wait()
                selected = engine if implementation == "python" else rust_engine
                return await asyncio.to_thread(selected.next_global_seq)

        return list(await asyncio.gather(allocate(), allocate()))

    for implementation in ("python", "rust"):
        assert sorted(_run_async(scenario(implementation))) == [1, 2]


@pytest.mark.parametrize("alias_kind", ["relative", "uri", "symlink", "hardlink"])
def test_database_aliases_share_ownership_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    path = tmp_path / "engine.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE identity_check (value INTEGER)")
    if alias_kind == "relative":
        monkeypatch.chdir(tmp_path)
        alias: str | Path = Path("engine.db")
    elif alias_kind == "uri":
        alias = f"file:{path.as_posix()}?mode=rw"
        path_lease = acquire_sqlite_database(path, "python")
        uri_lease = acquire_sqlite_database(alias, "python")
        try:
            assert (
                uri_lease._process_identity_lock_key
                == path_lease._process_identity_lock_key
            )
        finally:
            uri_lease.release()
            path_lease.release()
    elif alias_kind == "symlink":
        alias = tmp_path / "engine-link.db"
        try:
            os.symlink(path, alias)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links unavailable")
    else:
        alias = tmp_path / "engine-hardlink.db"
        try:
            os.link(path, alias)
        except OSError:
            pytest.skip("hard links unavailable")

    owner = acquire_sqlite_database(path, "python")
    try:
        with pytest.raises(SQLiteContextError) as conflict:
            acquire_sqlite_database(alias, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        owner.release()


def test_shared_memory_uri_query_order_is_canonicalized() -> None:
    owner = acquire_sqlite_database(
        "file:shared-memory?mode=memory&cache=shared", "python"
    )
    try:
        with pytest.raises(SQLiteContextError) as conflict:
            acquire_sqlite_database(
                "file:shared-memory?cache=shared&mode=memory", "rust"
            )
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
    finally:
        owner.release()


def test_shared_memory_uri_lock_does_not_create_files_in_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    owner = acquire_sqlite_database(
        "file:shared-memory?mode=memory&cache=shared", "python"
    )
    try:
        assert not list(tmp_path.glob("*.kogwistar-authority.lock"))
    finally:
        owner.release()


def test_private_memory_database_leases_are_context_scoped_without_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    with independent_sqlite_execution_context("python"):
        python_lease = acquire_sqlite_database(":memory:", "python")
    python_lease.refresh_file_identity(":memory:")
    try:
        with independent_sqlite_execution_context("rust"):
            rust_lease = acquire_sqlite_database(":memory:", "rust")
        try:
            assert python_lease._path_key != rust_lease._path_key
            assert not list(tmp_path.glob("*.kogwistar-authority.lock"))
        finally:
            rust_lease.release()
    finally:
        python_lease.release()


def test_database_replacement_is_rejected_while_old_identity_is_owned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "engine.db"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("CREATE TABLE old_database (value INTEGER)")
    connection.close()
    owner = acquire_sqlite_database(path, "python")
    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    with connection:
        connection.execute("CREATE TABLE new_database (value INTEGER)")
    connection.close()
    path.unlink()
    os.replace(replacement, path)
    try:
        with pytest.raises(SQLiteContextError) as changed:
            acquire_sqlite_database(path, "python")
        _assert_code(changed.value, "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED")
    finally:
        owner.release()


def test_database_deletion_is_rejected_while_identity_is_owned(tmp_path: Path) -> None:
    path = tmp_path / "engine.db"
    connection = sqlite3.connect(path)
    connection.close()
    owner = acquire_sqlite_database(path, "python")
    path.unlink()
    try:
        with pytest.raises(SQLiteContextError) as changed:
            acquire_sqlite_database(path, "python")
        _assert_code(changed.value, "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED")
    finally:
        owner.release()


@pytest.mark.requires_rust
def test_async_child_inherits_binding_and_independent_contexts_run_in_parallel(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with sqlite_execution_context("python"):
            async def child_switch() -> str:
                try:
                    select_sqlite_implementation("rust")
                except SQLiteContextError as error:
                    return error.code
                raise AssertionError("child task silently changed SQLite implementation")

            assert await asyncio.create_task(child_switch()) == (
                "KOGWISTAR_SQLITE_ENGINE_MISMATCH"
            )

        barrier = asyncio.Barrier(2)

        async def operate(implementation: str, database: Path) -> None:
            with independent_sqlite_execution_context(implementation):
                database.parent.mkdir(parents=True, exist_ok=True)
                if implementation == "python":
                    engine = EngineSQLite(database.parent)
                    await asyncio.to_thread(engine.ensure_initialized)
                    await barrier.wait()
                    await asyncio.to_thread(engine.next_global_seq)
                else:
                    engine = RustEngineSQLite(database.parent)
                    await asyncio.to_thread(engine.ensure_initialized)
                    await barrier.wait()
                    await asyncio.to_thread(engine.next_global_seq)

        await asyncio.gather(
            operate("python", tmp_path / "python" / "engine.db"),
            operate("rust", tmp_path / "rust" / "engine.db"),
        )

        with sqlite_execution_context("python"):
            with independent_sqlite_execution_context("rust"):
                assert select_sqlite_implementation("rust").implementation == "rust"
            assert select_sqlite_implementation("python").implementation == "python"

    _run_async(scenario())


def test_executor_context_must_be_copied_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "executor.db"
    with sqlite_execution_context("python"):
        owner = acquire_sqlite_database(path, "python")
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            def unpropagated() -> None:
                lease = acquire_sqlite_database(path, "rust")
                lease.release()

            with pytest.raises(SQLiteContextError) as missing:
                executor.submit(unpropagated).result()
            _assert_code(
                missing.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT"
            )
            assert executor.submit(
                context.run, select_sqlite_implementation, "python"
            ).result().implementation == "python"
        owner.release()


def test_cancellation_does_not_release_live_connection_ownership(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    allow_close = threading.Event()
    finished = threading.Event()
    engine = EngineSQLite(tmp_path)
    engine.ensure_initialized()

    def hold_connection() -> None:
        try:
            with independent_sqlite_execution_context("python"):
                connection = engine.connect()
                started.set()
                try:
                    allow_close.wait(timeout=10)
                finally:
                    connection.close()
        finally:
            finished.set()

    async def scenario() -> None:
        pending = asyncio.create_task(asyncio.to_thread(hold_connection))
        assert await asyncio.to_thread(started.wait, 10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with independent_sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                acquire_sqlite_database(engine.db_path, "rust")
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
        allow_close.set()
        assert await asyncio.to_thread(finished.wait, 10)
        with independent_sqlite_execution_context("rust"):
            lease = acquire_sqlite_database(engine.db_path, "rust")
            lease.release()

    _run_async(scenario())


@pytest.mark.requires_rust
def test_direct_native_entrypoint_cannot_bypass_context_guard(tmp_path: Path) -> None:
    from kogwistar import _rust

    path = tmp_path / "must-not-be-created.db"
    payload = json.dumps(
        {
            "path": str(path),
            "operation": {"kind": "open_init"},
        }
    )
    with sqlite_execution_context("python"):
        with pytest.raises(SQLiteContextError) as mismatch:
            _rust.store_sqlite_json(payload)
    _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
    assert not path.exists()


def test_rust_context_rejects_python_connection_before_open(tmp_path: Path) -> None:
    engine = EngineSQLite(tmp_path)
    with sqlite_execution_context("rust"):
        with pytest.raises(SQLiteContextError) as mismatch:
            engine.connect()
    _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
    assert not engine.db_path.exists()


@pytest.mark.requires_rust
def test_both_sqlite_libraries_may_be_loaded_for_disjoint_contexts(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    python_db = EngineSQLite(tmp_path / "python")
    python_db.ensure_initialized()
    with sqlite3.connect(python_db.db_path) as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)

    rust_path = tmp_path / "rust" / "engine.db"
    with sqlite_execution_context("rust"):
        result = _rust.store_sqlite_json(
            json.dumps({"path": str(rust_path), "operation": {"kind": "open_init"}})
        )
    assert json.loads(result) == {"initialized": True}


@pytest.mark.requires_rust
def test_direct_native_python_overlap_fails_before_database_mutation(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    python = EngineSQLite(tmp_path)
    path = python.db_path
    connection = python.connect()
    try:
        before = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        with sqlite_execution_context("rust"):
            with pytest.raises(SQLiteContextError) as conflict:
                _rust.store_sqlite_json(
                    json.dumps({"path": str(path), "operation": {"kind": "open_init"}})
                )
        _assert_code(conflict.value, "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT")
        after = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        assert after == before
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.requires_rust
def test_native_transaction_id_cannot_cross_contexts(tmp_path: Path) -> None:
    from kogwistar._rust import RustStoreValueError

    from kogwistar import _rust

    path = tmp_path / "transaction.db"
    transaction_id = "same-transaction-token"

    def native(operation: dict[str, object]) -> str:
        return _rust.store_sqlite_json(
            json.dumps(
                {
                    "path": str(path),
                    "transaction_id": transaction_id,
                    "operation": operation,
                }
            )
        )

    with sqlite_execution_context("rust"):
        native({"kind": "begin_transaction"})
        other_context = Context()
        with pytest.raises(RustStoreValueError) as mismatch:
            other_context.run(
                lambda: _run_in_sqlite_context(
                    "rust",
                    lambda: native(
                        {
                            "kind": "raw_append",
                            "namespace": "ns",
                            "event_id": "event",
                            "entity_kind": "node",
                            "entity_id": "node",
                            "op": "UPSERT",
                            "payload_json": "{}",
                        }
                    ),
                )
            )
        _assert_code(mismatch.value, "KOGWISTAR_SQLITE_ENGINE_MISMATCH")
        native({"kind": "rollback_transaction"})


@pytest.mark.requires_rust
def test_rejected_nested_native_begin_does_not_release_live_session(
    tmp_path: Path,
) -> None:
    from kogwistar import _rust

    path = tmp_path / "nested-transaction.db"
    def payload(kind: str) -> str:
        return json.dumps(
            {
                "path": str(path),
                "transaction_id": "outer-session",
                "operation": {"kind": kind},
            }
        )
    with sqlite_execution_context("rust"):
        _rust.store_sqlite_json(payload("begin_transaction"))
        with pytest.raises(SQLiteContextError) as nested:
            _rust.store_sqlite_json(payload("begin_transaction"))
        _assert_code(nested.value, "KOGWISTAR_SQLITE_TRANSACTION_ALREADY_ACTIVE")
        _rust.store_sqlite_json(payload("rollback_transaction"))


def _run_in_sqlite_context(implementation: str, function):
    with sqlite_execution_context(implementation):
        return function()
