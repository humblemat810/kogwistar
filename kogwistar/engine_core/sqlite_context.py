from __future__ import annotations

import inspect
import os
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


class SQLiteContextError(RuntimeError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SQLiteBinding:
    implementation: str
    context_id: str


_binding: ContextVar[SQLiteBinding | None] = ContextVar(
    "kogwistar_sqlite_binding", default=None
)
_registry_lock = threading.RLock()
_active_implementations: dict[tuple[object, ...], dict[str, int]] = {}
_active_path_identities: dict[str, tuple[tuple[object, ...] | None, int]] = {}
_native_session_leases: dict[tuple[str, str], SQLiteDatabaseLease] = {}


def _diagnostic(message: str) -> None:
    if os.getenv("KOGWISTAR_SQLITE_DIAGNOSTICS") == "1":
        print(f"[sqlite-ownership] {message}", file=sys.stderr, flush=True)


def current_sqlite_binding() -> SQLiteBinding | None:
    return _binding.get()


def select_sqlite_implementation(implementation: str) -> SQLiteBinding:
    if implementation not in {"python", "rust"}:
        raise ValueError("SQLite implementation must be 'python' or 'rust'")
    current = _binding.get()
    if current is not None:
        if current.implementation != implementation:
            raise SQLiteContextError(
                f"SQLite context is bound to {current.implementation}, not {implementation}",
                "KOGWISTAR_SQLITE_ENGINE_MISMATCH",
            )
        return current
    # Unscoped legacy calls are single-operation contexts. Applications that
    # span several calls use sqlite_execution_context at their task boundary.
    return SQLiteBinding(implementation, uuid.uuid4().hex)


@contextmanager
def sqlite_execution_context(implementation: str) -> Iterator[SQLiteBinding]:
    """Bind one implementation for this context; nested work cannot switch it."""
    current = _binding.get()
    if current is not None:
        if current.implementation != implementation:
            raise SQLiteContextError(
                f"SQLite context is bound to {current.implementation}, not {implementation}",
                "KOGWISTAR_SQLITE_ENGINE_MISMATCH",
            )
        yield current
        return
    binding = SQLiteBinding(implementation, uuid.uuid4().hex)
    token = _binding.set(binding)
    try:
        yield binding
    finally:
        _binding.reset(token)


@contextmanager
def independent_sqlite_execution_context(
    implementation: str,
) -> Iterator[SQLiteBinding]:
    """Start an explicitly independent context, even inside an inherited task."""
    if implementation not in {"python", "rust"}:
        raise ValueError("SQLite implementation must be 'python' or 'rust'")
    binding = SQLiteBinding(implementation, uuid.uuid4().hex)
    token = _binding.set(binding)
    try:
        yield binding
    finally:
        _binding.reset(token)


def reset_sqlite_context() -> None:
    """Reset ambient binding; intended for execution-boundary adapters/tests."""
    _binding.set(None)


@contextmanager
def sqlite_execution_for(*engines: object) -> Iterator[None]:
    implementations = {
        implementation
        for engine in engines
        if (metadata := getattr(engine, "metadata", None)) is not None
        and (implementation := getattr(metadata, "sqlite_implementation", None))
        is not None
    }
    if len(implementations) > 1:
        raise SQLiteContextError(
            "one workflow execution cannot mix Python and Rust SQLite authorities",
            "KOGWISTAR_SQLITE_ENGINE_MISMATCH",
        )
    if not implementations:
        yield
        return
    with sqlite_execution_context(next(iter(implementations))):
        yield


def sqlite_execution_bound(*engine_attributes: str):
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(self, *args, **kwargs):
                with sqlite_execution_for(
                    *(getattr(self, name) for name in engine_attributes)
                ):
                    return await function(self, *args, **kwargs)

            return async_wrapper

        @wraps(function)
        def sync_wrapper(self, *args, **kwargs):
            with sqlite_execution_for(
                *(getattr(self, name) for name in engine_attributes)
            ):
                return function(self, *args, **kwargs)

        return sync_wrapper

    return decorate


def _database_keys(database: str | os.PathLike[str]) -> set[tuple[object, ...]]:
    value = os.fsdecode(os.fspath(database))
    if value.startswith("file:"):
        parsed = urlsplit(value)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if query.get("mode") == ["memory"]:
            normalized_query = tuple(
                sorted((key, tuple(sorted(values))) for key, values in query.items())
            )
            return {("sqlite-uri-memory", parsed.netloc, parsed.path, normalized_query)}
        raw_path = unquote(parsed.path)
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        value = raw_path

    resolved = Path(value).expanduser().resolve(strict=False)
    normalized = os.path.normcase(os.path.abspath(os.fspath(resolved)))
    keys: set[tuple[object, ...]] = {("path", normalized)}
    try:
        stat = resolved.stat()
    except OSError:
        pass
    else:
        keys.add(("file-id", stat.st_dev, stat.st_ino))
    return keys


def _database_path_key(database: str | os.PathLike[str]) -> str:
    keys = _database_keys(database)
    for key in keys:
        if key[0] in {"path", "sqlite-uri"}:
            return str(key[1])
        if key[0] == "sqlite-uri-memory":
            return repr(key[1:])
    raise RuntimeError("SQLite database has no stable ownership key")


class SQLiteDatabaseLease:
    """Process-local ownership held for the lifetime of an actual connection."""

    def __init__(
        self,
        keys: set[tuple[object, ...]],
        implementation: str,
        path_key: str,
        file_identity: tuple[object, ...] | None,
    ) -> None:
        self._keys = keys
        self.implementation = implementation
        self._path_key = path_key
        self._file_identity = file_identity
        self._released = False

    def refresh_file_identity(self, database: str | os.PathLike[str]) -> None:
        new_keys = _database_keys(database)
        with _registry_lock:
            if self._released:
                raise RuntimeError("cannot refresh a released SQLite lease")
            file_identity = next(
                (key for key in new_keys if key[0] == "file-id"), None
            )
            active_identity = _active_path_identities.get(self._path_key)
            if (
                active_identity is not None
                and active_identity[0] is not None
                and active_identity[0] != file_identity
            ):
                raise SQLiteContextError(
                    "SQLite database file was replaced while a connection remained active",
                    "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED",
                )
            for key in new_keys - self._keys:
                owners = _active_implementations.get(key, {})
                if owners and self.implementation not in owners:
                    raise SQLiteContextError(
                        "SQLite database is already open through another implementation",
                        "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT",
                    )
            for key in new_keys - self._keys:
                owners = _active_implementations.setdefault(key, {})
                owners[self.implementation] = owners.get(self.implementation, 0) + 1
            self._keys.update(new_keys)
            if file_identity is not None and self._file_identity is None:
                _, count = active_identity or (None, 1)
                _active_path_identities[self._path_key] = (file_identity, count)
                self._file_identity = file_identity

    def release(self) -> None:
        with _registry_lock:
            if self._released:
                return
            self._released = True
            for key in self._keys:
                owners = _active_implementations.get(key)
                if not owners:
                    continue
                count = owners.get(self.implementation, 0) - 1
                if count > 0:
                    owners[self.implementation] = count
                else:
                    owners.pop(self.implementation, None)
                if not owners:
                    _active_implementations.pop(key, None)
            identity, count = _active_path_identities[self._path_key]
            if count <= 1:
                _active_path_identities.pop(self._path_key, None)
            else:
                _active_path_identities[self._path_key] = (identity, count - 1)
            _diagnostic(
                f"event=release implementation={self.implementation} "
                f"context={getattr(self, 'context_id', 'unknown')} "
                f"database={self._path_key}"
            )

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


def acquire_sqlite_database(
    database: str | os.PathLike[str], implementation: str
) -> SQLiteDatabaseLease:
    binding = select_sqlite_implementation(implementation)
    keys = _database_keys(database)
    path_key = _database_path_key(database)
    file_identity = next((key for key in keys if key[0] == "file-id"), None)
    with _registry_lock:
        active_identity = _active_path_identities.get(path_key)
        if (
            active_identity is not None
            and active_identity[0] is not None
            and active_identity[0] != file_identity
        ):
            raise SQLiteContextError(
                "SQLite database file was replaced while a connection remained active",
                "KOGWISTAR_SQLITE_DATABASE_IDENTITY_CHANGED",
            )
        for key in keys:
            owners = _active_implementations.get(key, {})
            if owners and implementation not in owners:
                raise SQLiteContextError(
                    "SQLite database is already open through another implementation",
                    "KOGWISTAR_SQLITE_DATABASE_ENGINE_CONFLICT",
                )
        for key in keys:
            owners = _active_implementations.setdefault(key, {})
            owners[implementation] = owners.get(implementation, 0) + 1
        identity, count = active_identity or (file_identity, 0)
        if identity is None and file_identity is not None:
            identity = file_identity
        _active_path_identities[path_key] = (identity, count + 1)
    lease = SQLiteDatabaseLease(keys, implementation, path_key, file_identity)
    lease.context_id = binding.context_id
    _diagnostic(
        f"event=acquire implementation={implementation} context={binding.context_id} "
        f"database={path_key} identity={file_identity}"
    )
    return lease


def native_sqlite_enter(
    database: str, operation: str, reuse_session: bool
) -> tuple[str, SQLiteDatabaseLease | None]:
    binding = select_sqlite_implementation("rust")
    session_key = (
        os.path.normcase(os.path.abspath(database)),
        binding.context_id,
    )
    persistent = reuse_session or operation == "begin_transaction"
    with _registry_lock:
        if persistent and session_key in _native_session_leases:
            if operation == "begin_transaction":
                raise SQLiteContextError(
                    "this SQLite execution context already owns a native transaction",
                    "KOGWISTAR_SQLITE_TRANSACTION_ALREADY_ACTIVE",
                )
            _native_session_leases[session_key].refresh_file_identity(database)
            return binding.context_id, None
        lease = acquire_sqlite_database(database, "rust")
        if persistent:
            _native_session_leases[session_key] = lease
            return binding.context_id, None
    return binding.context_id, lease


def native_sqlite_exit(
    database: str,
    context_id: str,
    operation: str,
    reuse_session: bool,
    succeeded: bool,
) -> None:
    # Failed begin must not leave a process-wide reservation without a live
    # native connection/session to own it.
    if operation == "begin_transaction" and not succeeded:
        with _registry_lock:
            lease = _native_session_leases.pop(
                (os.path.normcase(os.path.abspath(database)), context_id), None
            )
        if lease is not None:
            lease.release()
        return
    should_close = succeeded and (operation == "close" or (
        not reuse_session
        and operation in {"commit_transaction", "rollback_transaction"}
    ))
    if not should_close:
        return
    with _registry_lock:
        lease = _native_session_leases.pop(
            (os.path.normcase(os.path.abspath(database)), context_id), None
        )
    if lease is not None:
        lease.release()
