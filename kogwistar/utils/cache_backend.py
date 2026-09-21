"""Small cache-provider abstraction shared by the runtime and tests.

Joblib remains the CPython default.  PyPy uses DiskCache because its
cloudpickle path is not compatible with the PyPy 3.11 runtime used by CI.
The provider can be selected explicitly with ``KOGWISTAR_CACHE_BACKEND``.
"""

from __future__ import annotations

import os
import hashlib
import pickle
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, Protocol, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")
CacheBackend = Literal["auto", "joblib", "diskcache", "none"]


class MemoryLike(Protocol):
    """Common cache surface used by Kogwistar call sites."""

    def cache(self, function: Callable[P, R] | None = None, **kwargs: Any) -> Any:
        ...


class CacheBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional cache provider is absent."""


class _FunctionWrapper:
    def __init__(self, function: Callable[P, R]) -> None:
        self._function = function
        wraps(function)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self(*args, **kwargs)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        clear = getattr(self._function, "clear", None)
        if clear is None:
            clear = getattr(self._function, "cache_clear", None)
        if callable(clear):
            clear(*args, **kwargs)

    def check_call_in_cache(self, *args: Any, **kwargs: Any) -> bool:
        checker = getattr(self._function, "check_call_in_cache", None)
        if callable(checker):
            return bool(checker(*args, **kwargs))
        return False


class _NoCacheMemory:
    def __init__(self, location: str | Path | None = None, **_: Any) -> None:
        self.location = location

    def clear(self, *_: Any, **__: Any) -> None:
        return None

    @overload
    def cache(self, function: Callable[P, R], **kwargs: Any) -> _FunctionWrapper: ...

    @overload
    def cache(self, function: None = None, **kwargs: Any) -> Callable[
        [Callable[P, R]], _FunctionWrapper
    ]: ...

    def cache(
        self,
        function: Callable[P, R] | None = None,
        **_: Any,
    ) -> _FunctionWrapper | Callable[[Callable[P, R]], _FunctionWrapper]:
        def decorate(fn: Callable[P, R]) -> _FunctionWrapper:
            return _FunctionWrapper(fn)

        return decorate(function) if function is not None else decorate


class _DiskCacheMemory:
    def __init__(self, location: str | Path, **_: Any) -> None:
        from diskcache import Cache

        self.location = location
        self._cache = Cache(str(location))

    def clear(self, *_: Any, **__: Any) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()

    @overload
    def cache(self, function: Callable[P, R], **kwargs: Any) -> _FunctionWrapper: ...

    @overload
    def cache(self, function: None = None, **kwargs: Any) -> Callable[
        [Callable[P, R]], _FunctionWrapper
    ]: ...

    def cache(
        self,
        function: Callable[P, R] | None = None,
        **kwargs: Any,
    ) -> _FunctionWrapper | Callable[[Callable[P, R]], _FunctionWrapper]:
        ignored = tuple(kwargs.get("ignore", ()))
        memoize = self._cache.memoize(ignore=ignored)

        def decorate(fn: Callable[P, R]) -> _FunctionWrapper:
            return _FunctionWrapper(memoize(fn))

        return decorate(function) if function is not None else decorate


def _requested_backend(backend: CacheBackend | None) -> CacheBackend:
    value = backend or os.environ.get("KOGWISTAR_CACHE_BACKEND", "auto")
    normalized = value.strip().lower()
    if normalized not in {"auto", "joblib", "diskcache", "none"}:
        raise ValueError(
            "KOGWISTAR_CACHE_BACKEND must be auto, joblib, diskcache, or none"
        )
    return normalized  # type: ignore[return-value]


def _selected_backend(requested: CacheBackend) -> CacheBackend:
    if requested != "auto":
        return requested
    return "diskcache" if sys.implementation.name == "pypy" else "joblib"


class CacheMemory:
    """Memory-compatible facade over Joblib, DiskCache, or no caching."""

    def __init__(
        self,
        location: str | Path | None = None,
        *,
        backend: CacheBackend | None = None,
        **kwargs: Any,
    ) -> None:
        requested = _requested_backend(backend)
        selected = _selected_backend(requested)
        self.location = location

        if selected == "none" or location is None:
            self.backend = "none"
            self._delegate: Any = _NoCacheMemory(location, **kwargs)
            return

        try:
            if selected == "diskcache":
                self._delegate = _DiskCacheMemory(location, **kwargs)
            else:
                from joblib import Memory as JoblibMemory

                self._delegate = JoblibMemory(location=location, **kwargs)
        except (ImportError, AttributeError) as exc:
            if requested != "auto":
                raise CacheBackendUnavailable(
                    f"cache backend {selected!r} is not installed"
                ) from exc
            self.backend = "none"
            self._delegate = _NoCacheMemory(location, **kwargs)
            return
        self.backend = selected

    def cache(self, function: Callable[P, R] | None = None, **kwargs: Any) -> Any:
        return self._delegate.cache(function, **kwargs)

    def clear(self, *args: Any, **kwargs: Any) -> None:
        clear = getattr(self._delegate, "clear", None)
        if callable(clear):
            clear(*args, **kwargs)

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()


Memory = CacheMemory


def _joblib_module() -> Any | None:
    if _selected_backend(_requested_backend(None)) != "joblib":
        return None
    try:
        import joblib
    except (ImportError, AttributeError):
        return None
    return joblib


def cache_hash(value: Any) -> str:
    """Return a stable cache key using Joblib or portable pickle hashing."""

    joblib = _joblib_module()
    if joblib is not None:
        return str(joblib.hash(value))
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def cache_dump(value: Any, filename: str | Path) -> Any:
    """Persist a staged cache value without importing Joblib on PyPy."""

    joblib = _joblib_module()
    if joblib is not None:
        return joblib.dump(value, filename)
    with Path(filename).open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
    return [str(filename)]


def cache_load(filename: str | Path) -> Any:
    """Load a staged cache value using the selected cache provider."""

    joblib = _joblib_module()
    if joblib is not None:
        return joblib.load(filename)
    with Path(filename).open("rb") as stream:
        return pickle.load(stream)


__all__ = [
    "CacheBackendUnavailable",
    "CacheMemory",
    "Memory",
    "MemoryLike",
    "cache_dump",
    "cache_hash",
    "cache_load",
]
