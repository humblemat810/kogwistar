"""Small cache-provider abstraction shared by the runtime and tests.

Joblib remains the CPython default.  PyPy uses DiskCache because its
cloudpickle path is not compatible with the PyPy 3.11 runtime used by CI.
The provider can be selected explicitly with ``KOGWISTAR_CACHE_BACKEND``.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Generic, Literal, ParamSpec, Protocol, TypeVar, cast, overload

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
CacheBackend = Literal["auto", "joblib", "diskcache", "none"]


class CachedCallable(Protocol[P, R_co]):
    """Callable returned by a cache provider for a typed function."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...

    def clear(self, *args: object, **kwargs: object) -> None: ...

    def check_call_in_cache(self, *args: object, **kwargs: object) -> bool: ...


class CacheProvider(Protocol):
    """Minimal provider surface required by :class:`CacheMemory`."""

    @overload
    def cache(
        self, function: Callable[P, R], **kwargs: Any
    ) -> CachedCallable[P, R]: ...

    @overload
    def cache(
        self, function: None = None, **kwargs: Any
    ) -> Callable[[Callable[P, R]], CachedCallable[P, R]]: ...

    def cache(
        self, function: Callable[P, R] | None = None, **kwargs: Any
    ) -> CachedCallable[P, R] | Callable[[Callable[P, R]], CachedCallable[P, R]]: ...

    def clear(self, *args: object, **kwargs: object) -> None: ...

    def close(self) -> None: ...


class MemoryLike(CacheProvider, Protocol):
    """Common cache surface used by Kogwistar call sites."""

    """Named compatibility protocol for historical cache consumers."""


class CacheBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested optional cache provider is absent."""


class _FunctionWrapper(Generic[P, R]):
    def __init__(self, function: Callable[P, R]) -> None:
        self._function = function
        wraps(function)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self(*args, **kwargs)

    def clear(self, *args: object, **kwargs: object) -> None:
        clear = getattr(self._function, "clear", None)
        if clear is None:
            clear = getattr(self._function, "cache_clear", None)
        if callable(clear):
            clear(*args, **kwargs)

    def check_call_in_cache(self, *args: object, **kwargs: object) -> bool:
        checker = getattr(self._function, "check_call_in_cache", None)
        if callable(checker):
            return bool(checker(*args, **kwargs))
        return False


class _NoCacheMemory:
    def __init__(self, location: str | Path | None = None, **_: Any) -> None:
        self.location = location

    def clear(self, *_: object, **__: object) -> None:
        return None

    def close(self) -> None:
        return None

    @overload
    def cache(
        self, function: Callable[P, R], **kwargs: Any
    ) -> _FunctionWrapper[P, R]: ...

    @overload
    def cache(
        self, function: None = None, **kwargs: Any
    ) -> Callable[[Callable[P, R]], _FunctionWrapper[P, R]]: ...

    def cache(
        self,
        function: Callable[P, R] | None = None,
        **_: Any,
    ) -> _FunctionWrapper[P, R] | Callable[[Callable[P, R]], _FunctionWrapper[P, R]]:
        def decorate(fn: Callable[P, R]) -> _FunctionWrapper[P, R]:
            return _FunctionWrapper(fn)

        return decorate(function) if function is not None else decorate


class _DiskCacheMemory:
    def __init__(self, location: str | Path, **_: Any) -> None:
        from diskcache import Cache  # type: ignore[import-not-found]

        self.location = location
        self._cache = Cache(str(location))

    def clear(self, *_: object, **__: object) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()

    @overload
    def cache(
        self, function: Callable[P, R], **kwargs: Any
    ) -> _FunctionWrapper[P, R]: ...

    @overload
    def cache(
        self, function: None = None, **kwargs: Any
    ) -> Callable[[Callable[P, R]], _FunctionWrapper[P, R]]: ...

    def cache(
        self,
        function: Callable[P, R] | None = None,
        **kwargs: Any,
    ) -> _FunctionWrapper[P, R] | Callable[[Callable[P, R]], _FunctionWrapper[P, R]]:
        requested_ignore = tuple(kwargs.get("ignore", ()))

        def decorate(fn: Callable[P, R]) -> _FunctionWrapper[P, R]:
            # Joblib accepts parameter names for positional arguments. DiskCache
            # requires their positional indexes, so provide both forms where
            # possible. Keeping names also preserves keyword-call behavior.
            ignored: set[str | int] = set(requested_ignore)
            try:
                parameters = inspect.signature(fn).parameters.values()
            except (TypeError, ValueError):
                parameters = ()
            positional_index = 0
            for parameter in parameters:
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    if parameter.name in ignored:
                        ignored.add(positional_index)
                    positional_index += 1
            memoize = self._cache.memoize(ignore=tuple(ignored))
            return _FunctionWrapper(cast(Callable[P, R], memoize(fn)))

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
        self._delegate: CacheProvider

        if selected == "none" or location is None:
            self.backend = "none"
            self._delegate = _NoCacheMemory(location, **kwargs)
            return

        try:
            if selected == "diskcache":
                self._delegate = _DiskCacheMemory(location, **kwargs)
            else:
                from joblib import Memory as JoblibMemory

                self._delegate = cast(
                    CacheProvider, JoblibMemory(location=location, **kwargs)
                )
        except (ImportError, AttributeError) as exc:
            if requested != "auto":
                raise CacheBackendUnavailable(
                    f"cache backend {selected!r} is not installed"
                ) from exc
            self.backend = "none"
            self._delegate = _NoCacheMemory(location, **kwargs)
            return
        self.backend = selected

    def cache(
        self, function: Callable[P, R] | None = None, **kwargs: Any
    ) -> CachedCallable[P, R] | Callable[[Callable[P, R]], CachedCallable[P, R]]:
        return self._delegate.cache(function, **kwargs)

    def clear(self, *args: object, **kwargs: object) -> None:
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


def cache_hash(value: object) -> str:
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
