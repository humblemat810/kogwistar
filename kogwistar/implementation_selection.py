from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeAlias, TypeVar, cast

from kogwistar.json_types import JsonObject

_IMPLEMENTATION_ENV_NAMES = (
    "KOGWISTAR_IMPL_MODE",
    "KOGWISTAR_IMPL_CONTRACTS",
    "KOGWISTAR_IMPL_META_STORE",
    "KOGWISTAR_IMPL_GRAPH_STORE",
    "KOGWISTAR_IMPL_RUNTIME",
    "KOGWISTAR_IMPL_SERVER",
    "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
)
ImplementationSelectionSnapshot: TypeAlias = tuple[tuple[str, str | None], ...]

_selection_snapshot: ContextVar[ImplementationSelectionSnapshot | None] = (
    ContextVar("kogwistar_implementation_selection", default=None)
)
_P = ParamSpec("_P")
_R = TypeVar("_R")


def implementation_env(name: str, default: str | None = None) -> str | None:
    """Read selector configuration from the active run snapshot, if present."""
    snapshot = _selection_snapshot.get()
    if snapshot is None:
        return os.getenv(name, default)
    for key, value in snapshot:
        if key == name:
            return default if value is None else value
    return default


def capture_implementation_selection() -> ImplementationSelectionSnapshot:
    """Capture all implementation selectors as one immutable configuration."""
    return tuple((name, os.getenv(name)) for name in _IMPLEMENTATION_ENV_NAMES)


def implementation_selection_metadata(
    snapshot: ImplementationSelectionSnapshot,
) -> JsonObject:
    """Project selector configuration to a versioned, non-secret trace object."""
    return {
        "schema_version": 1,
        "source": "engine_construction",
        "selectors": {name: value for name, value in snapshot},
    }


@contextmanager
def implementation_selection_scope(
    snapshot: ImplementationSelectionSnapshot | None = None,
) -> Iterator[None]:
    """Freeze selectors for a run and reject nested authority changes."""
    active_snapshot = _selection_snapshot.get()
    if active_snapshot is not None:
        if snapshot is not None and snapshot != active_snapshot:
            raise RuntimeError(
                "nested workflow runtime has a different implementation selection"
            )
        yield
        return

    token = _selection_snapshot.set(snapshot or capture_implementation_selection())
    try:
        yield
    finally:
        _selection_snapshot.reset(token)


def implementation_selection_bound(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Bind selector configuration across a sync or async execution boundary."""
    if inspect.iscoroutinefunction(inspect.unwrap(function)):
        async_function = cast(Callable[_P, Awaitable[object]], function)

        @wraps(function)
        async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> object:
            owner = args[0] if args else None
            snapshot = getattr(owner, "_implementation_selection_snapshot", None)
            with implementation_selection_scope(snapshot):
                return await async_function(*args, **kwargs)

        return cast(Callable[_P, _R], async_wrapper)

    @wraps(function)
    def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        owner = args[0] if args else None
        snapshot = getattr(owner, "_implementation_selection_snapshot", None)
        with implementation_selection_scope(snapshot):
            return function(*args, **kwargs)

    return sync_wrapper
