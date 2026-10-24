from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar, cast

_IMPLEMENTATION_ENV_NAMES = (
    "KOGWISTAR_IMPL_MODE",
    "KOGWISTAR_IMPL_CONTRACTS",
    "KOGWISTAR_IMPL_META_STORE",
    "KOGWISTAR_IMPL_GRAPH_STORE",
    "KOGWISTAR_IMPL_RUNTIME",
    "KOGWISTAR_IMPL_SERVER",
    "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
)
_selection_snapshot: ContextVar[tuple[tuple[str, str | None], ...] | None] = (
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


@contextmanager
def implementation_selection_scope() -> Iterator[None]:
    """Freeze language selectors for one run; nested work inherits the snapshot."""
    if _selection_snapshot.get() is not None:
        yield
        return

    snapshot = tuple((name, os.getenv(name)) for name in _IMPLEMENTATION_ENV_NAMES)
    token = _selection_snapshot.set(snapshot)
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
            with implementation_selection_scope():
                return await async_function(*args, **kwargs)

        return cast(Callable[_P, _R], async_wrapper)

    @wraps(function)
    def sync_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with implementation_selection_scope():
            return function(*args, **kwargs)

    return sync_wrapper
