from __future__ import annotations

import asyncio
import inspect
import sys
import threading
from collections.abc import Awaitable
from typing import TypeVar, cast, overload

T = TypeVar("T")

async def _await_any(value: Awaitable[T]) -> T:
    return await value


def _run_coro_blocking(coro: Awaitable[T]) -> T:
    if sys.platform == "win32":
        runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
        try:
            return runner.run(_await_any(coro))
        finally:
            runner.close()
    return asyncio.run(_await_any(coro))


@overload
def run_sync_or_awaitable(value: Awaitable[T]) -> T | Awaitable[T]: ...


@overload
def run_sync_or_awaitable(value: T) -> T: ...


def run_sync_or_awaitable(value: T | Awaitable[T]) -> T | Awaitable[T]:
    if not inspect.isawaitable(value):
        return cast(T, value)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coro_blocking(cast(Awaitable[T], value))
    return cast(Awaitable[T], value)


def run_awaitable_blocking(awaitable: T | Awaitable[T]) -> T:
    if not inspect.isawaitable(awaitable):
        return cast(T, awaitable)
    awaitable_value = cast(Awaitable[T], awaitable)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coro_blocking(awaitable_value)

    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["result"] = _run_coro_blocking(awaitable_value)
        except BaseException as exc:  # pragma: no cover - propagated below
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise cast(BaseException, box["error"])
    return cast(T, box.get("result"))
