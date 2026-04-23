from __future__ import annotations

import asyncio
import inspect
import sys
import threading
from typing import Any


async def _await_any(value: Any) -> Any:
    return await value


def _run_coro_blocking(coro: Any) -> Any:
    if sys.platform == "win32":
        runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
        try:
            return runner.run(_await_any(coro))
        finally:
            runner.close()
    return asyncio.run(_await_any(coro))


def run_sync_or_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coro_blocking(value)
    return value


def run_awaitable_blocking(awaitable: Any) -> Any:
    if not inspect.isawaitable(awaitable):
        return awaitable
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coro_blocking(awaitable)

    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["result"] = _run_coro_blocking(awaitable)
        except BaseException as exc:  # pragma: no cover - propagated below
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")
