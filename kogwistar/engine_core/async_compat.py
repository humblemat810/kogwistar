from __future__ import annotations

import asyncio
import inspect
from typing import Any


def run_sync_or_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    return value
