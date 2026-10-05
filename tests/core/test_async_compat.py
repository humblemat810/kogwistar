from __future__ import annotations

import asyncio

from kogwistar.engine_core.async_compat import (
    run_awaitable_blocking,
    run_sync_or_awaitable,
)


async def _value() -> int:
    return 7


def test_sync_or_awaitable_preserves_sync_value() -> None:
    assert run_sync_or_awaitable(3) == 3


def test_awaitable_blocking_resolves_outside_event_loop() -> None:
    assert run_awaitable_blocking(_value()) == 7


def test_awaitable_blocking_resolves_from_running_event_loop() -> None:
    async def scenario() -> None:
        pending = run_sync_or_awaitable(_value())
        assert inspect_is_awaitable(pending)
        assert await pending == 7
        assert run_awaitable_blocking(_value()) == 7

    asyncio.run(scenario())


def inspect_is_awaitable(value: object) -> bool:
    return hasattr(value, "__await__")
