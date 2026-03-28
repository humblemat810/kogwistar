from __future__ import annotations

import asyncio
from typing import Any


class AsyncAttrProxy:
    """Async wrapper around a sync object.

    The wrapped object keeps its original method names. Any callable attribute is
    exposed as an async function that runs the sync implementation in a worker
    thread. Non-callable attributes are forwarded unchanged.
    """

    def __init__(self, target: Any):
        self._target = target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if callable(attr):

            async def _async_call(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            _async_call.__name__ = name
            return _async_call
        return attr


class AsyncEngineFacade:
    """Async view over a sync GraphKnowledgeEngine instance.

    This keeps the public engine method names unchanged while making the server
    path awaitable. The underlying engine still executes the same code paths, but
    the async wrapper prevents FastAPI request handlers from blocking the event
    loop on direct sync calls.
    """

    def __init__(self, engine: Any):
        self._engine = engine
        self.backend = AsyncAttrProxy(engine.backend)
        self.write = AsyncAttrProxy(engine.write)
        self.read = AsyncAttrProxy(engine.read)
        self.persist = AsyncAttrProxy(engine.persist)
        self.extract = AsyncAttrProxy(engine.extract)
        self.adjudicate = AsyncAttrProxy(engine.adjudicate)
        self.embed = AsyncAttrProxy(engine.embed)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._engine, name)
        if callable(attr):

            async def _async_call(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(attr, *args, **kwargs)

            _async_call.__name__ = name
            return _async_call
        return attr


__all__ = ["AsyncAttrProxy", "AsyncEngineFacade"]
