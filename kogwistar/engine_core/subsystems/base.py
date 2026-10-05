from __future__ import annotations

from typing import Generic, TypeVar

T_Engine = TypeVar("T_Engine")


class NamespaceProxy(Generic[T_Engine]):
    """Base class for namespaced subsystem APIs bound to one engine instance."""

    _e: T_Engine

    def __init__(self, engine: T_Engine) -> None:
        self._e = engine
