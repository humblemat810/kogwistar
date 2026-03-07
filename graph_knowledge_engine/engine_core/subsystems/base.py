from __future__ import annotations

from typing import Any


class NamespaceProxy:
    """Base class for namespaced subsystem APIs bound to one engine instance."""

    def __init__(self, engine: Any) -> None:
        self._e = engine
