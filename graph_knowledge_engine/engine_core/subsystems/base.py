from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import GraphKnowledgeEngine


class NamespaceProxy:
    """Base class for namespaced subsystem APIs bound to one engine instance."""

    def __init__(self, engine: GraphKnowledgeEngine) -> None:
        self._e = engine
