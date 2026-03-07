from __future__ import annotations

from .base import NamespaceProxy


class EmbedSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "iterative_defensive_emb_internal": "_iterative_defensive_emb",
            },
        )
