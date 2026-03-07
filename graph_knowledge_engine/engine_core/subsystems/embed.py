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

    def iterative_defensive_emb(self, emb_text0):
        return self._call("iterative_defensive_emb", emb_text0)

    def iterative_defensive_emb_internal(self, emb_text0):
        return self._call("iterative_defensive_emb_internal", emb_text0)
