from __future__ import annotations

from .base import NamespaceProxy


class EmbedSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def iterative_defensive_emb(self, emb_text0):
        if self._e.cached_embed:
            return self._e.cached_embed(emb_text0)
        return self.iterative_defensive_emb_internal(emb_text0)

    def iterative_defensive_emb_internal(self, emb_text0):
        success = False
        idx = self._e.embedding_length_limit
        embedding = None
        cnt = 0
        while not success:
            cnt += 1
            if cnt >= 10:
                break
            emb_text = emb_text0[:idx] + ("..." if idx < len(emb_text0) - 1 else "")
            try:
                embedding = self._e._ef([emb_text])[0]
                success = True
                break
            except Exception:
                idx //= 2
        while success:
            cnt += 1
            if cnt >= 13:
                break
            emb_text = emb_text0[:idx] + ("..." if idx < len(emb_text0) - 1 else "")
            try:
                embedding = self._e._ef([emb_text])[0]
                if idx >= len(emb_text0):
                    break
                idx = int(idx * 1.6)
            except Exception:
                success = False
        if embedding is None:
            raise Exception(
                "cannot get embedding after most defensive embedding strategy."
            )
        return embedding
