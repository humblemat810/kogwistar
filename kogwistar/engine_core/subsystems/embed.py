from __future__ import annotations

from .base import NamespaceProxy
from ..async_compat import run_sync_or_awaitable
from ...utils.embedding_vectors import normalize_embedding_vector


def _token_capability(value):
    if all(callable(getattr(value, name, None)) for name in ("count_tokens", "truncate_to_tokens")):
        limit = getattr(value, "max_input_tokens", None)
        if isinstance(limit, int) and limit > 0:
            return value
    return None


class EmbedSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def iterative_defensive_emb(self, emb_text0):
        if self._e.cached_embed:
            return self._e.cached_embed(emb_text0)
        return self.iterative_defensive_emb_internal(emb_text0)

    def iterative_defensive_emb_internal(self, emb_text0):
        token_capability = _token_capability(self._e._ef)
        profile = getattr(self._e, "embedding_profile", None)
        token_budget = getattr(profile, "crop_token_budget", None)
        if token_budget is None:
            token_budget = getattr(profile, "max_sequence_length", None)
        if token_capability is not None and token_budget is not None:
            return self._iterative_token_defensive_emb(
                str(emb_text0),
                token_capability,
                min(int(token_budget), token_capability.max_input_tokens),
            )
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
                embedding = run_sync_or_awaitable(self._e._ef([emb_text]))[0]
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
                embedding = run_sync_or_awaitable(self._e._ef([emb_text]))[0]
                if idx >= len(emb_text0):
                    break
                idx = int(idx * 1.6)
            except Exception:
                success = False
        if embedding is None:
            raise Exception(
                "cannot get embedding after most defensive embedding strategy."
            )
        return normalize_embedding_vector(embedding, allow_none=False)

    def _iterative_token_defensive_emb(self, text, capability, budget):
        """Embed the largest successful token-bounded prefix.

        Providers can reject a prompt for reasons other than length, so the
        initial request is attempted at the configured budget and then reduced
        conservatively. The character fallback above remains the compatibility
        path for providers without this explicit capability.
        """
        if not text:
            candidate = text
        else:
            candidate = capability.truncate_to_tokens(text, budget)
        try:
            embedding = run_sync_or_awaitable(self._e._ef([candidate]))[0]
            return normalize_embedding_vector(embedding, allow_none=False)
        except Exception as first_error:
            high = max(0, int(capability.count_tokens(candidate)) - 1)
            low = 1
            best = None
            while low <= high:
                middle = (low + high) // 2
                reduced = capability.truncate_to_tokens(text, middle)
                try:
                    best = run_sync_or_awaitable(self._e._ef([reduced]))[0]
                    low = middle + 1
                except Exception:
                    high = middle - 1
            if best is None:
                raise first_error
            return normalize_embedding_vector(best, allow_none=False)
