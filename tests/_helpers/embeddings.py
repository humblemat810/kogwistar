from __future__ import annotations

"""Shared deterministic embedding helpers for tests.

Use these when a test needs a predictable vector space without depending on
external embedding services.

Supported modes:
- ``constant``: returns a dense constant vector
- ``lexical_hash``: token-hash embedding similar to ``scripts/tutorial_ladder.py``
- ``provider`` / ``real``: return ``None`` so the engine uses its configured
  default embedding provider from the environment

Example:

```python
@pytest.mark.parametrize("embedding_kind", ["lexical_hash"], indirect=True)
def test_with_lexical_embeddings(conversation_engine):
    ...
```
"""

import hashlib
import math
import re
from typing import Sequence

try:
    from chromadb.api.types import Embeddings
    from chromadb.utils.embedding_functions import EmbeddingFunction
except ImportError:  # pragma: no cover - optional in lightweight test environments

    class EmbeddingFunction:  # type: ignore
        @staticmethod
        def name() -> str:
            return "default"

    Embeddings = list[list[float]]  # type: ignore


class ConstantEmbeddingFunction(EmbeddingFunction):
    """Simple dense baseline embedding function for tests."""

    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]


class LexicalHashEmbeddingFunction(EmbeddingFunction):
    """Deterministic lexical hash embedder used by tutorial-style tests."""

    def __init__(self, dim: int = 96) -> None:
        self._dim = dim

    @staticmethod
    def name() -> str:
        return "tutorial-lexical-hash-v1"

    def __call__(self, input: Sequence[str]) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            v = [0.0] * self._dim
            tokens = re.findall(r"[a-z0-9_]+", str(text or "").lower())
            for tok in tokens:
                idx = (
                    int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
                    % self._dim
                )
                v[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def build_test_embedding_function(kind: str, *, dim: int) -> EmbeddingFunction | None:
    """Return a test embedding function by name.

    ``provider`` and ``real`` mean "use the engine default" and therefore
    return ``None``.
    """

    normalized = str(kind or "constant").strip().lower()
    if normalized in {"constant", "default", "fake"}:
        return ConstantEmbeddingFunction(dim=dim)
    if normalized in {"lexical_hash", "lexical", "hash", "tutorial"}:
        return LexicalHashEmbeddingFunction(dim=dim)
    if normalized in {"provider", "real", "default_provider"}:
        return None
    raise ValueError(
        f"Unsupported test embedding kind: {kind!r}. "
        "Use 'constant', 'lexical_hash', or 'provider'."
    )
