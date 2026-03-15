"""
Embedding provider factory.

Supports multiple backends via the EMBEDDING_PROVIDER env var:
  - ollama   (default) — local Ollama instance
  - openai   — OpenAI API (text-embedding-3-small)
  - azure    — Azure OpenAI
  - google   — Google Vertex/Generative AI

Each provider implements the ChromaDB EmbeddingFunction protocol:
  __call__(documents: Sequence[str]) -> list[list[float]]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence, cast

import numpy as np

from ..utils.embedding_vectors import normalize_embedding_vector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB type imports (graceful fallback when chromadb absent)
# ---------------------------------------------------------------------------
try:
    from chromadb.api.types import EmbeddingFunction, Embeddings  # type: ignore
except Exception:

    class EmbeddingFunction:  # type: ignore[no-redef]
        @staticmethod
        def name() -> str:
            return "default"

    Embeddings = list[list[float]]  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Shared normalisation helper
# ---------------------------------------------------------------------------


def _l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
    """L2-normalise each vector; skip zero-vectors."""
    normed: list[list[float]] = []
    for vec in vectors:
        r = np.asarray(vec, dtype=float)
        norm_val = float(np.linalg.norm(r))
        normed.append(r.tolist() if norm_val == 0.0 else (r / norm_val).tolist())
    return normed


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Ollama-backed embeddings (default). Reads OLLAMA_HOST for endpoint."""

    @staticmethod
    def name() -> str:
        return "ollama"

    def __init__(self, model_name: str = "all-minilm:l6-v2"):
        self.model_name = model_name

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        import ollama  # deferred so absence doesn't crash other providers

        raw: list[list[float]] = []
        for p in documents_or_texts:
            out = ollama.embeddings(model=self.model_name, prompt=p)
            vec_any = cast(Any, out).embedding
            raw.append(normalize_embedding_vector(vec_any, allow_none=False) or [])
        return _l2_normalize(raw)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """OpenAI API embeddings."""

    @staticmethod
    def name() -> str:
        return "openai"

    def __init__(
        self, model_name: str = "text-embedding-3-small", api_key: str | None = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY env var required for openai embedding provider"
            )

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(
            input=list(documents_or_texts), model=self.model_name
        )
        raw = [item.embedding for item in resp.data]
        return _l2_normalize(raw)


class AzureEmbeddingFunction(EmbeddingFunction):
    """Azure OpenAI embeddings."""

    @staticmethod
    def name() -> str:
        return "azure"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-02-01",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT required for azure embedding provider"
            )

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        import openai

        client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
        resp = client.embeddings.create(
            input=list(documents_or_texts), model=self.model_name
        )
        raw = [item.embedding for item in resp.data]
        return _l2_normalize(raw)


class GoogleEmbeddingFunction(EmbeddingFunction):
    """Google Generative AI embeddings."""

    @staticmethod
    def name() -> str:
        return "google"

    def __init__(
        self, model_name: str = "text-embedding-004", api_key: str | None = None
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY env var required for google embedding provider"
            )

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        raw: list[list[float]] = []
        # Batch if possible, else one-by-one
        for text in documents_or_texts:
            result = genai.embed_content(
                model=f"models/{self.model_name}", content=text
            )
            raw.append(result["embedding"])
        return _l2_normalize(raw)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": OllamaEmbeddingFunction,
    "openai": OpenAIEmbeddingFunction,
    "azure": AzureEmbeddingFunction,
    "google": GoogleEmbeddingFunction,
}


def get_embedding_function(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> EmbeddingFunction:
    """
    Create an EmbeddingFunction based on env vars or explicit args.

    Env vars:
      EMBEDDING_PROVIDER  — one of: ollama, openai, azure, google  (default: ollama)
      EMBEDDING_MODEL     — provider-specific model name
    """
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "ollama")
    model = model or os.getenv("EMBEDDING_MODEL")

    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            f"Available: {', '.join(_PROVIDERS)}"
        )

    init_kwargs: dict[str, Any] = {**kwargs}
    if model:
        init_kwargs["model_name"] = model

    logger.info("Embedding provider: %s (model=%s)", provider, model or "(default)")
    return cls(**init_kwargs)


# Backwards compatibility alias
CustomEmbeddingFunction = OllamaEmbeddingFunction
