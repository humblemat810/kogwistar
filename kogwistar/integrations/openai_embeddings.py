from __future__ import annotations

import os
from typing import Callable, Optional


def build_azure_embedding_fn_from_env() -> Optional[
    Callable[[str], Optional[list[float]]]
]:
    """Build an optional Azure OpenAI embedding callable from environment vars.

    Returns None when dependency/config is unavailable, mirroring the engine's
    best-effort behavior.
    """
    try:
        from langchain_openai import AzureOpenAIEmbeddings
    except Exception:
        return None

    emb_deployment = os.getenv("OPENAI_EMBED_DEPLOYMENT")
    emb_endpoint = os.getenv("OPENAI_EMBED_ENDPOINT")
    emb_api_key = os.getenv("OPENAI_API_KEY_GPT4_1") or os.getenv("OPENAI_API_KEY")
    emb_api_ver = os.getenv("OPENAI_EMBED_API_VERSION", "2024-08-01-preview")
    if not (emb_deployment and emb_endpoint and emb_api_key):
        return None

    emb = AzureOpenAIEmbeddings(
        azure_deployment=emb_deployment,
        openai_api_key=emb_api_key,  # type: ignore[arg-type]
        azure_endpoint=emb_endpoint,
        openai_api_version=emb_api_ver,  # type: ignore[arg-type]
    )

    def _embed_fn(text: str) -> Optional[list[float]]:
        try:
            return emb.embed_query(text)
        except Exception:
            return None

    return _embed_fn
