from __future__ import annotations

from .base import NamespaceProxy


class IngestSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "ingest_text_with_llm": "_ingest_text_with_llm",
                "extract_graph_with_llm_internal": "_extract_graph_with_llm",
            },
        )
