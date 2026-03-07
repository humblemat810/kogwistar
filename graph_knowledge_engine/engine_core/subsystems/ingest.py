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

    def ingest_document_with_llm(self, *args, **kwargs):
        return self._call("ingest_document_with_llm", *args, **kwargs)

    def ingest_text_with_llm(self, *args, **kwargs):
        return self._call("ingest_text_with_llm", *args, **kwargs)

    def extract_graph_with_llm_internal(self, *args, **kwargs):
        return self._call("extract_graph_with_llm_internal", *args, **kwargs)

    def add_page(self, *args, **kwargs):
        return self._call("add_page", *args, **kwargs)
