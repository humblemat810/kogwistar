from __future__ import annotations

from .base import NamespaceProxy


class IngestSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def ingest_document_with_llm(self, *args, **kwargs):
        return self._e._impl_ingest_document_with_llm(*args, **kwargs)

    def ingest_text_with_llm(self, *args, **kwargs):
        return self._e._ingest_text_with_llm(*args, **kwargs)

    def extract_graph_with_llm_internal(self, *args, **kwargs):
        return self._e._extract_graph_with_llm(*args, **kwargs)

    def add_page(self, *args, **kwargs):
        return self._e.add_page(*args, **kwargs)
