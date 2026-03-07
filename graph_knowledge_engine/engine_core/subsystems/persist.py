from __future__ import annotations

from .base import NamespaceProxy


class PersistSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def preflight_validate(self, *args, **kwargs):
        return self._e._preflight_validate(*args, **kwargs)

    def resolve_llm_ids(self, *args, **kwargs):
        return self._e._resolve_llm_ids(*args, **kwargs)

    def build_deps(self, *args, **kwargs):
        return self._e._build_deps(*args, **kwargs)

    def assert_endpoints_exist(self, *args, **kwargs):
        return self._e._assert_endpoints_exist(*args, **kwargs)

    def exists_node(self, rid: str) -> bool:
        return self._e._exists_node(rid)

    def exists_edge(self, rid: str) -> bool:
        return self._e._exists_edge(rid)

    def exists_any(self, rid: str) -> bool:
        return self._e._exists_any(rid)

    def dealias_span(self, *args, **kwargs):
        return self._e._dealias_span(*args, **kwargs)

    def select_doc_context(self, *args, **kwargs):
        return self._e._select_doc_context(*args, **kwargs)

    def persist_graph_extraction(self, *args, **kwargs):
        return self._e._impl_persist_graph_extraction(*args, **kwargs)

    def persist_document_graph_extraction(self, *args, **kwargs):
        return self._e.persist_document_graph_extraction(*args, **kwargs)
