from __future__ import annotations

from .base import NamespaceProxy


class WriteSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Canonical write API
    def add_node(self, *args, **kwargs):
        return self._e._impl_add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        return self._e._impl_add_edge(*args, **kwargs)

    def add_document(self, *args, **kwargs):
        return self._e._impl_add_document(*args, **kwargs)

    def add_domain(self, *args, **kwargs):
        return self._e._impl_add_domain(*args, **kwargs)

    # Index/metadata helpers
    def index_node_docs(self, *args, **kwargs):
        return self._e._index_node_docs(*args, **kwargs)

    def index_node_refs(self, *args, **kwargs):
        return self._e._index_node_refs(*args, **kwargs)

    def index_edge_refs(self, *args, **kwargs):
        return self._e._index_edge_refs(*args, **kwargs)

    def fanout_endpoints_rows(self, *args, **kwargs):
        return self._e._fanout_endpoints_rows(*args, **kwargs)

    def node_doc_and_meta(self, *args, **kwargs):
        return self._e._node_doc_and_meta(*args, **kwargs)

    def edge_doc_and_meta(self, *args, **kwargs):
        return self._e._edge_doc_and_meta(*args, **kwargs)

    def strip_none(self, *args, **kwargs):
        return self._e._strip_none(*args, **kwargs)

    def json_or_none(self, *args, **kwargs):
        return self._e._json_or_none(*args, **kwargs)

    def delete_edge_ref_rows(self, *args, **kwargs):
        return self._e._delete_edge_ref_rows(*args, **kwargs)

    def delete_node_ref_rows(self, *args, **kwargs):
        return self._e._delete_node_ref_rows(*args, **kwargs)
