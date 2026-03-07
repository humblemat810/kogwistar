from __future__ import annotations

from .base import NamespaceProxy


class WriteSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "index_node_docs": "_index_node_docs",
                "index_node_refs": "_index_node_refs",
                "index_edge_refs": "_index_edge_refs",
                "fanout_endpoints_rows": "_fanout_endpoints_rows",
                "node_doc_and_meta": "_node_doc_and_meta",
                "edge_doc_and_meta": "_edge_doc_and_meta",
                "strip_none": "_strip_none",
                "json_or_none": "_json_or_none",
                "delete_edge_ref_rows": "_delete_edge_ref_rows",
                "delete_node_ref_rows": "_delete_node_ref_rows",
            },
        )

    # Canonical write API
    def add_node(self, *args, **kwargs):
        return self._call("add_node", *args, **kwargs)

    def add_edge(self, *args, **kwargs):
        return self._call("add_edge", *args, **kwargs)

    def add_document(self, *args, **kwargs):
        return self._call("add_document", *args, **kwargs)

    def add_domain(self, *args, **kwargs):
        return self._call("add_domain", *args, **kwargs)

    # Index/metadata helpers
    def index_node_docs(self, *args, **kwargs):
        return self._call("index_node_docs", *args, **kwargs)

    def index_node_refs(self, *args, **kwargs):
        return self._call("index_node_refs", *args, **kwargs)

    def index_edge_refs(self, *args, **kwargs):
        return self._call("index_edge_refs", *args, **kwargs)

    def fanout_endpoints_rows(self, *args, **kwargs):
        return self._call("fanout_endpoints_rows", *args, **kwargs)

    def node_doc_and_meta(self, *args, **kwargs):
        return self._call("node_doc_and_meta", *args, **kwargs)

    def edge_doc_and_meta(self, *args, **kwargs):
        return self._call("edge_doc_and_meta", *args, **kwargs)

    def strip_none(self, *args, **kwargs):
        return self._call("strip_none", *args, **kwargs)

    def json_or_none(self, *args, **kwargs):
        return self._call("json_or_none", *args, **kwargs)

    def delete_edge_ref_rows(self, *args, **kwargs):
        return self._call("delete_edge_ref_rows", *args, **kwargs)

    def delete_node_ref_rows(self, *args, **kwargs):
        return self._call("delete_node_ref_rows", *args, **kwargs)
