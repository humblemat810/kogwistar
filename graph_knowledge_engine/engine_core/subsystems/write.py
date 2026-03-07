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
