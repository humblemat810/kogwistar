from __future__ import annotations

from .base import NamespaceProxy


class ReadSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "node_ids_by_doc": "_nodes_by_doc",
                "edge_ids_by_doc": "_edge_ids_by_doc",
                "where_update_from_resolve_mode": "_where_update_from_resolve_mode",
                "nodes_by_doc_index": "_nodes_by_doc",
                "edge_ids_by_doc_index": "_edge_ids_by_doc",
            },
        )
