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
                "load_node_map": "_load_node_map",
                "load_edge_map": "_load_edge_map",
            },
        )

    # Canonical read API
    def get_nodes(self, *args, **kwargs):
        return self._call("get_nodes", *args, **kwargs)

    def get_edges(self, *args, **kwargs):
        return self._call("get_edges", *args, **kwargs)

    def query_nodes(self, *args, **kwargs):
        return self._call("query_nodes", *args, **kwargs)

    def query_edges(self, *args, **kwargs):
        return self._call("query_edges", *args, **kwargs)

    def where_update_from_resolve_mode(self, *args, **kwargs):
        return self._call("where_update_from_resolve_mode", *args, **kwargs)

    # Doc-index helpers
    def node_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._call("node_ids_by_doc", doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._call("edge_ids_by_doc", doc_id, insertion_method=insertion_method)

    # Legacy names retained during migration
    def nodes_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._call("nodes_by_doc_index", doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self._call("edge_ids_by_doc_index", doc_id, insertion_method=insertion_method)

    def load_node_map(self, *args, **kwargs):
        return self._call("load_node_map", *args, **kwargs)

    def load_edge_map(self, *args, **kwargs):
        return self._call("load_edge_map", *args, **kwargs)
