from __future__ import annotations

from .base import NamespaceProxy


class ReadSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    # Canonical read API
    def get_nodes(self, *args, **kwargs):
        return self._e._impl_get_nodes(*args, **kwargs)

    def get_edges(self, *args, **kwargs):
        return self._e._impl_get_edges(*args, **kwargs)

    def query_nodes(self, *args, **kwargs):
        return self._e._impl_query_nodes(*args, **kwargs)

    def query_edges(self, *args, **kwargs):
        return self._e._impl_query_edges(*args, **kwargs)

    def where_update_from_resolve_mode(self, *args, **kwargs):
        return self._e._where_update_from_resolve_mode(*args, **kwargs)

    # Doc-index helpers
    def node_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        if insertion_method:
            return self._e.ids_with_insertion_method(
                kind="node",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        if hasattr(self._e, "node_docs_collection"):
            rows = self._e.backend.node_docs_get(where={"doc_id": doc_id}, include=["metadatas"])
            result = set()
            for m in (rows.get("metadatas") or []):
                if m and m.get("node_id"):
                    result.add(m.get("node_id"))
            return sorted(result)
        got = self._e.backend.node_get(where={"doc_id": doc_id})
        return got.get("ids") or []

    def edge_ids_by_doc(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        if insertion_method:
            return self._e.ids_with_insertion_method(
                kind="edge",
                insertion_method=insertion_method,
                doc_id=doc_id,
            )
        eps = self._e.backend.edge_endpoints_get(where={"doc_id": doc_id}, include=["metadatas"])
        result = set()
        for m in (eps.get("metadatas") or []):
            if m and m.get("edge_id"):
                result.add(m.get("edge_id"))
        return sorted(result)

    # Legacy names retained during migration
    def nodes_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self.node_ids_by_doc(doc_id, insertion_method=insertion_method)

    def edge_ids_by_doc_index(self, doc_id: str, insertion_method: str | None = None) -> list[str]:
        return self.edge_ids_by_doc(doc_id, insertion_method=insertion_method)

    def load_node_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        node_type = kwargs.pop("node_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_node_map accepts only ids, node_type, and include")
        if ids is None:
            return {}
        nodes = self.get_nodes(ids=list(ids), node_type=node_type, include=include or ["documents"])
        return {n.safe_get_id(): n for n in nodes}

    def load_edge_map(self, *args, **kwargs):
        ids = kwargs.pop("ids", None)
        if ids is None and args:
            ids = args[0]
            args = args[1:]
        edge_type = kwargs.pop("edge_type", None)
        include = kwargs.pop("include", None)
        if args or kwargs:
            raise TypeError("load_edge_map accepts only ids, edge_type, and include")
        if ids is None:
            return {}
        edges = self.get_edges(ids=list(ids), edge_type=edge_type, include=include or ["documents"])
        return {e.safe_get_id(): e for e in edges}
