from __future__ import annotations

from ..models import Edge, Node
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
        g = self._e.backend.node_get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def exists_edge(self, rid: str) -> bool:
        g = self._e.backend.edge_get(ids=[rid])
        return (g.get("ids") or [None])[0] == rid

    def exists_any(self, rid: str) -> bool:
        return self.exists_node(rid) or self.exists_edge(rid)

    def dealias_span(self, *args, **kwargs):
        return self._e._dealias_span(*args, **kwargs)

    def select_doc_context(self, doc_id: str, max_nodes: int = 200, max_edges: int = 400):
        nodes = self._e.backend.node_get(where={"doc_id": doc_id}, include=["documents"])
        edges = self._e.backend.edge_get(where={"doc_id": doc_id}, include=["documents"])

        node_items = []
        for i, (nid, ndoc) in enumerate(
            zip(nodes.get("ids", []) or [], nodes.get("documents", []) or [])
        ):
            if i >= max_nodes:
                break
            n = Node.model_validate_json(ndoc)
            node_items.append({"id": nid, "label": n.label, "type": n.type, "summary": n.summary})

        edge_items = []
        for i, (eid, edoc) in enumerate(
            zip(edges.get("ids", []) or [], edges.get("documents", []) or [])
        ):
            if i >= max_edges:
                break
            e = Edge.model_validate_json(edoc)
            edge_items.append(
                {
                    "id": eid,
                    "relation": e.relation,
                    "source_ids": e.source_ids or [],
                    "target_ids": e.target_ids or [],
                }
            )

        return node_items, edge_items

    def persist_graph_extraction(self, *args, **kwargs):
        return self._e._impl_persist_graph_extraction(*args, **kwargs)

    def persist_document_graph_extraction(self, *args, **kwargs):
        return self._e.persist_document_graph_extraction(*args, **kwargs)
