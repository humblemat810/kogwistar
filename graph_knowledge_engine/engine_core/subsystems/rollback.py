from __future__ import annotations

from .base import NamespaceProxy


class RollbackSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def rollback_document(self, *args, **kwargs):
        return self._e._impl_rollback_document(*args, **kwargs)

    def delete_edges_by_ids(self, *args, **kwargs):
        return self._e._delete_edges_by_ids(*args, **kwargs)

    def prune_node_refs_for_doc(self, *args, **kwargs):
        return self._e._prune_node_refs_for_doc(*args, **kwargs)
