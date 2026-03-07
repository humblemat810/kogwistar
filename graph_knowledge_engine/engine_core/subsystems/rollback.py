from __future__ import annotations

from .base import NamespaceProxy


class RollbackSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "delete_edges_by_ids": "_delete_edges_by_ids",
                "prune_node_refs_for_doc": "_prune_node_refs_for_doc",
            },
        )
