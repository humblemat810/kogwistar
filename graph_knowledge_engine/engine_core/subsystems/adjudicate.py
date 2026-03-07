from __future__ import annotations

from .base import NamespaceProxy


class AdjudicateSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(
            engine,
            aliases={
                "target_from_node": "_target_from_node",
                "target_from_edge": "_target_from_edge",
                "fetch_target": "_fetch_target",
                "classify_endpoint_id": "_classify_endpoint_id",
                "split_endpoints": "_split_endpoints",
                "rebalance_same_as_edge": "_rebalance_same_as_edge",
                "choose_anchor": "_choose_anchor",
            },
        )
