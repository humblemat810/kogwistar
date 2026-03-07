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

    def target_from_node(self, *args, **kwargs):
        return self._call("target_from_node", *args, **kwargs)

    def target_from_edge(self, *args, **kwargs):
        return self._call("target_from_edge", *args, **kwargs)

    def fetch_target(self, *args, **kwargs):
        return self._call("fetch_target", *args, **kwargs)

    def classify_endpoint_id(self, *args, **kwargs):
        return self._call("classify_endpoint_id", *args, **kwargs)

    def split_endpoints(self, *args, **kwargs):
        return self._call("split_endpoints", *args, **kwargs)

    def rebalance_same_as_edge(self, *args, **kwargs):
        return self._call("rebalance_same_as_edge", *args, **kwargs)

    def choose_anchor(self, *args, **kwargs):
        return self._call("choose_anchor", *args, **kwargs)
