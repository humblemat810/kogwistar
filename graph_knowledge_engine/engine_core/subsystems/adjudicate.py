from __future__ import annotations

from .base import NamespaceProxy


class AdjudicateSubsystem(NamespaceProxy):
    def __init__(self, engine) -> None:
        super().__init__(engine)

    def target_from_node(self, *args, **kwargs):
        return self._e._target_from_node(*args, **kwargs)

    def target_from_edge(self, *args, **kwargs):
        return self._e._target_from_edge(*args, **kwargs)

    def fetch_target(self, *args, **kwargs):
        return self._e._fetch_target(*args, **kwargs)

    def classify_endpoint_id(self, *args, **kwargs):
        return self._e._classify_endpoint_id(*args, **kwargs)

    def split_endpoints(self, *args, **kwargs):
        return self._e._split_endpoints(*args, **kwargs)

    def rebalance_same_as_edge(self, *args, **kwargs):
        return self._e._rebalance_same_as_edge(*args, **kwargs)

    def choose_anchor(self, *args, **kwargs):
        return self._e._choose_anchor(*args, **kwargs)
