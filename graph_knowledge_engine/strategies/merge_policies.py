# strategies/merge_policies.py
from __future__ import annotations
from ..strategies import MergePolicy
from ..models import AdjudicationVerdict
from strategies import EngineLike
class PreferExistingCanonical(MergePolicy):
    def __init__(self, engine: EngineLike):
        self.e : EngineLike = engine

    def merge(self, left, right, verdict: AdjudicationVerdict) -> str:
        # node↔node, edge↔edge, cross-kind is handled inside engine methods you already wrote
        if hasattr(left, "kind"):   # AdjudicationTarget path
            if left.kind == right.kind:
                return self.e.commit_merge_target(left, right, verdict)
            # cross-kind: allow link
            return self.e.commit_merge_target(left, right, verdict)
        # Back-compat: raw Node/Edge
        if left.__class__.__name__ == right.__class__.__name__:
            return self.e.commit_merge(left, right, verdict)
        return self.e.commit_merge_target(self.e._target_from_node(left) if left.__class__.__name__=="Node" else self.e._target_from_edge(left),
                                      self.e._target_from_node(right) if right.__class__.__name__=="Node" else self.e._target_from_edge(right),
                                      verdict)