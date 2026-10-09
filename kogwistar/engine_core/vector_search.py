"""Backend-neutral scored vector search results."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

TNode = TypeVar("TNode")


def similarity_from_distance(
    distance: float | None,
    *,
    metric: str,
    distance_kind: str = "distance",
) -> float | None:
    """Convert a backend distance into a higher-is-better similarity.

    Backends all return lower-is-better values, but inner-product backends do
    not all encode them identically.  The adapter declares that encoding and
    this function keeps filtering/ranking policy in one place.
    """
    if distance is None:
        return None
    value = float(distance)
    if not isfinite(value):
        return None
    normalized_metric = str(metric).strip().lower()
    if normalized_metric == "cosine":
        return 1.0 - value
    if normalized_metric == "l2":
        return 1.0 / (1.0 + max(value, 0.0))
    if normalized_metric == "ip":
        if distance_kind == "negative_inner_product":
            return -value
        if distance_kind == "distance":
            return 1.0 - value
        raise ValueError(f"unsupported inner-product distance kind: {distance_kind!r}")
    raise ValueError(f"unsupported vector metric: {metric!r}")


@dataclass(frozen=True, slots=True)
class VectorSearchHit(Generic[TNode]):
    """A graph node with backend score semantics made explicit."""

    node: TNode
    raw_distance: float | None
    similarity: float | None
    metric: str
    distance_kind: str


__all__ = ["VectorSearchHit", "similarity_from_distance"]
