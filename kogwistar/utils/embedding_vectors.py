from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import cast

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]


def _is_numeric_scalar(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if np is not None and isinstance(value, (np.integer, np.floating)):  # type: ignore[attr-defined]
        return True
    return False


def _as_float(value: object) -> float:
    if not _is_numeric_scalar(value):
        raise TypeError("embedding value must be numeric")
    return float(cast(int | float, value))


def normalize_embedding_vector(
    raw: object, *, allow_none: bool = True
) -> list[float] | None:
    """Normalize one embedding vector to plain `list[float]`.

    Accepts Python sequences, numpy arrays, pgvector-ish values that expose
    `.tolist()`, and other iterable numeric containers.
    """
    if raw is None:
        if allow_none:
            return None
        raise ValueError("embedding vector is None")

    to_list = getattr(raw, "tolist", None)
    if callable(to_list):
        raw = to_list()

    if isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"embedding vector must be numeric, got {type(raw)!r}")

    if not isinstance(raw, Sequence):
        raw = list(cast(Iterable[object], raw))

    values = list(raw)
    if not values:
        return []
    if any(not _is_numeric_scalar(v) for v in values):
        raise TypeError("embedding vector must be a flat numeric sequence")
    return [_as_float(v) for v in values]


def normalize_embedding_rows(
    raw: object, *, allow_empty: bool = True, allow_none_rows: bool = False
) -> list[list[float] | None]:
    """Normalize one-or-many embeddings to `list[list[float] | None]`.

    This accepts:
    - a single vector: `[0.1, 0.2]`
    - multiple vectors: `[[0.1, 0.2], [0.3, 0.4]]`
    - numpy arrays with rank 1 or 2
    - typed backend/provider containers exposing `.tolist()`
    """
    if raw is None:
        if allow_empty:
            return []
        raise ValueError("embedding rows are None")

    to_list = getattr(raw, "tolist", None)
    if callable(to_list):
        raw = to_list()

    if isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"embedding rows must be numeric, got {type(raw)!r}")

    if not isinstance(raw, Sequence):
        raw = list(cast(Iterable[object], raw))

    rows = list(raw)
    if not rows:
        if allow_empty:
            return []
        raise ValueError("embedding rows are empty")

    first = rows[0]
    if _is_numeric_scalar(first):
        return [normalize_embedding_vector(rows, allow_none=False)]

    normalized: list[list[float] | None] = []
    for row in rows:
        normalized.append(
            normalize_embedding_vector(row, allow_none=allow_none_rows)
        )
    return normalized
