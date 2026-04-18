from __future__ import annotations

"""Helpers for append-only runtime artifacts.

Semantic note:
- replacement artifacts should preserve lookup continuity for the old id
- therefore `replace_existing=True` redirects prior active matches to the new id
  instead of leaving them as terminal tombstones
- the integer passed to `build_node(...)` is `created_at_ms`, which callers may
  use both as creation metadata and as a stable monotonic component for ids
"""

from dataclasses import dataclass
from typing import Any, Callable
import time

from kogwistar.engine_core.engine import scoped_namespace


@dataclass(frozen=True, slots=True)
class VersionedArtifactWriteResult:
    artifact_id: str
    created_at_ms: int
    replaced_ids: tuple[str, ...]


def write_versioned_artifact(
    engine: Any,
    *,
    namespace: str,
    match_where: dict[str, Any],
    build_node: Callable[[list[Any], int], Any],
    replace_existing: bool = True,
) -> VersionedArtifactWriteResult:
    """Replace matching active nodes with one fresh append-only artifact.

    The caller owns the node semantics; this helper only provides the generic
    replacement pattern:
    1. read current active matches
    2. build and write the fresh node
    3. optionally redirect the old ids to the new id

    Notes:
    - `replace_existing=True` means "supersede current active matches"
    - the callback receives `created_at_ms`, not a semantic version number
    - use `resolve_mode="redirect"` if you want old ids to resolve to the new node
    """
    with scoped_namespace(engine, namespace):
        existing = list(engine.read.get_nodes(where=match_where))
        replaced_ids = tuple(str(node.id) for node in existing)

        created_at_ms = int(time.time() * 1000)
        new_node = build_node(existing, created_at_ms)
        engine.write.add_node(new_node)
        if replace_existing:
            for old_node in existing:
                try:
                    engine.lifecycle.redirect_node(
                        str(old_node.id),
                        str(getattr(new_node, "id")),
                    )
                except Exception:
                    # The helper is best-effort on lifecycle replacement; callers
                    # still get a fresh versioned node even if an old row cannot
                    # be redirected in the current backend.
                    pass
        return VersionedArtifactWriteResult(
            artifact_id=str(getattr(new_node, "id")),
            created_at_ms=created_at_ms,
            replaced_ids=replaced_ids,
        )
