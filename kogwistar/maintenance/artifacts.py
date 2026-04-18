from __future__ import annotations

"""Helpers for append-only maintenance artifacts."""

import time
from typing import Any, Callable

from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.maintenance.models import VersionedArtifactWriteResult


def write_versioned_artifact(
    engine: Any,
    *,
    namespace: str,
    match_where: dict[str, Any],
    build_node: Callable[[list[Any], int], Any],
    replace_existing: bool = True,
) -> VersionedArtifactWriteResult:
    """Replace matching active nodes with one fresh append-only artifact."""
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
                    # This helper is best-effort on lifecycle replacement.
                    pass
        return VersionedArtifactWriteResult(
            artifact_id=str(getattr(new_node, "id")),
            created_at_ms=created_at_ms,
            replaced_ids=replaced_ids,
        )
