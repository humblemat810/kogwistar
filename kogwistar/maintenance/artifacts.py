"""Helpers for append-only maintenance artifacts."""

from __future__ import annotations

import time

from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Node
from kogwistar.maintenance.contracts import ArtifactNodeBuilder
from kogwistar.maintenance.models import VersionedArtifactWriteResult


def write_versioned_artifact(
    engine: GraphKnowledgeEngine,
    *,
    namespace: str,
    match_where: dict[str, object],
    build_node: ArtifactNodeBuilder,
    replace_existing: bool = True,
) -> VersionedArtifactWriteResult:
    """Replace matching active nodes unless the semantic artifact is already active."""
    with scoped_namespace(engine, namespace):
        existing: list[Node] = list(engine.read.get_nodes(where=match_where))

        created_at_ms = int(time.time() * 1000)
        new_node: Node = build_node(existing, created_at_ms)
        new_id = str(new_node.id)
        existing_match = next(
            (node for node in existing if str(getattr(node, "id", "")) == new_id),
            None,
        )
        if existing_match is not None:
            existing_metadata = getattr(existing_match, "metadata", {}) or {}
            existing_created_at_ms = int(existing_metadata.get("created_at_ms") or created_at_ms)
            return VersionedArtifactWriteResult(
                artifact_id=new_id,
                created_at_ms=existing_created_at_ms,
                replaced_ids=(),
                wrote_new_artifact=False,
            )

        replaced_ids = tuple(str(node.id) for node in existing)
        engine.write.add_node(new_node)
        if replace_existing:
            for old_node in existing:
                try:
                    engine.lifecycle.redirect_node(
                        str(old_node.id),
                        str(new_node.id),
                    )
                except Exception:
                    # This helper is best-effort on lifecycle replacement.
                    pass
        return VersionedArtifactWriteResult(
            artifact_id=new_id,
            created_at_ms=created_at_ms,
            replaced_ids=replaced_ids,
            wrote_new_artifact=True,
        )
