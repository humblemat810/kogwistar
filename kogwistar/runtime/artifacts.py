from __future__ import annotations

"""Compatibility re-exports for maintenance artifact helpers."""

from kogwistar.maintenance.artifacts import (
    VersionedArtifactWriteResult,
    write_versioned_artifact,
)

__all__ = [
    "VersionedArtifactWriteResult",
    "write_versioned_artifact",
]
