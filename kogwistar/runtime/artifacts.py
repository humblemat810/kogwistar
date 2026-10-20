from __future__ import annotations

"""Compatibility re-exports for maintenance artifact helpers."""

from kogwistar.maintenance.artifacts import (
    write_versioned_artifact,
)
from kogwistar.maintenance.models import VersionedArtifactWriteResult

__all__ = [
    "VersionedArtifactWriteResult",
    "write_versioned_artifact",
]
