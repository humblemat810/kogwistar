from __future__ import annotations

"""Compatibility re-exports for maintenance grouping helpers."""

from kogwistar.maintenance.grouped_artifacts import (
    write_grouped_versioned_artifacts,
)
from kogwistar.maintenance.models import GroupedArtifactWriteResult

__all__ = [
    "GroupedArtifactWriteResult",
    "write_grouped_versioned_artifacts",
]
