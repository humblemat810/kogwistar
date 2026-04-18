from __future__ import annotations

"""Compatibility re-exports for maintenance grouping helpers."""

from kogwistar.maintenance.grouped_artifacts import (
    GroupedArtifactWriteResult,
    write_grouped_versioned_artifacts,
)

__all__ = [
    "GroupedArtifactWriteResult",
    "write_grouped_versioned_artifacts",
]
