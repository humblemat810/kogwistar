from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VersionedArtifactWriteResult:
    artifact_id: str
    created_at_ms: int
    replaced_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupedArtifactWriteResult:
    group_key: str
    source_node_count: int
    write_result: VersionedArtifactWriteResult


@dataclass(frozen=True, slots=True)
class MaintenanceTemplateResult:
    grouped_results: list[GroupedArtifactWriteResult]
    source_node_count: int

    @property
    def emitted_group_keys(self) -> tuple[str, ...]:
        return tuple(result.group_key for result in self.grouped_results)
