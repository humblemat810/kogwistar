from __future__ import annotations

from dataclasses import dataclass

from kogwistar.maintenance.artifacts import VersionedArtifactWriteResult


@dataclass(frozen=True, slots=True)
class ExecutionWisdomTemplateResult:
    step_op: str
    failure_count: int
    run_ids: tuple[str, ...]
    write_result: VersionedArtifactWriteResult
