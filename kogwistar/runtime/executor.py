from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .models import WorkflowState
from .runtime import RunResult

TerminalStatus = Literal["succeeded", "failed", "cancelled", "suspended"]


@dataclass(frozen=True)
class RunRequest:
    workflow_id: str
    conversation_id: str
    turn_node_id: str
    initial_state: WorkflowState
    run_id: str | None = None
    cache_dir: str | None = None


class WorkflowExecutor(Protocol):
    """Runtime-neutral executor contract for sync/async workflow runtimes."""

    async def run(self, **kwargs) -> RunResult: ...
    def run_sync(self, **kwargs) -> RunResult: ...
