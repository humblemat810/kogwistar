from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .models import WorkflowState
from .runtime import RunResult

# Public ``RunResult.status`` deliberately says ``failure``.  Durable terminal
# projections use ``failed``; do not leak that storage spelling into runtime
# callers.
TerminalStatus = Literal["succeeded", "failure", "cancelled", "suspended"]


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

    async def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: Any | None = None,
    ) -> RunResult: ...

    def run_sync(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: Any | None = None,
    ) -> RunResult: ...
