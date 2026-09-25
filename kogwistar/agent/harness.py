"""Thin delegation layer; workflow runtime remains sole executor."""

from __future__ import annotations

import inspect
from typing import Any

from .profile import AgentProfile


class AgentHarness:
    def __init__(self, *, profile: AgentProfile, workflow_runtime: Any) -> None:
        self.profile = profile
        self.workflow_runtime = workflow_runtime

    def run(
        self,
        *,
        initial_state: dict[str, Any],
        conversation_id: str,
        workflow_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected_workflow = workflow_id or self.profile.workflow_id
        if not selected_workflow:
            raise ValueError("workflow_id is required by the agent profile or run call")
        return self.workflow_runtime.run(
            workflow_id=selected_workflow,
            conversation_id=conversation_id,
            initial_state=initial_state,
            **kwargs,
        )


class AsyncAgentHarness(AgentHarness):
    async def run(
        self,
        *,
        initial_state: dict[str, Any],
        conversation_id: str,
        workflow_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected_workflow = workflow_id or self.profile.workflow_id
        if not selected_workflow:
            raise ValueError("workflow_id is required by the agent profile or run call")
        result = self.workflow_runtime.run(
            workflow_id=selected_workflow,
            conversation_id=conversation_id,
            initial_state=initial_state,
            **kwargs,
        )
        return await result if inspect.isawaitable(result) else result
