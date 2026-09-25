"""Thin delegation layer; workflow runtime remains sole executor."""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any

from .profile import AgentProfile


class AgentHarness:
    def __init__(
        self,
        *,
        profile: AgentProfile,
        workflow_runtime: Any,
        known_workflows: set[str] | frozenset[str] | None = None,
        known_providers: set[str] | frozenset[str] | None = None,
        known_model_profiles: set[str] | frozenset[str] | None = None,
        known_hooks: set[str] | frozenset[str] | None = None,
        caller_capabilities: tuple[str, ...] = (),
        revoked_capabilities: tuple[str, ...] = (),
        principal_id: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        security_scope: str | None = None,
    ) -> None:
        self.profile = profile
        self.workflow_runtime = workflow_runtime
        self._binding_context = {
            "known_workflows": known_workflows,
            "known_providers": known_providers,
            "known_model_profiles": known_model_profiles,
            "known_hooks": known_hooks,
            "caller_capabilities": caller_capabilities,
        }
        self._effective_capabilities = profile.effective_capabilities(
            caller_capabilities=caller_capabilities,
            revoked_capabilities=revoked_capabilities,
        )
        self._authority_context = MappingProxyType(
            {
                "principal_id": principal_id or profile.agent_id,
                "tenant_id": tenant_id,
                "project_id": project_id,
                "security_scope": security_scope,
                "effective_capabilities": self._effective_capabilities,
            }
        )
        self.validate_bindings()

    def _prepared_state(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        state = dict(initial_state)
        if self._effective_capabilities:
            state["effective_capabilities"] = list(self._effective_capabilities)
        return state

    def validate_bindings(self) -> None:
        """Fail closed before dispatch when registries/caller scope are supplied."""

        self.profile.validate_bindings(**self._binding_context)

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
            initial_state=self._prepared_state(initial_state),
            _authority_context=self._authority_context,
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
            initial_state=self._prepared_state(initial_state),
            _authority_context=self._authority_context,
            **kwargs,
        )
        return await result if inspect.isawaitable(result) else result
