"""Thin delegation layer; workflow runtime remains sole executor."""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any

from .limits import AgentBudgetPolicy
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
        self._known_workflows = (
            frozenset(known_workflows) if known_workflows is not None else None
        )
        self._binding_context = {
            "known_workflows": self._known_workflows,
            "known_providers": known_providers,
            "known_model_profiles": known_model_profiles,
            "known_hooks": known_hooks,
            "caller_capabilities": caller_capabilities,
        }
        self._effective_capabilities = profile.effective_capabilities(
            caller_capabilities=caller_capabilities,
            revoked_capabilities=revoked_capabilities,
        )
        self._budget_policy = AgentBudgetPolicy(**dict(profile.budget_policy))
        self._authority_context = MappingProxyType(
            {
                "principal_id": principal_id or profile.agent_id,
                "tenant_id": tenant_id,
                "project_id": project_id,
                "security_scope": security_scope,
                "effective_capabilities": self._effective_capabilities,
                "budget_limits": MappingProxyType(self._budget_policy.limits()),
            }
        )
        self.validate_bindings()

    def _prepared_state(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        state = dict(initial_state)
        if self._effective_capabilities:
            state["effective_capabilities"] = list(self._effective_capabilities)
        if self._budget_policy.limits():
            self._budget_policy.seed(state)
        return state

    def validate_bindings(self) -> None:
        """Fail closed before dispatch when registries/caller scope are supplied."""

        self.profile.validate_bindings(**self._binding_context)

    def _select_workflow(self, workflow_id: str | None) -> str:
        selected_workflow = workflow_id or self.profile.workflow_id
        if not selected_workflow:
            raise ValueError("workflow_id is required by the agent profile or run call")
        if (
            self._known_workflows is not None
            and selected_workflow not in self._known_workflows
        ):
            raise ValueError(f"unknown workflow: {selected_workflow}")
        return selected_workflow

    def run(
        self,
        *,
        initial_state: dict[str, Any],
        conversation_id: str,
        workflow_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected_workflow = self._select_workflow(workflow_id)
        return self.workflow_runtime.run(
            workflow_id=selected_workflow,
            conversation_id=conversation_id,
            initial_state=self._prepared_state(initial_state),
            _authority_context=self._authority_context,
            **kwargs,
        )

    def resume_from_latest_checkpoint(self, **kwargs: Any) -> Any:
        """Resume with current trusted ACL/budget authority."""

        kwargs["_parent_authority_context"] = self._authority_context
        return self.workflow_runtime.resume_from_latest_checkpoint(**kwargs)


class AsyncAgentHarness(AgentHarness):
    async def run(
        self,
        *,
        initial_state: dict[str, Any],
        conversation_id: str,
        workflow_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        selected_workflow = self._select_workflow(workflow_id)
        result = self.workflow_runtime.run(
            workflow_id=selected_workflow,
            conversation_id=conversation_id,
            initial_state=self._prepared_state(initial_state),
            _authority_context=self._authority_context,
            **kwargs,
        )
        return await result if inspect.isawaitable(result) else result

    async def resume_from_latest_checkpoint(self, **kwargs: Any) -> Any:
        kwargs["_parent_authority_context"] = self._authority_context
        result = self.workflow_runtime.resume_from_latest_checkpoint(**kwargs)
        return await result if inspect.isawaitable(result) else result
