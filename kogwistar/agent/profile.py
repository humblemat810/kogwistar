"""Serializable agent composition profile.

Requested capabilities are declarations, never grants.  Effective authority
must still be the caller/delegation/ACL intersection at the invocation seam.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_names(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if name and name != "*" and name not in result:
            result.append(name)
    return result


class AgentProfile(BaseModel):
    """Agent configuration without execution or authority ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    workflow_id: str | None = None
    workflow_mode: Literal["normal", "plan", "goal"] = "normal"
    model_profile: str = "default"
    requested_tool_capabilities: list[str] = Field(default_factory=list)
    context_policy: dict[str, object] = Field(default_factory=dict)
    budget_policy: dict[str, object] = Field(default_factory=dict)
    skill_providers: list[str] = Field(default_factory=list)
    plugin_ids: list[str] = Field(default_factory=list)
    hook_ids: list[str] = Field(default_factory=list)
    acl_required: bool = True

    @field_validator(
        "agent_id", "model_profile", "workflow_id", mode="before"
    )
    @classmethod
    def _clean_scalar(cls, value: object) -> object:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            raise ValueError("profile identifiers must not be empty")
        return text

    @field_validator(
        "requested_tool_capabilities", "skill_providers", "plugin_ids", "hook_ids",
        mode="before",
    )
    @classmethod
    def _clean_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("profile name lists must be strings or sequences")
        return _clean_names(list(value))

    def effective_capabilities(
        self,
        *,
        caller_capabilities: list[str] | tuple[str, ...] = (),
        revoked_capabilities: list[str] | tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return delegation intersection; never expands caller authority."""

        requested = set(self.requested_tool_capabilities)
        caller = set(_clean_names(list(caller_capabilities)))
        revoked = set(_clean_names(list(revoked_capabilities)))
        return tuple(sorted((requested & caller) - revoked))

    def stable_json(self) -> str:
        """Return deterministic profile data for provenance and cache keys."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def validate_bindings(
        self,
        *,
        known_workflows: set[str] | frozenset[str] | None = None,
        known_providers: set[str] | frozenset[str] | None = None,
        known_model_profiles: set[str] | frozenset[str] | None = None,
        known_hooks: set[str] | frozenset[str] | None = None,
        caller_capabilities: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Reject unknown selections and capability escalation before dispatch."""

        if not self.acl_required:
            raise PermissionError(
                "AgentProfile cannot disable host ACL enforcement; "
                "use an explicitly unprotected local test composition instead"
            )
        if known_workflows is not None and self.workflow_id not in known_workflows:
            raise ValueError(f"unknown workflow: {self.workflow_id}")
        if known_providers is not None:
            unknown = set(self.skill_providers) - set(known_providers)
            if unknown:
                raise ValueError(f"unknown skill providers: {sorted(unknown)}")
        if known_model_profiles is not None and self.model_profile not in set(known_model_profiles):
            raise ValueError(f"unknown model profile: {self.model_profile}")
        if known_hooks is not None:
            unknown = set(self.hook_ids) - set(known_hooks)
            if unknown:
                raise ValueError(f"unknown hooks: {sorted(unknown)}")
        requested = set(self.requested_tool_capabilities)
        caller = set(_clean_names(list(caller_capabilities)))
        if requested - caller:
            raise PermissionError(
                "profile requests capabilities absent from caller authority: "
                + ", ".join(sorted(requested - caller))
            )
