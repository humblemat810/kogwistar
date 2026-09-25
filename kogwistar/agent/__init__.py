"""Thin agent composition contracts built on Kogwistar primitives.

This package does not contain another workflow engine.  It provides typed
profile, provider, catalog, and deterministic skill-ingestion helpers which
delegate execution to the existing workflow runtime.
"""

from .catalog import CatalogEntry, CatalogSearchResult, CatalogStore
from .harness import AgentHarness, AsyncAgentHarness
from .bindings import (
    FunctionFakeTool,
    SequenceFakeModel,
    dynamic_invocation,
    make_nested_invocation_handler,
    register_catalog_search_step,
    register_model_step,
    register_tool_step,
    static_invocation,
    validate_invocation_request,
)
from .limits import AgentBudgetPolicy, budget_hints, refresh_budget_hints
from .profile import AgentProfile
from .providers import (
    ProviderCollisionError,
    ProviderIdentity,
    ProviderOwnershipError,
    ProviderRegistry,
    ProviderRegistration,
)
from .read_tools import AgentReadTools, ReadPage, ReadScope
from .skills import (
    SkillGraphArtifact,
    SkillGraphEdge,
    SkillGraphNode,
    SkillStepKind,
    catalog_entries_from_artifact,
    parse_skill_text,
    validate_command_argv,
    validate_package_relative_path,
)
from .workflows import (
    build_goal_workflow,
    build_normal_workflow,
    build_plan_workflow,
    graph_signature,
    without_agent_mode_metadata,
)

__all__ = [
    "AgentHarness",
    "AgentProfile",
    "AsyncAgentHarness",
    "AgentBudgetPolicy",
    "CatalogEntry",
    "CatalogSearchResult",
    "CatalogStore",
    "ProviderIdentity",
    "ProviderCollisionError",
    "ProviderOwnershipError",
    "ProviderRegistry",
    "ProviderRegistration",
    "AgentReadTools",
    "FunctionFakeTool",
    "ReadPage",
    "ReadScope",
    "SequenceFakeModel",
    "SkillGraphArtifact",
    "SkillGraphEdge",
    "SkillGraphNode",
    "SkillStepKind",
    "catalog_entries_from_artifact",
    "parse_skill_text",
    "validate_command_argv",
    "validate_package_relative_path",
    "build_goal_workflow",
    "build_normal_workflow",
    "build_plan_workflow",
    "budget_hints",
    "dynamic_invocation",
    "graph_signature",
    "make_nested_invocation_handler",
    "refresh_budget_hints",
    "register_catalog_search_step",
    "register_model_step",
    "register_tool_step",
    "static_invocation",
    "validate_invocation_request",
    "without_agent_mode_metadata",
]
