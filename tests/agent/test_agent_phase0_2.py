"""Phase 0-2 agent contracts; all payloads deterministic and offline."""

from __future__ import annotations

import pytest

from kogwistar.agent import (
    AgentHarness,
    AgentProfile,
    AsyncAgentHarness,
    CatalogEntry,
    CatalogStore,
    ProviderRegistry,
    SkillGraphArtifact,
    catalog_entries_from_artifact,
    parse_skill_text,
    validate_command_argv,
    validate_package_relative_path,
)
from kogwistar.agent.providers import ProviderCollisionError, ProviderOwnershipError


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.regression]


def _entry(
    logical_id: str,
    *,
    provider_id: str = "project",
    revision: int = 1,
    fingerprint: str = "fp-1",
    semantic_ready: bool = False,
    scope: str = "project-a",
    metadata: dict[str, object] | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        logical_id=logical_id,
        provider_id=provider_id,
        provider_local_id=logical_id.rsplit(":", 1)[-1],
        provider_version="v1",
        kind="skill",
        name=logical_id.rsplit(":", 1)[-1],
        summary="deploy service safely",
        source_fingerprint=fingerprint,
        revision=revision,
        group_ids=["group:deploy"],
        scope=scope,
        semantic_ready=semantic_ready,
        metadata=metadata or {},
    )


def test_profile_intersects_requested_capabilities_without_escalation() -> None:
    profile = AgentProfile(
        agent_id="agent-a",
        requested_tool_capabilities=["graph.read", "process.execute", "graph.read"],
    )

    assert profile.effective_capabilities(
        caller_capabilities=["graph.read", "network.write"],
        revoked_capabilities=["network.write"],
    ) == ("graph.read",)


def test_provider_registry_rejects_collision_and_disposes_owned_provider() -> None:
    class Provider:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    registry = ProviderRegistry()
    provider = Provider()
    registration = registry.register(provider_id="project", provider=provider)
    with pytest.raises(ProviderCollisionError):
        registry.register(provider_id="project", provider=Provider())

    registry.dispose(registration)
    assert provider.closed is True
    with pytest.raises(KeyError):
        registry.get("project")
    with pytest.raises(ProviderOwnershipError):
        registry.dispose(registration)


def test_catalog_applies_acl_scope_group_and_lexical_fallback() -> None:
    allowed = {"alice:deploy"}
    catalog = CatalogStore(
        acl_checker=lambda entry, principal: f"{principal}:{entry.name}" in allowed
    )
    catalog.upsert(_entry("skill:deploy", metadata={"aliases": ["release"]}))
    catalog.upsert(_entry("skill:private", scope="private"))

    exact = catalog.search("release", principal="alice", scope="project-a")
    assert [(result.entry.logical_id, result.match) for result in exact] == [
        ("skill:deploy", "exact")
    ]
    assert catalog.search("private", principal="alice", scope="project-a") == ()
    assert [entry.logical_id for entry in catalog.browse("group:deploy", principal="alice")] == [
        "skill:deploy"
    ]


def test_catalog_semantic_mode_requires_ready_projection_and_revision_is_monotonic() -> None:
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(_entry("skill:one", semantic_ready=False))
    assert catalog.search("one", mode="semantic") == ()
    catalog.upsert(_entry("skill:one", revision=2, fingerprint="fp-2", semantic_ready=True))
    assert catalog.search("one", mode="semantic")[0].entry.revision == 2
    with pytest.raises(ValueError, match="stale"):
        catalog.upsert(_entry("skill:one", revision=1, fingerprint="fp-1"))
    with pytest.raises(ValueError, match="collision"):
        catalog.upsert(_entry("skill:one", revision=2, fingerprint="other"))


def test_provider_unload_removes_only_current_provider_projection() -> None:
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(_entry("project:one", provider_id="project"))
    catalog.upsert(_entry("wiki:one", provider_id="wiki"))
    assert catalog.remove_provider("project") == ("project:one",)
    assert catalog.get("project:one") is None
    assert catalog.get("wiki:one") is not None
    assert catalog.history("project:one")[0].provider_id == "project"


def test_skill_ingestion_is_deterministic_bounded_and_json_compatible() -> None:
    source = (
        "---\n"
        "name: Deploy Service\n"
        "description: Deploy with review\n"
        "---\n"
        "capability: process.execute\n"
        "mcp: project.deploy\n"
        "1. Check the current revision.\n"
        "```bash\n"
        "kubectl get pods\n"
        "```\n"
    )
    first = parse_skill_text(source, provider_id="project", provider_local_id="deploy")
    second = parse_skill_text(source, provider_id="project", provider_local_id="deploy")

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.artifact_fingerprint() == second.artifact_fingerprint()
    assert any(node.kind == "command_template" for node in first.nodes)
    assert any(node.kind == "mcp" for node in first.nodes)
    assert all(not node.invocable for node in first.nodes)
    assert isinstance(first.model_dump(mode="json")["edges"], list)
    projected = catalog_entries_from_artifact(first)
    assert projected[0].logical_id == "project:deploy"
    assert all("project:deploy" in entry.group_ids for entry in projected)


def test_skill_ingestion_rejects_unsafe_commands_and_paths_without_execution() -> None:
    with pytest.raises(ValueError):
        validate_command_argv("sh -c 'rm -rf /'")
    with pytest.raises(ValueError):
        validate_package_relative_path("../secret.sh")
    with pytest.raises(ValueError):
        validate_package_relative_path("C:/secret.sh")
    assert validate_command_argv("python script.py --dry-run") == [
        "python",
        "script.py",
        "--dry-run",
    ]


def test_unbound_effectful_skill_node_cannot_be_marked_invocable() -> None:
    artifact = parse_skill_text("- run deployment", provider_id="p", provider_local_id="s")
    payload = artifact.model_dump(mode="python")
    payload["nodes"].append(
        {
            "node_id": "unsafe",
            "kind": "command_template",
            "name": "deploy",
            "required_capabilities": ["process.execute"],
            "invocable": True,
        }
    )
    with pytest.raises(ValueError, match="validated"):
        SkillGraphArtifact.model_validate(payload)


def test_agent_harness_delegates_to_existing_runtime() -> None:
    class FakeRuntime:
        def run(self, **kwargs: object) -> dict[str, object]:
            return kwargs

    result = AgentHarness(
        profile=AgentProfile(agent_id="a", workflow_id="wf"),
        workflow_runtime=FakeRuntime(),
    ).run(initial_state={"x": 1}, conversation_id="c")
    assert result["workflow_id"] == "wf"
    assert result["initial_state"] == {"x": 1}


@pytest.mark.asyncio
async def test_async_agent_harness_awaits_existing_async_runtime() -> None:
    class FakeRuntime:
        async def run(self, **kwargs: object) -> dict[str, object]:
            return kwargs

    result = await AsyncAgentHarness(
        profile=AgentProfile(agent_id="a", workflow_id="wf"),
        workflow_runtime=FakeRuntime(),
    ).run(initial_state={}, conversation_id="c")
    assert result["workflow_id"] == "wf"
