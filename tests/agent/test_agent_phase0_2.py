"""Phase 0-2 agent contracts; all payloads deterministic and offline."""

from __future__ import annotations

import asyncio
import time

import pytest

from kogwistar.agent import (
    AgentHarness,
    AgentProfile,
    AsyncAgentHarness,
    CatalogEntry,
    CatalogGroup,
    CatalogStore,
    ContextSnapshot,
    HookRegistry,
    HookSpec,
    ProviderRegistry,
    SkillProjectionStore,
    SkillGraphArtifact,
    catalog_entries_from_artifact,
    parse_skill_text,
    select_skill_subgraph,
    validate_command_argv,
    validate_package_relative_path,
    validate_skill_artifact,
    build_plan_workflow,
    build_goal_workflow,
    validate_agent_design,
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


def test_harness_binding_validation_is_fail_closed_when_registries_are_supplied() -> None:
    profile = AgentProfile(
        agent_id="agent-a",
        workflow_id="wf-known",
        skill_providers=["skills"],
        hook_ids=["observe"],
        requested_tool_capabilities=["graph.read"],
    )
    AgentHarness(
        profile=profile,
        workflow_runtime=object(),
        known_workflows={"wf-known"},
        known_providers={"skills"},
        known_hooks={"observe"},
        caller_capabilities=("graph.read",),
    )
    with pytest.raises(ValueError):
        AgentHarness(
            profile=profile,
            workflow_runtime=object(),
            known_workflows={"wf-other"},
            known_providers={"skills"},
            known_hooks={"observe"},
            caller_capabilities=("graph.read",),
        )


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


def test_catalog_optional_semantic_ranker_runs_after_acl_and_readiness_filter() -> None:
    seen: list[str] = []

    def rank(query: str, entries: tuple[CatalogEntry, ...]) -> dict[str, float]:
        seen.extend(entry.logical_id for entry in entries)
        assert query == "deploy"
        return {entry.logical_id: float(index) for index, entry in enumerate(entries, 1)}

    catalog = CatalogStore(
        semantic_ranker=rank,
        acl_checker=lambda entry, principal: entry.name != "private"
        and principal == "alice",
    )
    catalog.upsert(_entry("skill:first", semantic_ready=True))
    catalog.upsert(_entry("skill:private", semantic_ready=True))
    catalog.upsert(_entry("skill:pending", semantic_ready=False))
    results = catalog.search("deploy", mode="semantic", principal="alice", limit=10)
    assert [result.entry.logical_id for result in results] == ["skill:first"]
    assert seen == ["skill:first"]
    assert results[0].match == "semantic"


def test_catalog_supports_bm25_alias_and_graph_group_projection() -> None:
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(
        _entry("skill:deploy", metadata={"aliases": ["rollout"]})
    )
    catalog.upsert_group(CatalogGroup(group_id="group:ops", name="Operations"))
    assert catalog.search("safely", mode="bm25")[0].match == "bm25"
    assert catalog.search("safely", mode="bm25")[0].score > 0
    assert catalog.search("rollout", mode="lexical")[0].match == "exact"
    assert catalog.group_tree(root_id="group:ops")[0].group_id == "group:ops"


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


@pytest.mark.asyncio
async def test_sync_and_async_agent_harnesses_preserve_same_lifecycle_contract() -> None:
    def lifecycle(**kwargs: object) -> dict[str, object]:
        return {
            "status": "completed",
            "workflow_id": kwargs["workflow_id"],
            "conversation_id": kwargs["conversation_id"],
            "steps": ("observe", "execute", "done"),
            "checkpoint": {"state": dict(kwargs["initial_state"])},
            "provenance": {"run_id": "run-agent-parity"},
        }

    class SyncRuntime:
        def run(self, **kwargs: object) -> dict[str, object]:
            return lifecycle(**kwargs)

    class AsyncRuntime:
        async def run(self, **kwargs: object) -> dict[str, object]:
            return lifecycle(**kwargs)

    profile = AgentProfile(agent_id="a", workflow_id="wf")
    sync_result = AgentHarness(
        profile=profile,
        workflow_runtime=SyncRuntime(),
    ).run(initial_state={"input": "x"}, conversation_id="c")
    async_result = await AsyncAgentHarness(
        profile=profile,
        workflow_runtime=AsyncRuntime(),
    ).run(initial_state={"input": "x"}, conversation_id="c")
    assert async_result == sync_result


def test_profile_context_and_hook_contracts_are_stable_and_bounded() -> None:
    profile = AgentProfile(agent_id="a", workflow_id="wf", hook_ids=["observe"])
    assert profile.stable_json() == profile.stable_json()
    profile.validate_bindings(
        known_workflows={"wf"}, known_hooks={"observe"}, caller_capabilities=[]
    )
    snapshot = ContextSnapshot(
        conversation_id="c", items=[{"id": "n1"}], source_refs=["n1"]
    )
    assert len(snapshot.stable_fingerprint()) == 64
    hooks = HookRegistry()
    hooks.register(
        HookSpec(hook_id="observe", callback=lambda payload: {"seen": payload["id"]})
    )
    result = hooks.run({"id": "n1"})
    assert result[0].annotations == {"seen": "n1"}
    with pytest.raises(ValueError, match="unknown workflow"):
        profile.validate_bindings(known_workflows={"other"})
    profile.validate_bindings(known_model_profiles={"default"})
    with pytest.raises(ValueError, match="unknown model profile"):
        profile.validate_bindings(known_model_profiles={"other"})
    with pytest.raises(PermissionError, match="cannot disable host ACL"):
        AgentProfile(
            agent_id="untrusted", workflow_id="wf", acl_required=False
        ).validate_bindings()


def test_hook_failure_modes_timeout_and_disposal_are_deterministic() -> None:
    closed = {"value": False}

    def observe(_payload: object) -> dict[str, object]:
        return {"ok": True}

    class Closable:
        def __call__(self, _payload: object) -> dict[str, object]:
            return {}

        def close(self) -> None:
            closed["value"] = True

    hooks = HookRegistry()
    hooks.register(HookSpec(hook_id="observe", callback=observe, order=2))
    hooks.register(
        HookSpec(
            hook_id="optional",
            callback=lambda _payload: (_ for _ in ()).throw(RuntimeError("ignored")),
            failure_mode="fail_open",
            order=1,
        )
    )
    hooks.register(HookSpec(hook_id="close", callback=Closable(), order=3))
    results = hooks.run({})
    assert [item.hook_id for item in results] == ["optional", "observe", "close"]
    assert results[0].status == "failed"
    hooks.unregister("close")
    assert closed["value"] is True


def test_sync_hook_timeout_bounds_run_without_blocking_agent_path() -> None:
    def blocked(_payload: object) -> dict[str, object]:
        time.sleep(0.2)
        return {"late": True}

    hooks = HookRegistry()
    hooks.register(HookSpec(hook_id="blocked", callback=blocked, timeout_ms=10))
    started = time.monotonic()
    result = hooks.run({})
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert result[0].status == "failed"
    assert "timed out" in (result[0].error or "")


def test_async_hook_timeout_also_moves_sync_callback_off_event_loop() -> None:
    def blocked(_payload: object) -> dict[str, object]:
        time.sleep(0.2)
        return {}

    hooks = HookRegistry()
    hooks.register(HookSpec(hook_id="blocked", callback=blocked, timeout_ms=10))
    started = time.monotonic()
    result = asyncio.run(hooks.arun({}))
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert result[0].status == "failed"


def test_skill_projection_validates_scope_and_rejects_stale_revision() -> None:
    artifact = parse_skill_text(
        "- Check deployment",
        provider_id="project",
        provider_local_id="deploy",
    ).model_copy(update={"project_id": "p", "projection_revision": 2})
    validate_skill_artifact(artifact, project_id="p")
    store = SkillProjectionStore()
    store.upsert(artifact)
    with pytest.raises(ValueError, match="stale"):
        store.upsert(artifact.model_copy(update={"projection_revision": 1}))
    selected = select_skill_subgraph(artifact, node_ids=[artifact.nodes[0].node_id])
    assert len(selected.nodes) == 1
    assert selected.edges == []


def test_scoped_in_memory_skill_and_catalog_projection_require_exact_scope() -> None:
    artifact = parse_skill_text(
        "- Check deployment",
        provider_id="project",
        provider_local_id="deploy",
    ).model_copy(update={"tenant_id": "tenant-a", "project_id": "project-a"})
    projections = SkillProjectionStore(
        tenant_id="tenant-a",
        project_id="project-a",
        acl_enabled=False,
    )
    projections.upsert(artifact)
    with pytest.raises(PermissionError, match="tenant scope"):
        projections.upsert(artifact.model_copy(update={"tenant_id": "tenant-b"}))
    assert projections.get(
        "project", "deploy", tenant_id="tenant-a", project_id="project-a"
    ) == artifact
    assert projections.get(
        "project", "deploy", tenant_id="tenant-b", project_id="project-a"
    ) is None

    catalog = CatalogStore(
        tenant_id="tenant-a",
        project_id="project-a",
        acl_enabled=False,
    )
    entry = catalog_entries_from_artifact(artifact)[0]
    catalog.upsert(entry)
    with pytest.raises(PermissionError, match="tenant scope"):
        catalog.upsert(entry.model_copy(update={"tenant_id": "tenant-b"}))
    assert catalog.get(
        entry.logical_id, tenant_id="tenant-a", project_id="project-a"
    ) == entry
    assert catalog.get(
        entry.logical_id, tenant_id="tenant-b", project_id="project-a"
    ) is None
    with pytest.raises(PermissionError, match="group tenant scope"):
        catalog.upsert_group(
            CatalogGroup(
                group_id="group:foreign",
                name="Foreign",
                tenant_id="tenant-b",
                project_id="project-a",
            )
        )


def test_plan_and_goal_design_lint_is_advisory() -> None:
    assert validate_agent_design(build_plan_workflow(), mode="plan") == ()
    assert validate_agent_design(build_goal_workflow(), mode="goal") == ()
    assert validate_agent_design(build_plan_workflow(), mode="goal")
