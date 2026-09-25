"""Optional provider plugins are import-safe, bounded, and authority-neutral."""

from __future__ import annotations

import pytest

from kogwistar.agent import (
    CatalogStore,
    CatalogGroup,
    CatalogEntry,
    FilesystemSkillProvider,
    LlmWikiIngestionAdapter,
    McpDiscoveryProvider,
    ProjectGlossaryProvider,
    ProjectPluginManifest,
    ProviderRegistry,
    SkillProjectionStore,
    SkillExecutionPolicy,
    build_skill_execution_evidence,
    ingest_filesystem_skill,
    ingest_project_glossary_to_knowledge,
    deduplicate_inferred_edges,
    mark_inferred_edges_as_candidates,
    select_mcp_schemas,
)
from kogwistar.agent.skills import (
    attach_glossary_references,
    catalog_entries_from_artifact,
    parse_skill_text,
    validate_skill_artifact,
)
from kogwistar.agent.plugins import make_skill_projection_cleanup
from kogwistar.agent import SkillProjectionRequest, prepare_skill_execution


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def test_filesystem_provider_descriptors_are_fingerprintable_and_safe(tmp_path) -> None:
    skill = tmp_path / "deploy.md"
    skill.write_text("# Deploy\n\n1. validate\n", encoding="utf-8")
    provider = FilesystemSkillProvider(tmp_path)
    descriptors = provider.descriptors()
    assert descriptors[0]["provider_local_id"] == "deploy.md"
    assert provider.load("deploy.md").startswith("# Deploy")
    with pytest.raises(PermissionError):
        provider.load("../secret.md")
    registry = ProviderRegistry()
    registry.register(provider_id=provider.provider_id, provider=provider)
    catalog = CatalogStore(acl_enabled=False)
    catalog.ingest_descriptors(registry.discovery_descriptors())
    assert catalog.get("filesystem.skills:deploy.md") is not None


def test_mcp_discovery_does_not_imply_invocation() -> None:
    provider = McpDiscoveryProvider(lambda: [{"provider_local_id": "search", "kind": "mcp", "name": "search", "source_fingerprint": "fp"}])
    assert provider.descriptors()[0]["provider_local_id"] == "search"
    with pytest.raises(PermissionError):
        provider.invoke("search")


def test_mcp_schema_selection_is_explicit_authorized_and_bounded() -> None:
    provider = McpDiscoveryProvider(
        lambda: [],
        describe=lambda item: {"name": item, "input_schema": {"type": "object"}},
    )
    selected = select_mcp_schemas(
        provider, ["search"], authorize=lambda item: item == "search"
    )
    assert list(selected) == ["search"]
    with pytest.raises(PermissionError):
        select_mcp_schemas(provider, ["secret"], authorize=lambda _item: False)
    with pytest.raises(PermissionError, match="authorization callback"):
        select_mcp_schemas(provider, ["search"])


def test_filesystem_skill_ingestion_preserves_raw_provider_and_projection(tmp_path) -> None:
    skill = tmp_path / "deploy.md"
    skill.write_text("# Deploy\n\n1. validate\n", encoding="utf-8")
    provider = FilesystemSkillProvider(tmp_path)
    projection = SkillProjectionStore()
    catalog = CatalogStore(acl_enabled=False)
    artifact = ingest_filesystem_skill(
        provider,
        "deploy.md",
        projection=projection,
        catalog=catalog,
        project_id="project-1",
        tenant_id="tenant-1",
    )
    assert artifact.source_fingerprint
    assert provider.load("deploy.md").startswith("# Deploy")
    projected = projection.get(provider.provider_id, "deploy.md")
    assert projected == artifact
    assert projected is not artifact
    assert catalog.get("filesystem.skills:deploy.md") is not None


def test_provider_failure_can_be_isolated_from_local_discovery(tmp_path) -> None:
    skill = tmp_path / "local.md"
    skill.write_text("# Local\n", encoding="utf-8")
    registry = ProviderRegistry()
    registry.register(provider_id="local", provider=FilesystemSkillProvider(tmp_path))
    registry.register(
        provider_id="broken",
        provider=McpDiscoveryProvider(
            lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        ),
    )
    assert registry.discovery_descriptors(isolate_failures=True)


def test_llm_wiki_adapter_is_optional_and_uses_same_artifact_contract() -> None:
    provider = LlmWikiIngestionAdapter(
        lambda source: parse_skill_text(
            str(source["text"]), provider_id="wiki", provider_local_id="skill"
        ).model_copy(update={"tenant_id": source.get("tenant_id")}),
        authorize=lambda source: source.get("tenant_id") == "tenant-1",
    )
    artifact = provider.parse({"text": "# Skill\n\n1. inspect\n", "tenant_id": "tenant-1"})
    assert artifact.provider_id == "wiki"
    assert artifact.nodes
    with pytest.raises(PermissionError):
        provider.parse({"text": "# Skill", "tenant_id": "tenant-2"})
    with pytest.raises(ValueError, match="byte bound"):
        LlmWikiIngestionAdapter(
            lambda source: artifact,
            authorize=lambda _source: True,
            max_source_bytes=8,
        ).parse({"text": "too large"})

    mismatched = LlmWikiIngestionAdapter(
        lambda _source: parse_skill_text(
            "# Skill", provider_id="wiki", provider_local_id="skill"
        ).model_copy(update={"tenant_id": "tenant-2"}),
        authorize=lambda _source: True,
    )
    with pytest.raises(PermissionError, match="tenant scope"):
        mismatched.parse({"text": "# Skill", "tenant_id": "tenant-1"})


def test_semantic_provider_edges_remain_candidate_with_provenance() -> None:
    artifact = parse_skill_text("# Skill\n\n1. inspect\n", provider_id="wiki", provider_local_id="skill")
    candidate = mark_inferred_edges_as_candidates(artifact, confidence=0.7)
    assert all(edge.status == "candidate" for edge in candidate.edges)
    assert all(edge.metadata["source_provider"] == "wiki" for edge in candidate.edges)
    assert all(edge.metadata["confidence"] == 0.7 for edge in candidate.edges)
    duplicate = candidate.edges[0].model_copy(update={"edge_id": "duplicate", "metadata": {**candidate.edges[0].metadata, "confidence": 0.2}})
    deduped = deduplicate_inferred_edges(candidate.model_copy(update={"edges": [candidate.edges[0], duplicate]}))
    assert len(deduped.edges) == len(candidate.edges)
    assert all(edge.status == "candidate" for edge in deduped.edges)


def test_semantic_provider_failure_keeps_previous_valid_projection() -> None:
    store = SkillProjectionStore()
    valid = parse_skill_text("# v1\n", provider_id="wiki", provider_local_id="skill")
    store.upsert(valid)
    provider = LlmWikiIngestionAdapter(
        lambda _source: (_ for _ in ()).throw(RuntimeError("wiki offline")),
        authorize=lambda _source: True,
    )
    with pytest.raises(RuntimeError, match="offline"):
        provider.parse({"text": "# v2"})
    retained = store.get("wiki", "skill")
    assert retained == valid
    assert retained is not valid


def test_provider_failure_mode_and_unload_cleanup_retract_current_projection_only() -> None:
    class Broken:
        provider_id = "broken"
        provider_version = "v1"

        def descriptors(self):
            raise RuntimeError("offline")

        def close(self):
            return None

    registry = ProviderRegistry()
    provider = Broken()
    registry.register(provider_id="broken", provider=provider, failure_mode="isolate")
    assert registry.discovery_descriptors() == ()
    projection = SkillProjectionStore()
    projection.upsert(parse_skill_text("# A", provider_id="broken", provider_local_id="a"))
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(
        CatalogEntry(
            logical_id="broken:a",
            provider_id="broken",
            provider_local_id="a",
            kind="skill",
            name="a",
            summary="a",
            source_fingerprint="fp",
        )
    )
    registry.unload(
        "broken",
        cleanup=lambda provider_id: (projection.remove_provider(provider_id), catalog.remove_provider(provider_id)),
    )
    assert projection.get("broken", "a") is None
    assert catalog.get("broken:a") is None
    assert catalog.history("broken:a")


def test_provider_cleanup_token_does_not_remove_newer_generation() -> None:
    class Provider:
        provider_id = "project"
        provider_version = "v1"

        def close(self):
            return None

    registry = ProviderRegistry()
    first = registry.register(provider_id="project", provider=Provider(), version="v1")
    second = registry.register(provider_id="project", provider=Provider(), version="v2")
    projection = SkillProjectionStore()
    catalog = CatalogStore(acl_enabled=False)
    for registration, local_id in ((first, "old"), (second, "new")):
        artifact = parse_skill_text(
            f"# {local_id}", provider_id="project", provider_local_id=local_id
        ).model_copy(
            update={
                "provenance": {
                    "provider_version": registration.identity.version,
                    "provider_lifecycle_token": registration.lifecycle_token,
                }
            }
        )
        projection.upsert(artifact)
        catalog.ingest_descriptors(catalog_entries_from_artifact(artifact))
    cleanup = make_skill_projection_cleanup(projection=projection, catalog=catalog)
    registry.unload("project", "v1", cleanup=cleanup)
    assert projection.get("project", "old") is None
    assert projection.get("project", "new") is not None
    assert catalog.get("project:new") is not None


def test_project_glossary_knowledge_records_and_skill_refs_remain_separate() -> None:
    provider = ProjectGlossaryProvider(
        ProjectPluginManifest("project.plugin", "project-1", "tenant-1"),
        [{"id": "term-1", "term": "DW", "aliases": ["deploy window"], "definition": "release period"}],
    )
    records: list[dict[str, object]] = []
    refs = ingest_project_glossary_to_knowledge(provider, write=lambda record: records.append(dict(record)) or "knowledge:term-1")
    assert refs == ("knowledge:term-1",)
    assert records[0]["namespace"] == "project:project-1:knowledge"
    artifact = parse_skill_text("# Skill\n", provider_id="project", provider_local_id="skill")
    node_id = artifact.nodes[0].node_id
    linked = attach_glossary_references(artifact, glossary_entity_ids={node_id: "term-1"})
    assert linked.nodes[0].metadata["glossary_entity_ids"] == ["term-1"]


def test_stale_semantic_parse_cannot_replace_newer_projection() -> None:
    store = SkillProjectionStore()
    v2 = parse_skill_text(
        "# v2", provider_id="wiki", provider_local_id="skill"
    ).model_copy(update={"projection_revision": 2})
    v1 = v2.model_copy(update={"projection_revision": 1, "source_fingerprint": "old"})
    store.upsert(v2)
    with pytest.raises(ValueError, match="stale"):
        store.upsert(v1)


def test_project_glossary_requires_acl_and_supports_alias_search() -> None:
    provider = ProjectGlossaryProvider(
        ProjectPluginManifest("project.plugin", "project-1", "tenant-1"),
        [{"term": "Deploy Window", "aliases": ["DW"], "definition": "release period", "id": "term-1"}],
    )
    with pytest.raises(PermissionError):
        provider.search("DW")
    results = provider.search("dw", authorize=lambda term: term["id"] == "term-1")
    assert results[0]["term"] == "Deploy Window"


def test_skill_projection_uses_fixed_workspace_lane_and_execution_is_bounded() -> None:
    request = SkillProjectionRequest(
        provider_id="project",
        provider_local_id="deploy",
        source_fingerprint="fp",
        project_id="acme",
        tenant_id="tenant-1",
    )
    assert request.lane_id == "ws:tenant-1:acme:g:projection:lane:skills"
    artifact = parse_skill_text(
        "# Deploy\n\nCapability: graph.read\n\nMCP: project.deploy\n\n1. inspect",
        provider_id="project",
        provider_local_id="deploy",
    )
    plan = prepare_skill_execution(
        artifact,
        node_ids=[node.node_id for node in artifact.nodes],
        effective_capabilities={"graph.read"},
    )
    assert plan.source_fingerprint == artifact.source_fingerprint
    assert plan.required_capabilities == ["graph.read"]
    assert plan.mcp_schema_refs == ["project.deploy"]
    assert plan.evidence_refs


def test_effectful_skill_requires_explicit_bounded_execution_policy_and_emits_evidence() -> None:
    artifact = parse_skill_text(
        "# Deploy\n\n```bash\nkubectl get pods\n```\n",
        provider_id="project",
        provider_local_id="deploy",
    ).model_copy(update={"projection_revision": 3})
    command_id = next(node.node_id for node in artifact.nodes if node.kind == "command_template")
    with pytest.raises(PermissionError, match="not authorized"):
        prepare_skill_execution(
            artifact,
            node_ids=[command_id],
            effective_capabilities={"process.execute"},
        )
    bound_node = next(node for node in artifact.nodes if node.node_id == command_id).model_copy(
        update={"binding_status": "validated", "invocable": True}
    )
    artifact = artifact.model_copy(
        update={
            "nodes": [
                bound_node if node.node_id == command_id else node
                for node in artifact.nodes
            ]
        }
    )
    policy = SkillExecutionPolicy(
        max_output_bytes=4096,
        max_time_ms=5000,
        cwd="scripts",
        environment_keys=["KUBECONFIG"],
        sandbox="skill-default",
    )
    plan = prepare_skill_execution(
        artifact,
        node_ids=[command_id],
        effective_capabilities={"process.execute"},
        execution_policy=policy,
    )
    evidence = build_skill_execution_evidence(
        plan,
        run_id="run-1",
        workflow_step_ref="step-2",
        outcome="success",
        result_ref="artifact:result-1",
    )
    assert evidence.state_patch()["skill_execution_evidence"]["source_fingerprint"] == artifact.source_fingerprint
    assert evidence.mcp_schema_refs == []


def test_catalog_supports_multiple_groups_without_duplicate_identity() -> None:
    catalog = CatalogStore(acl_enabled=False)
    entry = catalog_entries_from_artifact(
        parse_skill_text("# Shared\n", provider_id="project", provider_local_id="shared")
    )[0]
    catalog.upsert(entry.model_copy(update={"group_ids": ["group:ops", "group:platform"]}))
    catalog.upsert_group(CatalogGroup(group_id="group:ops", name="Operations"))
    catalog.upsert_group(CatalogGroup(group_id="group:platform", name="Platform"))
    assert [item.logical_id for item in catalog.browse("group:ops")] == ["project:shared"]
    assert [item.logical_id for item in catalog.browse("group:platform")] == ["project:shared"]
    assert len(catalog.search("shared")) == 1


def test_skill_artifact_rejects_scope_fingerprint_and_size_violations() -> None:
    artifact = parse_skill_text("# Skill\n", provider_id="project", provider_local_id="skill").model_copy(
        update={"tenant_id": "tenant-a", "project_id": "project-a"}
    )
    with pytest.raises(ValueError, match="fingerprint"):
        validate_skill_artifact(artifact, expected_fingerprint="wrong")
    with pytest.raises(PermissionError, match="tenant"):
        validate_skill_artifact(artifact, tenant_id="tenant-b")
    with pytest.raises(PermissionError, match="project"):
        validate_skill_artifact(artifact, project_id="project-b")
    oversized = artifact.model_copy(
        update={
            "nodes": [node.model_copy(update={"node_id": f"n-{index}"}) for index, node in enumerate(artifact.nodes * 257)]
        }
    )
    with pytest.raises(ValueError, match="exceeds"):
        type(artifact).model_validate(oversized.model_dump(mode="python"))
