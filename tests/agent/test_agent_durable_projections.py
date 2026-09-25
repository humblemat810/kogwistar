"""Deterministic durable skill/catalog projection acceptance tests."""

from threading import Event, Thread
import asyncio

import pytest

from kogwistar.agent import (
    CatalogEntry,
    CatalogGroup,
    CatalogStore,
    DurableCatalogStore,
    DurableSkillCatalogMaterializer,
    AsyncDurableSkillCatalogMaterializer,
    DurableSkillProjectionStore,
    ProviderInactiveError,
    ProviderCollisionError,
    ProviderRegistry,
    SkillProjectionStore,
    catalog_entries_from_artifact,
    compile_approved_proposal_to_skill,
    materialize_skill_artifact,
)
from kogwistar.agent.read_tools import AgentReadTools, ReadScope
from kogwistar.agent.plugins import (
    FilesystemSkillProvider,
    LlmWikiIngestionAdapter,
    ingest_filesystem_skill,
)
from kogwistar.agent.skills import parse_skill_text
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.async_named_projection import AsyncSQLiteNamedProjectionStore
from kogwistar.wisdom.proposals import ProposalEvaluation, WisdomRevisionProposal


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def _artifact(version: str, revision: int):
    return parse_skill_text(
        f"# Deploy {version}\n\n1. validate {version}\n",
        provider_id="project",
        provider_local_id="deploy",
        skill_version=version,
    ).model_copy(update={"projection_revision": revision, "tenant_id": "tenant-1", "project_id": "project-1"})


def test_skill_projection_survives_restart_and_rejects_stale_revision() -> None:
    metadata = InMemoryMetaStore()
    first = _artifact("v1", 1)
    store = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    assert store.upsert(first).source_fingerprint == first.source_fingerprint

    restarted = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    assert restarted.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") == first

    second = _artifact("v2", 2)
    restarted.upsert(second)
    assert DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    ).get("project", "deploy", tenant_id="tenant-1", project_id="project-1") == second
    with pytest.raises(ValueError, match="stale"):
        store.upsert(first)
    assert [item.projection_revision for item in restarted.history("project", "deploy", tenant_id="tenant-1", project_id="project-1")] == [1, 2]
    rows = metadata.list_named_projections(restarted.namespace)
    skill_revisions = [row for row in rows if str(row["key"]).startswith("skill-revision:")]
    skill_pointers = [row for row in rows if str(row["key"]).startswith("skill-current:")]
    assert len(skill_revisions) == 2
    assert len(skill_pointers) == 1
    assert all("history" not in row["payload"] for row in skill_revisions)
    restarted.remove("project", "deploy")
    assert restarted.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") is None
    restarted.upsert(second)
    assert restarted.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") == second
    assert [item.projection_revision for item in restarted.history("project", "deploy", tenant_id="tenant-1", project_id="project-1")] == [1, 2]


def test_catalog_projection_survives_restart_enforces_acl_and_returns_copies() -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    entry = catalog_entries_from_artifact(artifact)[0]
    catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda item, principal: principal == "alice" and item.tenant_id == "tenant-1",
    )
    catalog.upsert(entry)

    restarted = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda item, principal: principal == "alice" and item.tenant_id == "tenant-1",
    )
    assert restarted.get(entry.logical_id, principal="bob", tenant_id="tenant-1", project_id="project-1") is None
    assert restarted.history(entry.logical_id, principal="bob", tenant_id="tenant-1", project_id="project-1") == ()
    visible = restarted.get(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1")
    assert visible is not None
    visible.name = "mutated locally"
    assert restarted.get(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1").name == entry.name
    assert [item.revision for item in restarted.history(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1")] == [1]
    restarted.remove_provider(entry.provider_id)
    assert restarted.get(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1") is None
    restarted.upsert(entry)
    assert restarted.get(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1") == entry
    assert [item.revision for item in restarted.history(entry.logical_id, principal="alice", tenant_id="tenant-1", project_id="project-1")] == [1]


def test_catalog_uses_immutable_revision_rows_and_current_pointer() -> None:
    metadata = InMemoryMetaStore()
    first = catalog_entries_from_artifact(_artifact("v1", 1))[0]
    second = catalog_entries_from_artifact(_artifact("v2", 2))[0]
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog.upsert(first)
    catalog.upsert(second)
    assert [entry.revision for entry in catalog.history(
        first.logical_id, tenant_id="tenant-1", project_id="project-1"
    )] == [1, 2]
    rows = metadata.list_named_projections(catalog._projection_namespace)
    revisions = [row for row in rows if str(row["key"]).startswith("catalog-revision:")]
    pointers = [row for row in rows if str(row["key"]).startswith("catalog-current:")]
    assert len(revisions) == 2
    assert len(pointers) == 1
    assert all("history" not in row["payload"] for row in revisions)


def test_catalog_revision_is_carried_from_skill_artifact() -> None:
    artifact = _artifact("v2", 2)
    entries = catalog_entries_from_artifact(artifact)
    assert entries
    assert all(isinstance(entry, CatalogEntry) and entry.revision == 2 for entry in entries)


def test_provider_scope_mismatch_is_not_accepted_by_durable_projection() -> None:
    metadata = InMemoryMetaStore()
    store = DurableSkillProjectionStore(metadata, tenant_id="tenant-1", project_id="project-1")
    foreign = _artifact("foreign", 1).model_copy(update={"tenant_id": "tenant-2"})
    with pytest.raises(PermissionError):
        store.upsert(foreign)


def test_durable_projection_identity_isolated_by_tenant_and_project() -> None:
    metadata = InMemoryMetaStore()
    first = _artifact("v1", 1)
    second = first.model_copy(update={"tenant_id": "tenant-2"})
    first_store = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    second_store = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-2",
        project_id="project-1",
        acl_enabled=False,
    )
    first_store.upsert(first)
    second_store.upsert(second)
    assert first_store.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") == first
    assert second_store.get("project", "deploy", tenant_id="tenant-2", project_id="project-1") == second
    assert first_store.get("project", "deploy", tenant_id="tenant-2", project_id="project-1") is None

    first_catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    second_catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-2", project_id="project-1", acl_enabled=False
    )
    first_entry = catalog_entries_from_artifact(first)[0]
    second_entry = catalog_entries_from_artifact(second)[0]
    first_catalog.upsert(first_entry)
    second_catalog.upsert(second_entry)
    assert first_catalog.get(
        first_entry.logical_id, tenant_id="tenant-1", project_id="project-1"
    ) == first_entry
    assert second_catalog.get(
        second_entry.logical_id, tenant_id="tenant-2", project_id="project-1"
    ) == second_entry


def test_durable_projection_direct_read_is_acl_and_scope_gated() -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    store = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _artifact, principal: principal == "alice",
    )
    store.upsert(artifact)
    assert store.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") is None
    assert store.get("project", "deploy", principal="bob", tenant_id="tenant-1", project_id="project-1") is None
    assert store.get("project", "deploy", principal="alice", tenant_id="tenant-2", project_id="project-1") is None
    assert store.get("project", "deploy", principal="alice", tenant_id="tenant-1", project_id="project-1") == artifact


def test_materializer_commits_catalog_and_artifact_for_restart_safe_reading() -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    projections = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _artifact, principal: principal == "alice",
    )
    catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _entry, principal: principal == "alice",
    )
    DurableSkillCatalogMaterializer(projections=projections, catalog=catalog).materialize(artifact)

    restarted_projections = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _artifact, principal: principal == "alice",
    )
    restarted_catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _entry, principal: principal == "alice",
    )
    root_id = catalog_entries_from_artifact(artifact)[0].logical_id
    tools = AgentReadTools(
        catalog=restarted_catalog,
        skill_projection_store=restarted_projections,
        acl_required=False,
    )
    graph = tools.skill_get(
        root_id,
        scope=ReadScope(principal_id="alice", tenant_id="tenant-1", project_id="project-1"),
        representation="graph",
    )
    assert graph is not None
    assert graph["artifact"]["provider_local_id"] == "deploy"
    native_graph = restarted_projections.graph(
        "project",
        "deploy",
        principal="alice",
        tenant_id="tenant-1",
        project_id="project-1",
    )
    assert native_graph is not None
    assert native_graph["revision"] == artifact.projection_revision
    assert {node["node_id"] for node in native_graph["nodes"]} == {
        node.node_id for node in artifact.nodes
    }
    assert {edge["edge_id"] for edge in native_graph["edges"]} == {
        edge.edge_id for edge in artifact.edges
    }
    restricted = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_checker=lambda _artifact, principal: principal == "alice",
    )
    assert restricted.graph(
        "project",
        "deploy",
        principal="bob",
        tenant_id="tenant-1",
        project_id="project-1",
    ) is None


def test_async_materializer_uses_async_named_projection_contract() -> None:
    class _AsyncMetadata:
        def __init__(self) -> None:
            self.inner = InMemoryMetaStore()
            self.calls: list[str] = []

        async def get_named_projection(self, namespace, key):
            self.calls.append("get")
            await asyncio.sleep(0)
            return self.inner.get_named_projection(namespace, key)

        async def list_named_projections(self, namespace):
            self.calls.append("list")
            await asyncio.sleep(0)
            return self.inner.list_named_projections(namespace)

        async def compare_and_swap_named_projection(self, namespace, key, payload, **values):
            self.calls.append("cas")
            await asyncio.sleep(0)
            return self.inner.compare_and_swap_named_projection(
                namespace, key, payload, **values
            )

        async def compare_and_swap_named_projections(self, updates):
            self.calls.append("batch")
            await asyncio.sleep(0)
            return self.inner.compare_and_swap_named_projections(updates)

    async_metadata = _AsyncMetadata()
    artifact = _artifact("async-v1", 1)

    async def _run() -> None:
        materializer = AsyncDurableSkillCatalogMaterializer(
            metadata=async_metadata,
            tenant_id="tenant-1",
            project_id="project-1",
        )
        await materializer.materialize(artifact)
        assert await materializer.reconcile() == 1

    asyncio.run(_run())
    assert "batch" in async_metadata.calls
    projections = DurableSkillProjectionStore(
        async_metadata.inner,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    assert projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) == artifact


def test_async_sqlite_materializer_preserves_sqlite_batch_atomicity(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "async-meta")
    async_metadata = AsyncSQLiteNamedProjectionStore(metadata)
    artifact = _artifact("async-sqlite-v1", 1)

    async def _run() -> None:
        await async_metadata.ensure_initialized()
        materializer = AsyncDurableSkillCatalogMaterializer(
            metadata=async_metadata,
            tenant_id="tenant-1",
            project_id="project-1",
        )
        await materializer.materialize(artifact)
        assert await materializer.reconcile() == 1

    asyncio.run(_run())
    restarted = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    assert restarted.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) == artifact


def test_atomic_materializer_rejection_leaves_no_half_catalog_or_artifact(monkeypatch) -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    monkeypatch.setattr(metadata, "compare_and_swap_named_projections", lambda _updates: False)
    with pytest.raises(ValueError, match="changed concurrently"):
        DurableSkillCatalogMaterializer(projections=projections, catalog=catalog).materialize(artifact)
    assert projections.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") is None
    assert catalog.get("project:deploy", tenant_id="tenant-1", project_id="project-1") is None


def test_filesystem_ingestion_uses_one_durable_materialization_commit(tmp_path) -> None:
    skill_path = tmp_path / "deploy.md"
    skill_path.write_text("# Deploy\n\n1. validate\n", encoding="utf-8")
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    artifact = ingest_filesystem_skill(
        FilesystemSkillProvider(tmp_path),
        "deploy.md",
        projection=projections,
        catalog=catalog,
        materializer=DurableSkillCatalogMaterializer(projections=projections, catalog=catalog),
        tenant_id="tenant-1",
        project_id="project-1",
    )
    assert projections.get("filesystem.skills", "deploy.md", tenant_id="tenant-1", project_id="project-1") == artifact
    assert catalog.get(
        "filesystem.skills:deploy.md", tenant_id="tenant-1", project_id="project-1"
    ) is not None


def test_retired_provider_parse_cannot_resurrect_durable_projection(tmp_path) -> None:
    skill_path = tmp_path / "deploy.md"
    skill_path.write_text("# Deploy\n\n1. validate\n", encoding="utf-8")
    provider = FilesystemSkillProvider(tmp_path)
    registry = ProviderRegistry()
    registration = registry.register(provider_id=provider.provider_id, provider=provider)
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    registry.dispose(registration)
    with pytest.raises(ProviderInactiveError):
        ingest_filesystem_skill(
            provider,
            "deploy.md",
            projection=projections,
            catalog=catalog,
            materializer=DurableSkillCatalogMaterializer(
                projections=projections, catalog=catalog
            ),
            provider_registry=registry,
            provider_registration=registration,
            tenant_id="tenant-1",
            project_id="project-1",
        )
    assert projections.get(
        provider.provider_id,
        "deploy.md",
        tenant_id="tenant-1",
        project_id="project-1",
    ) is None
    assert catalog.get(
        f"{provider.provider_id}:deploy.md",
        tenant_id="tenant-1",
        project_id="project-1",
    ) is None


def test_provider_lifecycle_token_blocks_stale_artifact_after_restart() -> None:
    metadata = InMemoryMetaStore()
    provider = object()
    first_registry = ProviderRegistry(metadata=metadata)
    first = first_registry.register(provider_id="project", provider=provider)
    artifact = _artifact("v1", 1).model_copy(
        update={
            "provenance": {
                "provider_lifecycle_token": first.lifecycle_token,
            }
        }
    )
    first_registry.dispose(first)

    second_registry = ProviderRegistry(metadata=metadata)
    second = second_registry.register(provider_id="project", provider=provider)
    assert second.generation > first.generation
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    with pytest.raises(ProviderInactiveError, match="earlier provider lifecycle"):
        materialize_skill_artifact(
            artifact,
            projection=projections,
            catalog=catalog,
            materializer=DurableSkillCatalogMaterializer(
                projections=projections, catalog=catalog
            ),
            provider_registry=second_registry,
            provider_registration=second,
            expected_provider=provider,
        )


def test_provider_registry_publishes_local_state_only_after_lifecycle_cas() -> None:
    metadata = InMemoryMetaStore()
    registry = ProviderRegistry(metadata=metadata)
    original = metadata.compare_and_swap_named_projection

    def reject(*args, **kwargs):
        return False

    metadata.compare_and_swap_named_projection = reject
    with pytest.raises(ProviderCollisionError):
        registry.register(provider_id="project", provider=object())
    assert registry.list() == ()
    metadata.compare_and_swap_named_projection = original


def test_materialization_batch_rejects_lifecycle_change_after_local_check() -> None:
    metadata = InMemoryMetaStore()
    registry = ProviderRegistry(metadata=metadata)
    provider = object()
    registration = registry.register(provider_id="project", provider=provider)
    artifact = _artifact("guarded", 1).model_copy(
        update={
            "provenance": {
                "provider_lifecycle_token": registration.lifecycle_token,
            }
        }
    )
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections, catalog=catalog
    )
    original_batch = metadata.compare_and_swap_named_projections
    original_single = metadata.compare_and_swap_named_projection

    def retire_then_reject(updates):
        row = metadata.get_named_projection(
            "agent_provider_lifecycle", "project@v1"
        )
        assert row is not None
        payload = dict(row["payload"])
        assert original_single(
            "agent_provider_lifecycle",
            "project@v1",
            payload,
            expected_last_authoritative_seq=row["last_authoritative_seq"],
            expected_last_materialized_seq=row["last_materialized_seq"],
            last_authoritative_seq=row["last_authoritative_seq"] + 1,
            last_materialized_seq=row["last_materialized_seq"] + 1,
            projection_schema_version=1,
            materialization_status="retired",
        )
        return original_batch(updates)

    metadata.compare_and_swap_named_projections = retire_then_reject
    try:
        with pytest.raises(ValueError, match="changed concurrently"):
            materialize_skill_artifact(
                artifact,
                projection=projections,
                catalog=catalog,
                materializer=materializer,
                provider_registry=registry,
                provider_registration=registration,
                expected_provider=provider,
            )
    finally:
        metadata.compare_and_swap_named_projections = original_batch
    assert projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None


def test_provider_dispose_keeps_local_owner_when_lifecycle_cas_fails() -> None:
    metadata = InMemoryMetaStore()
    registry = ProviderRegistry(metadata=metadata)
    registration = registry.register(provider_id="project", provider=object())
    original = metadata.compare_and_swap_named_projection
    metadata.compare_and_swap_named_projection = lambda *args, **kwargs: False
    with pytest.raises(ProviderCollisionError):
        registry.dispose(registration)
    assert registry.get("project") is registration
    metadata.compare_and_swap_named_projection = original


def test_unload_serializes_with_active_materialization_then_retracts_it() -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    materializer = DurableSkillCatalogMaterializer(projections=projections, catalog=catalog)
    registry = ProviderRegistry()
    provider = object()
    registration = registry.register(provider_id="project", provider=provider)
    entered = Event()
    release = Event()
    cleaned = Event()

    def commit() -> None:
        def operation() -> None:
            entered.set()
            assert release.wait(timeout=2)
            materializer.materialize(artifact)

        registry.run_if_active(registration, operation)

    def unload() -> None:
        registry.unload(
            "project",
            cleanup=lambda provider_id: (
                projections.remove_provider(provider_id),
                catalog.remove_provider(provider_id),
                cleaned.set(),
            ),
        )

    commit_thread = Thread(target=commit)
    commit_thread.start()
    assert entered.wait(timeout=2)
    unload_thread = Thread(target=unload)
    unload_thread.start()
    assert not cleaned.wait(timeout=0.05)
    release.set()
    commit_thread.join(timeout=2)
    unload_thread.join(timeout=2)
    assert not commit_thread.is_alive()
    assert not unload_thread.is_alive()
    assert cleaned.is_set()
    assert projections.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") is None
    assert catalog.get("project:deploy", tenant_id="tenant-1", project_id="project-1") is None


def test_durable_provider_cleanup_retires_graph_and_catalog_in_one_batch() -> None:
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections, catalog=catalog
    )
    artifact = _artifact("v1", 1)
    materializer.materialize(artifact)

    original = metadata.compare_and_swap_named_projections
    metadata.compare_and_swap_named_projections = lambda _updates: False
    with pytest.raises(ValueError, match="cleanup changed concurrently"):
        materializer.remove_provider("project")
    assert projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    metadata.compare_and_swap_named_projections = original

    removed = materializer.remove_provider("project")
    assert "project:deploy" in removed
    assert any(key.startswith("project:deploy#step:") for key in removed)
    assert projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert projections.graph(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert catalog.get(
        "project:deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None


def test_durable_reconcile_repairs_crash_between_revision_and_serving_pointers() -> None:
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections, catalog=catalog
    )
    artifact = _artifact("v1", 1)
    _, prepared = projections.prepare_upsert_update(artifact)
    revision_only = [prepared[0]]
    assert metadata.compare_and_swap_named_projections(revision_only)

    restarted_projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    restarted_catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    restarted = DurableSkillCatalogMaterializer(
        projections=restarted_projections, catalog=restarted_catalog
    )
    assert restarted_projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert restarted.reconcile() == 1
    assert restarted_projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    assert restarted_projections.graph(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    assert restarted_catalog.get(
        "project:deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None


def test_wisdom_approved_skill_can_materialize_with_catalog_atomically() -> None:
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    proposal = WisdomRevisionProposal(
        proposal_id="proposal-1",
        workflow_id="workflow-1",
        run_id="run-1",
        summary="Inspect before deployment.",
        step_op="inspect",
        reasoning_trace=["failure pattern"],
        evidence_run_ids=["run-1"],
        confidence=0.9,
        created_at_ms=1,
        status="approved",
    )
    evaluation = ProposalEvaluation(
        result_id="evaluation-1",
        proposal_id="proposal-1",
        decision="approved",
        rationale="reviewed",
        result_kind="wisdom_lesson",
        created_at_ms=2,
    )
    artifact = compile_approved_proposal_to_skill(
        proposal,
        evaluation,
        store=projections,
        materializer=DurableSkillCatalogMaterializer(projections=projections, catalog=catalog),
        tenant_id="tenant-1",
        project_id="project-1",
    )
    assert artifact is not None
    assert projections.get(
        "kogwistar.wisdom", "proposal-1", tenant_id="tenant-1", project_id="project-1"
    ) == artifact
    assert catalog.get(
        "kogwistar.wisdom:proposal-1", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    restarted = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    assert restarted.get(
        "kogwistar.wisdom",
        "proposal-1",
        tenant_id="tenant-1",
        project_id="project-1",
    ) == artifact


def test_wisdom_durable_revisions_keep_prior_lineage_after_restart() -> None:
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    proposal = WisdomRevisionProposal(
        proposal_id="proposal-1",
        workflow_id="workflow-1",
        run_id="run-1",
        summary="Inspect before deployment.",
        step_op="inspect",
        reasoning_trace=["failure pattern"],
        evidence_run_ids=["run-1"],
        confidence=0.9,
        created_at_ms=1,
        status="approved",
    )
    evaluation = ProposalEvaluation(
        result_id="evaluation-1",
        proposal_id="proposal-1",
        decision="approved",
        rationale="reviewed",
        result_kind="wisdom_lesson",
        created_at_ms=2,
    )
    materializer = DurableSkillCatalogMaterializer(projections=projections, catalog=catalog)
    first = compile_approved_proposal_to_skill(
        proposal,
        evaluation,
        store=projections,
        materializer=materializer,
        tenant_id="tenant-1",
        project_id="project-1",
        projection_revision=1,
    )
    second = compile_approved_proposal_to_skill(
        proposal,
        evaluation,
        store=projections,
        materializer=materializer,
        tenant_id="tenant-1",
        project_id="project-1",
        projection_revision=2,
    )
    assert first is not None and second is not None
    restarted = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    history = restarted.history(
        "kogwistar.wisdom",
        "proposal-1",
        tenant_id="tenant-1",
        project_id="project-1",
    )
    assert [item.projection_revision for item in history] == [1, 2]
    assert all(item.provenance["proposal_id"] == proposal.proposal_id for item in history)


def test_llm_wiki_artifact_uses_same_durable_materialization_path(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "llm-wiki-meta")
    metadata.ensure_initialized()
    artifact = parse_skill_text(
        "# Deploy\n\n1. validate\n",
        provider_id="wiki",
        provider_local_id="deploy",
        skill_version="v1",
    ).model_copy(
        update={
            "projection_revision": 1,
            "tenant_id": "tenant-1",
            "project_id": "project-1",
        }
    )
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    provider = object()
    registry = ProviderRegistry()
    registration = registry.register(provider_id="wiki", provider=provider)
    artifact = artifact.model_copy(
        update={
            "provenance": {
                **artifact.provenance,
                "provider_lifecycle_token": registration.lifecycle_token,
            }
        }
    )
    materialized = materialize_skill_artifact(
        artifact,
        projection=projections,
        catalog=catalog,
        materializer=DurableSkillCatalogMaterializer(projections=projections, catalog=catalog),
        provider_registry=registry,
        provider_registration=registration,
    )
    assert materialized.provider_id == artifact.provider_id
    assert materialized.provider_local_id == artifact.provider_local_id
    assert materialized.provenance["provider_lifecycle_token"] == registration.lifecycle_token
    assert projections.get(
        "wiki", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) == materialized
    assert catalog.get("project:deploy", tenant_id="tenant-1", project_id="project-1") is None
    assert catalog.get("wiki:deploy", tenant_id="tenant-1", project_id="project-1") is not None
    restarted_metadata = EngineSQLite(tmp_path / "llm-wiki-meta")
    restarted_metadata.ensure_initialized()
    restarted_projections = DurableSkillProjectionStore(
        restarted_metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    restarted_catalog = DurableCatalogStore(
        restarted_metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    assert restarted_projections.graph(
        "wiki", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    assert restarted_catalog.get(
        "wiki:deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None


def test_llm_wiki_failure_keeps_previous_durable_projection_after_restart(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "llm-wiki-failure-meta")
    metadata.ensure_initialized()
    artifact = _artifact("v1", 1).model_copy(
        update={"provider_id": "wiki", "tenant_id": "tenant-1", "project_id": "project-1"}
    )
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections, catalog=catalog
    )
    materializer.materialize(artifact)
    failing = LlmWikiIngestionAdapter(
        lambda _source: (_ for _ in ()).throw(RuntimeError("wiki offline")),
        authorize=lambda _source: True,
    )
    with pytest.raises(RuntimeError, match="wiki offline"):
        failing.parse({"text": "# replacement"})

    restarted_metadata = EngineSQLite(tmp_path / "llm-wiki-failure-meta")
    restarted_metadata.ensure_initialized()
    restarted = DurableSkillProjectionStore(
        restarted_metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    try:
        assert restarted.get(
            "wiki", "deploy", tenant_id="tenant-1", project_id="project-1"
        ) == artifact
    finally:
        close = getattr(restarted_metadata, "close", None)
        if callable(close):
            close()


def test_durable_materialization_rejects_missing_provider_lifecycle_token() -> None:
    metadata = InMemoryMetaStore()
    projections = DurableSkillProjectionStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    catalog = DurableCatalogStore(
        metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
    )
    provider = object()
    registry = ProviderRegistry()
    registration = registry.register(provider_id="project", provider=provider)
    with pytest.raises(ProviderInactiveError, match="missing provider lifecycle token"):
        materialize_skill_artifact(
            _artifact("missing-token", 1).model_copy(
                update={"tenant_id": "tenant-1", "project_id": "project-1"}
            ),
            projection=projections,
            catalog=catalog,
            materializer=DurableSkillCatalogMaterializer(
                projections=projections, catalog=catalog
            ),
            provider_registry=registry,
            provider_registration=registration,
            expected_provider=provider,
        )


def test_same_revision_requires_normalized_artifact_content_match() -> None:
    metadata = InMemoryMetaStore()
    artifact = _artifact("v1", 1)
    store = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    store.upsert(artifact)
    with pytest.raises(ValueError, match="content collision"):
        store.upsert(artifact.model_copy(update={"parser_version": "v2"}))


def test_public_projection_and_catalog_reads_return_copies() -> None:
    artifact = _artifact("v1", 1)
    projections = SkillProjectionStore()
    projections.upsert(artifact)
    returned = projections.get("project", "deploy")
    assert returned is not None
    returned.nodes[0].name = "mutated"
    assert projections.get("project", "deploy").nodes[0].name != "mutated"

    entry = catalog_entries_from_artifact(artifact)[0]
    catalog = CatalogStore(acl_enabled=False)
    catalog.upsert(entry)
    catalog.upsert_group(CatalogGroup(group_id="group:ops", name="Operations"))
    catalog.browse()[0].name = "mutated"
    catalog.search(entry.name)[0].entry.name = "mutated again"
    catalog.group_tree()[0].name = "mutated group"
    assert catalog.get(entry.logical_id).name == entry.name
    assert catalog.group_tree()[0].name == "Operations"


def test_durable_catalog_group_tree_is_scope_and_acl_filtered() -> None:
    metadata = InMemoryMetaStore()
    catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        group_acl_checker=lambda group, principal: principal == "alice",
    )
    catalog.upsert_group(
        CatalogGroup(
            group_id="group:ops",
            name="Operations",
            tenant_id="tenant-1",
            project_id="project-1",
        )
    )
    assert catalog.group_tree(principal="bob", tenant_id="tenant-1", project_id="project-1") == ()
    assert catalog.group_tree(principal="alice", tenant_id="tenant-2", project_id="project-1") == ()
    assert catalog.group_tree(principal="alice", tenant_id="tenant-1", project_id="project-1")[0].group_id == "group:ops"


def test_sqlite_named_projection_survives_store_recreation(tmp_path) -> None:
    metadata = EngineSQLite(tmp_path / "agent-meta")
    metadata.ensure_initialized()
    try:
        artifact = _artifact("v1", 1)
        DurableSkillProjectionStore(
            metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
        ).upsert(artifact)
        restarted = DurableSkillProjectionStore(
            metadata, tenant_id="tenant-1", project_id="project-1", acl_enabled=False
        )
        assert restarted.get("project", "deploy", tenant_id="tenant-1", project_id="project-1") == artifact
    finally:
        close = getattr(metadata, "close", None)
        if callable(close):
            close()
