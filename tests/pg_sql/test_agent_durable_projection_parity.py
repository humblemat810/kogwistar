"""Live PostgreSQL parity for agent skill/catalog projections."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from kogwistar.agent import (
    AsyncDurableSkillCatalogMaterializer,
    DurableCatalogStore,
    DurableSkillCatalogMaterializer,
    DurableSkillProjectionStore,
)
from kogwistar.agent.catalog import scoped_projection_namespace
from kogwistar.agent.skills import parse_skill_text
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.async_named_projection import AsyncPostgresNamedProjectionStore
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore
from tests.conftest import _run_async_windows_safe


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.requires_pgvector]


def test_agent_projection_pg_restart_reconcile_and_uninstall(sa_engine, pg_schema) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture unavailable")
    metadata = EnginePostgresMetaStore(engine=sa_engine, schema=pg_schema)
    metadata.ensure_initialized()
    artifact = parse_skill_text(
        "# Deploy\n\n1. validate\n",
        provider_id="project",
        provider_local_id="deploy",
    ).model_copy(
        update={
            "tenant_id": "tenant-pg",
            "project_id": "project-pg",
            "projection_revision": 1,
        }
    )
    projections = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-pg",
        project_id="project-pg",
        acl_enabled=False,
    )
    catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-pg",
        project_id="project-pg",
        acl_enabled=False,
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections,
        catalog=catalog,
    )
    materializer.materialize(artifact)
    crash_artifact = parse_skill_text(
        "# Deploy crash window\n\n1. validate\n",
        provider_id="project",
        provider_local_id="deploy-crash",
    ).model_copy(
        update={
            "tenant_id": "tenant-pg",
            "project_id": "project-pg",
            "projection_revision": 1,
        }
    )
    _, crash_updates = projections.prepare_upsert_update(crash_artifact)
    assert crash_updates
    # Simulate a process stop after immutable revision persistence and before
    # current/graph/catalog serving pointers are promoted.
    assert metadata.compare_and_swap_named_projections(crash_updates[:1])

    restarted = DurableSkillCatalogMaterializer(
        projections=DurableSkillProjectionStore(
            metadata,
            tenant_id="tenant-pg",
            project_id="project-pg",
            acl_enabled=False,
        ),
        catalog=DurableCatalogStore(
            metadata,
            tenant_id="tenant-pg",
            project_id="project-pg",
            acl_enabled=False,
        ),
    )
    assert restarted.projections.get(
        "project", "deploy", tenant_id="tenant-pg", project_id="project-pg"
    ) is not None
    assert restarted.projections.graph(
        "project", "deploy", tenant_id="tenant-pg", project_id="project-pg"
    ) is not None
    assert restarted.catalog.get(
        "project:deploy", tenant_id="tenant-pg", project_id="project-pg"
    ) is not None
    assert restarted.reconcile() >= 1
    assert restarted.projections.get(
        "project", "deploy-crash", tenant_id="tenant-pg", project_id="project-pg"
    ) is not None

    removed = restarted.remove_provider("project")
    assert "project:deploy" in removed
    assert restarted.projections.get(
        "project", "deploy", tenant_id="tenant-pg", project_id="project-pg"
    ) is None
    assert restarted.catalog.get(
        "project:deploy", tenant_id="tenant-pg", project_id="project-pg"
    ) is None


def test_agent_projection_rust_pg_batch_cas_restart_reconcile_and_uninstall(
    pg_dsn, pg_schema
) -> None:
    """Native PostgreSQL CAS must preserve the same durable projection contract."""
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture unavailable")

    metadata = RustEnginePostgresMetaStore(dsn=pg_dsn, schema=pg_schema)
    metadata.ensure_initialized()
    cas_updates = [
        {
            "namespace": "native-cas",
            "key": "first",
            "payload": {"value": 1},
            "expected_last_authoritative_seq": None,
            "expected_last_materialized_seq": None,
            "last_authoritative_seq": 1,
            "last_materialized_seq": 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        },
        {
            "namespace": "native-cas",
            "key": "second",
            "payload": {"value": 2},
            "expected_last_authoritative_seq": 99,
            "expected_last_materialized_seq": 99,
            "last_authoritative_seq": 1,
            "last_materialized_seq": 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        },
    ]
    assert not metadata.compare_and_swap_named_projections(cas_updates)
    assert metadata.get_named_projection("native-cas", "first") is None
    assert metadata.get_named_projection("native-cas", "second") is None
    cas_updates[1]["expected_last_authoritative_seq"] = None
    cas_updates[1]["expected_last_materialized_seq"] = None
    assert metadata.compare_and_swap_named_projections(cas_updates)
    metadata.clear_projection_namespace("native-cas")

    def _create_once() -> bool:
        contender = RustEnginePostgresMetaStore(dsn=pg_dsn, schema=pg_schema)
        return contender.compare_and_swap_named_projection(
            "native-cas-race",
            "same-key",
            {"value": 1},
            expected_last_authoritative_seq=None,
            expected_last_materialized_seq=None,
            last_authoritative_seq=1,
            last_materialized_seq=1,
            projection_schema_version=1,
            materialization_status="ready",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: _create_once(), range(2))) == [False, True]
    metadata.clear_projection_namespace("native-cas-race")
    artifact = parse_skill_text(
        "# Native deploy\n\n1. validate\n",
        provider_id="project",
        provider_local_id="deploy-native",
    ).model_copy(
        update={
            "tenant_id": "tenant-rust-pg",
            "project_id": "project-rust-pg",
            "projection_revision": 1,
        }
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=DurableSkillProjectionStore(
            metadata,
            tenant_id="tenant-rust-pg",
            project_id="project-rust-pg",
            acl_enabled=False,
        ),
        catalog=DurableCatalogStore(
            metadata,
            tenant_id="tenant-rust-pg",
            project_id="project-rust-pg",
            acl_enabled=False,
        ),
    )
    materializer.materialize(artifact)
    crash_artifact = parse_skill_text(
        "# Native crash window\n\n1. validate\n",
        provider_id="project",
        provider_local_id="deploy-native-crash",
    ).model_copy(
        update={
            "tenant_id": "tenant-rust-pg",
            "project_id": "project-rust-pg",
            "projection_revision": 1,
        }
    )
    crash_projection = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
        acl_enabled=False,
    )
    _, crash_updates = crash_projection.prepare_upsert_update(crash_artifact)
    assert crash_updates
    assert metadata.compare_and_swap_named_projections(crash_updates[:1])

    # The materializer requires one shared metadata owner; rebuild with one
    # native facade to model a process restart without split authorities.
    shared = RustEnginePostgresMetaStore(dsn=pg_dsn, schema=pg_schema)
    restarted = DurableSkillCatalogMaterializer(
        projections=DurableSkillProjectionStore(
            shared,
            tenant_id="tenant-rust-pg",
            project_id="project-rust-pg",
            acl_enabled=False,
        ),
        catalog=DurableCatalogStore(
            shared,
            tenant_id="tenant-rust-pg",
            project_id="project-rust-pg",
            acl_enabled=False,
        ),
    )
    assert restarted.projections.get(
        "project", "deploy-native", tenant_id="tenant-rust-pg", project_id="project-rust-pg"
    ) is not None
    assert restarted.projections.graph(
        "project", "deploy-native", tenant_id="tenant-rust-pg", project_id="project-rust-pg"
    ) is not None
    restricted = DurableSkillProjectionStore(
        shared,
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
        acl_checker=lambda _artifact, principal: principal == "alice",
    )
    assert restricted.graph(
        "project",
        "deploy-native",
        principal="bob",
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
    ) is None
    assert restarted.catalog.get(
        "project:deploy-native",
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
    ) is not None
    assert restarted.reconcile() >= 1
    assert restarted.projections.get(
        "project",
        "deploy-native-crash",
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
    ) is not None

    removed = restarted.remove_provider("project")
    assert "project:deploy-native" in removed
    assert restarted.projections.get(
        "project", "deploy-native", tenant_id="tenant-rust-pg", project_id="project-rust-pg"
    ) is None
    assert restarted.catalog.get(
        "project:deploy-native",
        tenant_id="tenant-rust-pg",
        project_id="project-rust-pg",
    ) is None


def test_agent_projection_async_pg_uses_native_async_cas(
    async_sa_engine, async_pg_schema
) -> None:
    if async_sa_engine is None or async_pg_schema is None:
        pytest.skip("async PostgreSQL fixture unavailable")

    async def _run() -> None:
        metadata = AsyncPostgresNamedProjectionStore(
            async_sa_engine, schema=async_pg_schema
        )
        await metadata.ensure_initialized()
        async def _create_once():
            return await metadata.compare_and_swap_named_projection(
                "async-cas",
                "same-key",
                {"value": 1},
                expected_last_authoritative_seq=None,
                expected_last_materialized_seq=None,
                last_authoritative_seq=1,
                last_materialized_seq=1,
                projection_schema_version=1,
                materialization_status="ready",
            )

        assert sorted(await asyncio.gather(_create_once(), _create_once())) == [False, True]
        artifact = parse_skill_text(
            "# Deploy async\n\n1. validate\n",
            provider_id="project",
            provider_local_id="deploy-async",
        ).model_copy(
            update={
                "tenant_id": "tenant-pg",
                "project_id": "project-pg",
                "projection_revision": 1,
            }
        )
        materializer = AsyncDurableSkillCatalogMaterializer(
            metadata=metadata,
            tenant_id="tenant-pg",
            project_id="project-pg",
        )
        await materializer.materialize(artifact)
        assert await materializer.reconcile() == 1
        namespace = scoped_projection_namespace(
            "agent_skill_graph", tenant_id="tenant-pg", project_id="project-pg"
        )
        rows = await metadata.list_named_projections(namespace)
        assert any(row["payload"].get("record_kind") == "skill_current" for row in rows)
        await materializer.remove_provider("project")
        rows = await metadata.list_named_projections(namespace)
        current_rows = [
            row
            for row in rows
            if row["payload"].get("record_kind")
            in {"skill_current", "skill_graph_current", "skill_graph_node", "skill_graph_edge"}
        ]
        assert current_rows
        assert all(row["materialization_status"] == "retired" for row in current_rows)
        catalog_namespace = scoped_projection_namespace(
            "agent_catalog", tenant_id="tenant-pg", project_id="project-pg"
        )
        catalog_rows = await metadata.list_named_projections(catalog_namespace)
        catalog_current = [
            row
            for row in catalog_rows
            if row["payload"].get("record_kind") == "catalog_current"
        ]
        assert catalog_current
        assert all(row["materialization_status"] == "retired" for row in catalog_current)

    _run_async_windows_safe(_run())
