"""Goal B read-only facade contracts; fake sources only."""

from __future__ import annotations

import pytest

from kogwistar.agent import (
    AgentReadTools,
    CatalogStore,
    ProviderRegistry,
    ReadScope,
    catalog_entries_from_artifact,
    parse_skill_text,
)
from kogwistar.server.capability_kernel import CapabilityKernel, CapabilitySpec


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.regression]


def _scope(**kwargs: object) -> ReadScope:
    values: dict[str, object] = {
        "principal_id": "alice",
        "tenant_id": "tenant-a",
        "project_id": "project-a",
    }
    values.update(kwargs)
    return ReadScope(**values)


def _tools() -> tuple[AgentReadTools, object]:
    artifact = parse_skill_text(
        "name: Deploy\n- inspect rollout\n",
        provider_id="project",
        provider_local_id="deploy",
        skill_version="v2",
    )
    catalog = CatalogStore(acl_enabled=False)
    for entry in catalog_entries_from_artifact(artifact):
        entry.tenant_id = "tenant-a"
        entry.project_id = "project-a"
        catalog.upsert(entry)

    class Provider:
        def get_skill(self, local_id: str) -> dict[str, str]:
            return {"local_id": local_id, "body": "provider-native source"}

    providers = ProviderRegistry()
    providers.register(provider_id="project", provider=Provider())
    kernel = CapabilityKernel()
    kernel.register(CapabilitySpec(name="skill.read", description="read skill"))
    tools = AgentReadTools(
        catalog=catalog,
        providers=providers,
        skill_artifacts={"project:deploy": artifact},
        mcp_descriptors={
            "mcp:project.deploy": {
                "name": "project.deploy",
                "tenant_id": "tenant-a",
                "project_id": "project-a",
                "input_schema": {"type": "object"},
            }
        },
        capability_kernel=kernel,
        execution_source={
            "items": [
                {"run_id": "r1", "tenant_id": "tenant-a", "project_id": "project-a"},
                {"run_id": "r2", "tenant_id": "tenant-a", "project_id": "project-a"},
            ],
            "list_run_events": [],
            "get_run": {
                "run_id": "r1",
                "tenant_id": "tenant-a",
                "project_id": "project-a",
            },
        },
        memory_source=lambda **kwargs: [
            {
                "memory_id": "m1",
                "conversation_id": kwargs["conversation_id"],
                "tenant_id": "tenant-a",
                "project_id": "project-a",
            }
        ],
        wisdom_source=lambda **kwargs: [
            {"id": "w1", "status": "approved", "tenant_id": "tenant-a"},
            {"id": "w2", "status": "pending", "tenant_id": "tenant-a"},
            {"id": "w3", "tenant_id": "tenant-a"},
        ],
        visibility_checker=lambda item, scope: (
            item.get("tenant_id") in (None, scope.tenant_id)
            and item.get("project_id") in (None, scope.project_id)
        ),
        max_limit=2,
    )
    return tools, artifact


def test_catalog_read_is_bounded_and_cursored() -> None:
    tools, _ = _tools()
    page = tools.catalog_browse(scope=_scope(), limit=1)
    assert len(page.items) == 1
    assert page.next_cursor is not None
    next_page = tools.catalog_browse(scope=_scope(), cursor=page.next_cursor, limit=1)
    assert next_page.items
    with pytest.raises(ValueError):
        tools.catalog_browse(scope=_scope(), limit=3)


def test_skill_descriptor_graph_and_raw_paths_are_additive() -> None:
    tools, _ = _tools()
    descriptor = tools.skill_get("project:deploy", scope=_scope())
    graph = tools.skill_get("project:deploy", scope=_scope(), representation="graph")
    raw = tools.skill_get("project:deploy", scope=_scope(), representation="raw")
    assert descriptor and "nodes" not in descriptor
    assert graph and graph["artifact"]["provider_id"] == "project"
    assert raw and raw["resource"]["body"] == "provider-native source"


def test_memory_defaults_to_current_conversation_and_requires_opt_in() -> None:
    tools, _ = _tools()
    assert tools.memory_search("deploy", scope=_scope(conversation_id="c1")).items
    with pytest.raises(PermissionError):
        tools.memory_search("deploy", scope=_scope(conversation_id="c1"), conversation_id="c2")
    allowed = _scope(conversation_id="c1", allow_cross_conversation=True)
    assert tools.memory_search("deploy", scope=allowed, conversation_id="c2").items


def test_wisdom_default_only_serves_approved_and_capability_is_descriptive() -> None:
    tools, _ = _tools()
    page = tools.wisdom_search("lesson", scope=_scope())
    assert [item["id"] for item in page.items] == ["w1"]
    with pytest.raises(PermissionError):
        tools.wisdom_search("lesson", scope=_scope(), status="pending")
    assert tools.capability_describe("skill.read", scope=_scope())["name"] == "skill.read"


def test_execution_read_filters_scope_and_does_not_mutate_source() -> None:
    tools, _ = _tools()
    source = tools.execution_source
    before = list(source["items"])
    page = tools.execution_read("items", scope=_scope(), limit=2)
    assert [item["run_id"] for item in page.items] == ["r1", "r2"]
    assert source["items"] == before


def test_explicit_run_read_methods_remain_bounded_and_read_only() -> None:
    tools, _ = _tools()
    page = tools.run_events("r1", scope=_scope(), limit=1)
    assert page.items == ()
    assert tools.run_status("r1", scope=_scope())["run_id"] == "r1"
    assert tools.skill_resource_get("project:deploy", "README.md", scope=_scope()) is None


def test_mcp_describe_is_discovery_only_and_scope_filtered() -> None:
    tools, _ = _tools()
    descriptor = tools.mcp_describe("mcp:project.deploy", scope=_scope())
    assert descriptor and descriptor["input_schema"]["type"] == "object"
    assert tools.mcp_describe("mcp:missing", scope=_scope()) is None
    assert tools.mcp_describe("mcp:project.deploy", scope=_scope(tenant_id="other")) is None


def test_rest_and_mcp_adapters_can_share_read_dispatch_contract() -> None:
    tools, _ = _tools()
    payload = {"query": "deploy", "limit": 1}
    rest_result = tools.dispatch("catalog.search", payload=payload, scope=_scope())
    mcp_result = tools.dispatch("catalog.search", payload=payload, scope=_scope())
    assert rest_result == mcp_result
    assert rest_result["items"][0]["logical_id"] == "project:deploy"
    with pytest.raises(ValueError, match="unsupported"):
        tools.dispatch("skill.invoke", payload={}, scope=_scope())
