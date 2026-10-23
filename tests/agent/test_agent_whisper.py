from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from kogwistar.agent import (
    AgentProfile,
    AgentReadTools,
    CatalogEntry,
    CatalogStore,
    ReadScope,
    SequenceFakeModel,
    WhisperConfig,
    build_whisper_context,
    register_model_step,
)
from kogwistar.runtime import MappingStepResolver, StepContext
from kogwistar.runtime.base_runtime import apply_state_update_inplace

pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.workflow]


def _scope(**overrides: object) -> ReadScope:
    values: dict[str, object] = {
        "principal_id": "alice",
        "security_scope": "project-a",
        "tenant_id": "tenant-a",
        "project_id": "project-a",
    }
    values.update(overrides)
    return ReadScope(**values)


def _ctx(state: dict[str, object], *, authority: Mapping[str, object] | None = None) -> StepContext:
    return StepContext(
        run_id="run-whisper",
        workflow_id="whisper-test",
        workflow_node_id="model",
        op="agent.model_call",
        token_id="token-whisper",
        attempt=1,
        step_seq=1,
        cache_dir=None,
        state=state,
        authority_context=authority or {},
    )


def _entry(
    logical_id: str,
    *,
    name: str,
    kind: str,
    required_capabilities: list[str],
    tenant_id: str = "tenant-a",
) -> CatalogEntry:
    return CatalogEntry(
        logical_id=logical_id,
        provider_id="tools",
        provider_local_id=logical_id,
        kind=kind,
        name=name,
        summary=f"Use {name} to inspect deployment status",
        source_fingerprint=f"fp-{logical_id}",
        required_capabilities=required_capabilities,
        scope="project-a",
        tenant_id=tenant_id,
        project_id="project-a",
    )


def _read_tools(*, knowledge_source: object | None = None) -> AgentReadTools:
    catalog = CatalogStore(
        acl_enabled=True,
        acl_checker=lambda entry, principal: (
            principal == "alice" and entry.logical_id != "tool:hidden"
        ),
    )
    catalog.upsert(
        _entry("tool:status", name="deployment status", kind="tool", required_capabilities=["deploy.read"])
    )
    catalog.upsert(
        _entry("tool:write", name="deployment writer", kind="mcp", required_capabilities=["deploy.write"])
    )
    catalog.upsert(
        _entry("tool:hidden", name="hidden deployment status", kind="tool", required_capabilities=["deploy.read"])
    )
    catalog.upsert(
        _entry("tool:unscoped", name="other deployment status", kind="tool", required_capabilities=["deploy.read"], tenant_id="tenant-b")
    )
    return AgentReadTools(
        catalog=catalog,
        knowledge_source=knowledge_source,
        visibility_checker=lambda item, scope: (
            item.get("tenant_id") == scope.tenant_id
            and item.get("project_id") == scope.project_id
        ),
    )


def test_whisper_is_disabled_by_default_and_call_config_overrides_profile() -> None:
    profile = AgentProfile(
        agent_id="alice",
        whisper=WhisperConfig(enabled=True),
    )
    model = SequenceFakeModel([{"answer": "ok"}])
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        model,
        profile=profile,
        whisper_config=WhisperConfig(enabled=False),
        whisper_read_tools=_read_tools(),
        whisper_scope=_scope(),
    )
    result = resolver.resolve("agent.model_call")(_ctx({"agent_prompt": "deployment status"}))
    assert result.status == "success"
    assert "whisper" not in model.contexts[0]
    assert "agent_whisper_context" not in result.state_update[0][1]


def test_whisper_is_absent_from_model_context_by_default() -> None:
    model = SequenceFakeModel([{"answer": "ok"}])
    resolver = MappingStepResolver()
    register_model_step(resolver, model)
    result = resolver.resolve("agent.model_call")(_ctx({"agent_prompt": "deployment status"}))
    assert result.status == "success"
    assert "whisper" not in model.contexts[0]


def test_model_step_whisper_config_can_enable_profile_default() -> None:
    model = SequenceFakeModel([{"answer": "ok"}])
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        model,
        profile=AgentProfile(agent_id="alice"),
        whisper_config=WhisperConfig(enabled=True, knowledge_depth=0, knowledge_limit=0),
        whisper_read_tools=_read_tools(),
        whisper_scope=_scope(),
    )
    result = resolver.resolve("agent.model_call")(_ctx({"agent_prompt": "deployment status"}))
    assert result.status == "success"
    assert model.contexts[0]["whisper"]["status"] == "empty"


def test_tool_hints_use_bm25_are_capped_at_three_and_never_invoke() -> None:
    rank_calls: list[str] = []
    catalog = CatalogStore(
        acl_enabled=False,
        semantic_ranker=lambda query, _entries: (rank_calls.append(query) or {}),
    )
    for index in range(5):
        catalog.upsert(
            _entry(
                f"tool:deploy:{index}",
                name=f"deployment status {index}",
                kind="tool",
                required_capabilities=["deploy.read"],
            )
        )
    tools = AgentReadTools(catalog=catalog, acl_required=False)
    result = build_whisper_context(
        query="deployment status",
        config=WhisperConfig(enabled=True, tool_limit=3, knowledge_limit=0),
        read_tools=tools,
        scope=_scope(),
        effective_capabilities=("deploy.read",),
    )
    assert len(result.tool_hints) == 3
    assert rank_calls == []


def test_profile_whisper_runs_before_each_call_and_records_same_result() -> None:
    model = SequenceFakeModel([{"answer": "one"}, {"answer": "two"}])
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        model,
        profile=AgentProfile(
            agent_id="alice", whisper=WhisperConfig(enabled=True, knowledge_depth=0)
        ),
        whisper_read_tools=_read_tools(
            knowledge_source={
                "search": [
                    {
                        "entity_id": "knowledge:deploy",
                        "title": "Deployments",
                        "summary": "Current deployment guidance",
                        "source_ref": "doc:deploy#rev-2",
                        "security_scope": "project-a",
                        "tenant_id": "tenant-a",
                        "project_id": "project-a",
                    }
                ]
            }
        ),
        whisper_scope=_scope(),
    )
    handler = resolver.resolve("agent.model_call")
    first_state: dict[str, object] = {
        "agent_prompt": "deployment status",
        "agent_budget_hints": {"max_model_calls": 2, "remaining_model_calls": 1},
    }
    second_state: dict[str, object] = {
        "agent_prompt": "deployment status",
        "agent_budget_hints": {"max_model_calls": 2, "remaining_model_calls": 1},
    }
    for state in (first_state, second_state):
        result = handler(_ctx(state, authority={"effective_capabilities": ("deploy.read",)}))
        assert result.status == "success"
        apply_state_update_inplace(state, result.state_update)
        assert state["agent_whisper_context"] == model.contexts[len(model.contexts) - 1]["whisper"]
    assert len(model.contexts) == 2
    assert model.contexts[0]["budget"] == {
        "max_model_calls": 2,
        "remaining_model_calls": 1,
    }
    whisper = model.contexts[0]["whisper"]
    assert whisper["knowledge"][0]["source_refs"] == ["doc:deploy#rev-2"]
    assert [hint["id"] for hint in whisper["tool_hints"]] == ["tool:status"]
    assert "deploy.write" not in json.dumps(whisper)
    assert "tool:hidden" not in json.dumps(whisper)
    assert "tool:unscoped" not in json.dumps(whisper)


def test_whisper_one_hop_preserves_order_deduplicates_and_caps_at_ten() -> None:
    class Source:
        def search(self, **_kwargs: object) -> list[Mapping[str, object]]:
            return [
                {"entity_id": f"seed:{index}", "title": f"Seed {index}", "security_scope": "project-a", "tenant_id": "tenant-a", "project_id": "project-a"}
                for index in range(3)
            ]

        def expand(self, *, entity_id: str, limit: int, **_kwargs: object) -> list[Mapping[str, object]]:
            base = int(entity_id.rsplit(":", 1)[1])
            values = [
                {"entity_id": entity_id, "title": "duplicate seed", "security_scope": "project-a", "tenant_id": "tenant-a", "project_id": "project-a"},
                *[
                    {"entity_id": f"neighbor:{base}:{index}", "title": "Neighbor", "security_scope": "project-a", "tenant_id": "tenant-a", "project_id": "project-a"}
                    for index in range(limit)
                ],
            ]
            return values[:limit]

    result = build_whisper_context(
        query="deployment",
        config=WhisperConfig(enabled=True, knowledge_depth=1, knowledge_limit=10),
        read_tools=_read_tools(knowledge_source=Source()),
        scope=_scope(),
    )
    ids = [item.id for item in result.knowledge]
    assert ids[:3] == ["seed:0", "seed:1", "seed:2"]
    assert len(ids) == len(set(ids)) <= 10


def test_whisper_payload_is_bounded_and_marks_truncation() -> None:
    source = {
        "search": [
            {
                "entity_id": f"entity:{index}",
                "title": "A" * 100,
                "summary": "B" * 320,
                "source_ref": "source:" + ("x" * 180),
                "security_scope": "project-a",
                "tenant_id": "tenant-a",
                "project_id": "project-a",
            }
            for index in range(10)
        ]
    }
    config = WhisperConfig(enabled=True, knowledge_depth=0, payload_max_chars=700)
    result = build_whisper_context(
        query="deployment",
        config=config,
        read_tools=_read_tools(knowledge_source=source),
        scope=_scope(),
    )
    encoded = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    assert len(encoded) <= config.payload_max_chars
    assert result.truncated


def test_whisper_failure_is_bounded_and_model_call_fails_open() -> None:
    class BrokenSource:
        def search(self, **_kwargs: object) -> list[Mapping[str, object]]:
            raise RuntimeError("secret backend path and credentials")

    model = SequenceFakeModel([{"answer": "continued"}])
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        model,
        whisper_config=WhisperConfig(enabled=True),
        whisper_read_tools=_read_tools(knowledge_source=BrokenSource()),
        whisper_scope=_scope(),
    )
    state: dict[str, object] = {"agent_prompt": "deployment status"}
    result = resolver.resolve("agent.model_call")(_ctx(state))
    assert result.status == "success"
    apply_state_update_inplace(state, result.state_update)
    assert model.contexts[0]["whisper"] == {
        "status": "degraded",
        "tool_hints": [],
        "knowledge": [],
        "truncated": False,
        "degraded_reason": "retrieval_unavailable",
    }
    assert "secret backend" not in json.dumps(model.contexts[0])


def test_whisper_scope_is_derived_from_trusted_authority_not_workflow_state() -> None:
    model = SequenceFakeModel([{"answer": "ok"}])
    resolver = MappingStepResolver()
    register_model_step(
        resolver,
        model,
        whisper_config=WhisperConfig(enabled=True),
        whisper_read_tools=_read_tools(),
    )
    state: dict[str, object] = {
        "agent_prompt": "deployment status",
        "tenant_id": "tenant-a",
        "principal_id": "alice",
    }
    result = resolver.resolve("agent.model_call")(_ctx(state, authority={"principal_id": "alice"}))
    assert result.status == "success"
    assert model.contexts[0]["whisper"]["tool_hints"] == []
    assert result.state_update[0][1]["agent_whisper_context"]["degraded_reason"] == "read_scope_unavailable"
