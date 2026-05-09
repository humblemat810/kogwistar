from __future__ import annotations

from types import SimpleNamespace

from kogwistar.policy import (
    DefaultArtifactVisibilityPolicy,
    DefaultDerivedKnowledgePolicy,
    DefaultKnowledgeLifecyclePolicy,
    DefaultPromotionPolicy,
    DefaultProjectionEligibilityPolicy,
    DefaultWisdomPolicy,
    PromotionContext,
    SourceQueryDecision,
)


def test_default_promotion_policy_requires_sync_and_threshold():
    policy = DefaultPromotionPolicy()

    pending = policy.decide(
        PromotionContext(
            promotion_mode="pending",
            auto_accept_threshold=0.2,
        )
    )
    assert not pending.should_promote

    promoted = policy.decide(
        PromotionContext(
            promotion_mode="sync",
            auto_accept_threshold=0.9,
        )
    )
    assert promoted.should_promote

    blocked = policy.decide(
        PromotionContext(
            promotion_mode="sync",
            auto_accept_threshold=0.99,
        )
    )
    assert not blocked.should_promote


def test_default_visibility_and_projection_policy_are_conservative():
    visibility = DefaultArtifactVisibilityPolicy()
    projection = DefaultProjectionEligibilityPolicy()

    assert visibility.visibility_for({"artifact_kind": "candidate_link"}) == "internal"
    assert visibility.visibility_for({"artifact_kind": "promoted_knowledge"}) == "internal"
    assert visibility.visibility_for({"artifact_kind": "execution_wisdom"}) == "internal"
    assert visibility.visibility_for({"artifact_kind": "conversation_note"}) == "internal"
    assert visibility.visibility_for({"knowledge_stage": "review"}) == "review"
    assert visibility.visibility_for({"knowledge_stage": "promoted"}) == "knowledge"
    assert visibility.visibility_for({"knowledge_stage": "derived"}) == "wisdom"
    assert not projection.is_projection_eligible({"visibility": "internal"})
    assert projection.is_projection_eligible({"visibility": "projection"})
    assert projection.is_projection_eligible({"projection_visible": True})


def test_default_derived_knowledge_policy_keeps_grouping_stable():
    policy = DefaultDerivedKnowledgePolicy()
    node = SimpleNamespace(label="Shared Entity", metadata={"label": "Shared Entity"})

    assert policy.group_key(node) == "Shared Entity"
    assert policy.source_query(workspace_id="w1") == SourceQueryDecision(
        where={"workspace_id": "w1"}
    )
    assert policy.match_where(workspace_id="w1", label="Shared Entity") == {
        "workspace_id": "w1",
        "label": "Shared Entity",
    }
    metadata = policy.build_metadata(
        workspace_id="w1",
        label="Shared Entity",
        source_node_ids=["a", "b"],
        replaces_ids=["old"],
        created_at_ms=123,
        artifact_kind="caller_defined_synthesis",
    )
    assert metadata["artifact_kind"] == "caller_defined_synthesis"
    assert metadata["source_node_ids"] == ["a", "b"]
    assert metadata["replaces_ids"] == ["old"]


def test_default_wisdom_policy_uses_configurable_failure_threshold():
    policy = DefaultWisdomPolicy(min_failure_signals=3)
    assert policy.min_failure_signals == 3
    assert policy.source_query(workspace_id="w1") == SourceQueryDecision(
        where={"workspace_id": "w1"}
    )
    assert policy.match_where(workspace_id="w1", step_op="distill") == {
        "workspace_id": "w1",
        "step_op": "distill",
    }
    metadata = policy.build_metadata(
        workspace_id="w1",
        step_op="distill",
        failure_count=3,
        evidence_run_ids=["r1", "r2"],
        replaces_ids=["old"],
        created_at_ms=123,
        artifact_kind="caller_defined_wisdom",
    )
    assert metadata["artifact_kind"] == "caller_defined_wisdom"
    assert metadata["failure_count"] == 3
    assert metadata["evidence_run_ids"] == ["r1", "r2"]


def test_default_lifecycle_policy_requires_provenance_for_durable_artifacts():
    policy = DefaultKnowledgeLifecyclePolicy()

    assert not policy.requires_provenance("review")
    assert policy.requires_provenance("derived")
    assert policy.requires_provenance("wisdom")
    assert not policy.requires_provenance("derived_knowledge")
    assert not policy.requires_provenance("execution_wisdom")
    assert policy.replacement_ids([SimpleNamespace(id="n1"), "n2"]) == ["n1", "n2"]
