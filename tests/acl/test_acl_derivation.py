from __future__ import annotations

import pytest

from kogwistar.acl import ACLInput, ACLGraph, derive_acl, join_acl_inputs
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.fake_backend import build_fake_backend


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.regression]


def test_strict_join_is_high_water_and_keeps_source_audit() -> None:
    result = derive_acl(
        [
            ACLInput("user-query", "private", owner_id="user-a", source_kind="query"),
            ACLInput("public-doc", "public", source_kind="knowledge"),
        ]
    )

    assert result.original_acl.mode == "private"
    assert result.final_mode == "private"
    assert result.policy == "STRICT"
    assert result.declassification_result == "not_requested"
    assert result.source_ids == ("user-query", "public-doc")
    assert result.to_dict()["visible_context_acl_join"] == "private"


def test_mixed_private_owners_clear_owner_privilege() -> None:
    joined = join_acl_inputs(
        [
            ACLInput("a", "private", owner_id="user-a"),
            ACLInput("b", "private", owner_id="user-b"),
        ]
    )

    assert joined.mode == "private"
    assert joined.owner_id is None


def test_clean_room_is_explicit_and_does_not_override_seen_private_input() -> None:
    assert join_acl_inputs([], clean_room=True).mode == "public"
    assert join_acl_inputs([ACLInput("private", "private")], clean_room=True).mode == "private"


def test_llm_guarded_is_post_join_declassification_only() -> None:
    with pytest.warns(RuntimeWarning, match="opt-in declassification"):
        result = derive_acl(
            [ACLInput("public-doc", "public")],
            policy="LLM_GUARDED",
            acknowledge_llm_guarded=True,
            declassifier=lambda _audit: {
                "approved": True,
                "proposed_acl": "public",
                "confidence": 0.99,
                "classifier_model": "guard-model",
                "classifier_version": "v1",
                "reason": "clean-room public derivation",
                "evidence": ["policy-check-1"],
            },
        )

    assert result.original_acl.mode == "public"
    assert result.final_mode == "public"
    assert result.declassification_result == "approved"
    assert result.to_dict()["classifier_model"] == "guard-model"


def test_guard_cannot_make_output_more_restrictive_than_strict_join() -> None:
    with pytest.warns(RuntimeWarning):
        result = derive_acl(
            [ACLInput("public-source", "public")],
            policy="LLM_GUARDED",
            acknowledge_llm_guarded=True,
            declassifier=lambda _audit: {"approved": True, "proposed_acl": "private"},
        )

    assert result.original_acl.mode == "public"
    assert result.final_mode == "public"
    assert result.declassification_result == "rejected"


def test_guard_failure_and_malformed_response_keep_original_taint() -> None:
    with pytest.warns(RuntimeWarning):
        failed = derive_acl(
            [ACLInput("private", "private")],
            policy="LLM_GUARDED",
            acknowledge_llm_guarded=True,
            declassifier=lambda _audit: (_ for _ in ()).throw(RuntimeError("offline")),
        )
    assert failed.final_mode == "private"
    assert failed.declassification_result == "failed"

    with pytest.warns(RuntimeWarning):
        malformed = derive_acl(
            [ACLInput("private", "private")],
            policy="LLM_GUARDED",
            acknowledge_llm_guarded=True,
            declassifier=lambda _audit: {"approved": True, "proposed_acl": "bogus"},
        )
    assert malformed.final_mode == "private"
    assert malformed.declassification_result == "malformed"


def test_guarded_policy_requires_explicit_acknowledgement() -> None:
    with pytest.raises(PermissionError, match="requires explicit acknowledgement"):
        derive_acl([ACLInput("x", "private")], policy="LLM_GUARDED")


def test_derivation_audit_round_trips_through_acl_truth() -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=None,
        backend_factory=build_fake_backend,
        kg_graph_type="knowledge",
    )
    audit = derive_acl(
        [ACLInput("private-source", "private", owner_id="user-a")],
        object_id="answer-1",
        generation_id="run-1",
    ).to_dict()
    engine.record_acl(
        grain="artifact",
        truth_graph="knowledge",
        entity_id="answer-1",
        version=1,
        mode="private",
        owner_id="user-a",
        source_ids=["private-source"],
        derivation_type="model_output",
        derivation_audit=audit,
    )
    record = engine.acl_graph.latest_record(
        grain="artifact", truth_graph="knowledge", entity_id="answer-1"
    )
    assert record is not None
    assert record.derivation_audit["generation_id"] == "run-1"
    engine.acl_graph = ACLGraph()
    engine.rebuild_acl_graph_from_truth()
    rebuilt = engine.acl_graph.latest_record(
        grain="artifact", truth_graph="knowledge", entity_id="answer-1"
    )
    assert rebuilt is not None
    assert rebuilt.derivation_audit["final_acl"] == "private"
