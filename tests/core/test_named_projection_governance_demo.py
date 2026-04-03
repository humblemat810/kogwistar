from __future__ import annotations

import pytest

from kogwistar.demo.named_projection_governance_demo import (
    run_named_projection_governance_demo,
)
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.core]


@pytest.mark.ci
def test_named_projection_governance_demo_uses_generic_projection_substrate(tmp_path):
    result = run_named_projection_governance_demo(
        data_dir=tmp_path / "named_projection_demo",
        backend_factory=build_fake_backend,
    )

    assert result["summary"]["history_namespace"] == "bridge_governance_history"
    assert result["summary"]["projection_namespace"] == "bridge_governance"
    assert result["summary"]["history_event_count"] == 5
    assert result["summary"]["projection_keys_before_clear"] == [
        "interaction-alpha",
        "interaction-beta",
    ]
    assert result["summary"]["rebuilt_matches_before_clear"] is True
    assert result["summary"]["namespace_clear_removed_all"] is True

    alpha = result["details"]["interaction_alpha"]["projection"]
    beta = result["details"]["interaction_beta"]["projection"]

    assert alpha["materialization_status"] == "ready"
    assert alpha["projection_schema_version"] == 1
    assert alpha["last_authoritative_seq"] == 5
    assert alpha["last_materialized_seq"] == 3
    assert alpha["payload"]["latest_status"] == "approved"
    assert alpha["payload"]["latest_decision"] == "approved"
    assert alpha["payload"]["participants"] == ["governor", "policy", "router"]
    assert alpha["payload"]["policy_versions"] == ["v1", "v2"]
    assert result["details"]["interaction_alpha"]["status_transitions"] == [
        "rebuilding",
        "ready",
    ]
    assert (
        result["details"]["interaction_alpha"]["rebuilt_projection"]["payload"]
        == alpha["payload"]
    )

    assert beta["materialization_status"] == "ready"
    assert beta["last_authoritative_seq"] == 5
    assert beta["last_materialized_seq"] == 5
    assert beta["payload"]["latest_status"] == "rejected"
    assert beta["payload"]["latest_decision"] == "rejected"
    assert beta["payload"]["participants"] == ["auditor", "router"]
    assert result["details"]["interaction_beta"]["status_transitions"] == [
        "rebuilding",
        "ready",
    ]

    assert result["details"]["projections_after_namespace_clear"] == []
