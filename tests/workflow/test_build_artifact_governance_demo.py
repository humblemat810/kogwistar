from __future__ import annotations

import pytest

from kogwistar.demo import run_build_artifact_governance_demo
from tests._helpers.fake_backend import build_fake_backend

pytestmark = [pytest.mark.workflow]


def _assert_governance_demo_result(result: dict) -> None:
    assert result["summary"]["safe_status"] == "succeeded"
    assert result["summary"]["unsafe_status"] == "failure"
    assert result["summary"]["invariant_pass"] is True
    assert result["summary"]["workflow_graphs_visible"] == {
        "safe": True,
        "unsafe": True,
    }
    assert (
        result["details"]["safe"]["public_projection_strategy"]
        == "BuildArtifact['public']"
    )
    assert result["details"]["safe"]["public_projection_matches_public_artifact"] is True

    safe_final = result["details"]["safe"]["final_state"]
    unsafe_final = result["details"]["unsafe"]["final_state"]

    assert safe_final["validation_passed"] is True
    assert safe_final["published_artifact"]["mode"] == "public"
    assert "source_maps" not in safe_final["published_artifact"]
    assert "raw_sources" not in safe_final["published_artifact"]
    assert "source_root" not in safe_final["published_artifact"]["metadata"]
    assert "source_map_manifest" not in safe_final["published_artifact"]["metadata"]
    assert "internal_notes" not in safe_final["published_artifact"]["metadata"]

    assert unsafe_final["validation_passed"] is False
    assert unsafe_final["final_status"] == "artifact_rejected"
    assert not result["details"]["unsafe"]["published_artifact"]
    assert any(
        "source_maps" in error or "source_root" in error
        for error in unsafe_final["validation_errors"]
    )

    assert result["details"]["safe"]["event_types"] == [
        "artifact_built",
        "artifact_filtered",
        "artifact_validated",
        "artifact_published",
    ]
    assert result["details"]["unsafe"]["event_types"] == [
        "artifact_built",
        "artifact_rejected",
    ]
    assert "apply_public_mode" in result["details"]["safe"]["step_ops"]
    assert "apply_public_mode" not in result["details"]["unsafe"]["step_ops"]
    assert result["details"]["safe"]["replay_state"]["published_artifact"]["mode"] == (
        "public"
    )
    assert result["details"]["unsafe"]["replay_state"]["validation_passed"] is False
    assert result["details"]["safe"]["public_artifact_violations"] == []


@pytest.mark.ci
def test_build_artifact_governance_demo_prevents_source_map_leakage(tmp_path):
    result = run_build_artifact_governance_demo(
        data_dir=tmp_path / "artifact_governance_demo",
        backend_factory=build_fake_backend,
    )
    _assert_governance_demo_result(result)


@pytest.mark.ci_full
@pytest.mark.parametrize("backend_kind", ["chroma", "pg"], indirect=True)
def test_build_artifact_governance_demo_persistent_backends(
    workflow_engine,
    conversation_engine,
    backend_kind,
):
    _ = backend_kind
    result = run_build_artifact_governance_demo(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        reset_data=False,
    )
    _assert_governance_demo_result(result)
