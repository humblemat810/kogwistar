from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _readiness():
    path = ROOT / "scripts" / "rust_port_readiness.py"
    spec = importlib.util.spec_from_file_location("rust_port_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rehearsal_is_valid_but_cannot_promote_ownership() -> None:
    payload = json.loads(
        (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
            encoding="utf-8"
        )
    )

    result = _readiness().validate_canary_evidence(payload)

    assert result["valid"] is True
    assert result["production_complete"] is False
    assert result["ownership_promotion_allowed"] is False


def test_canary_validation_fails_closed_on_missing_stage_or_mismatch() -> None:
    payload = json.loads(
        (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["evidence_kind"] = "production"
    payload["stages"][2]["unexplained_correctness_mismatches"] = 1
    payload["stages"].pop()

    result = _readiness().validate_canary_evidence(payload)

    assert result["valid"] is False
    assert result["ownership_promotion_allowed"] is False
    assert any("stages must cover" in error for error in result["errors"])
    assert any("unexplained_correctness_mismatches" in error for error in result["errors"])


def test_complete_production_evidence_is_eligible_for_external_promotion() -> None:
    payload = json.loads(
        (ROOT / "contracts" / "canary" / "adr015-rehearsal-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["evidence_kind"] = "production"
    payload["candidate_identity_sha256"] = "a" * 64

    result = _readiness().validate_canary_evidence(payload)

    assert result["valid"] is True
    assert result["production_complete"] is True
    assert result["ownership_promotion_allowed"] is True
