from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kogwistar.runtime import WorkerIdentityAttestationV1

pytestmark = [pytest.mark.ci, pytest.mark.runtime]

def test_worker_identity_attestation_matches_shared_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "worker-identity-attestation-v1.json"
        ).read_text(encoding="utf-8")
    )
    valid = fixture["valid"]
    model = WorkerIdentityAttestationV1.model_validate(valid)

    assert model.to_wire() == valid
    for invalid in fixture["invalid"]:
        with pytest.raises(ValidationError):
            WorkerIdentityAttestationV1.model_validate(invalid)
