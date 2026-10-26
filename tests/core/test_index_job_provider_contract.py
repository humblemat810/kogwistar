from __future__ import annotations

import json
from pathlib import Path

from kogwistar.engine_core.index_job_provider import embedding_job_provider_input

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = json.loads(
    (_ROOT / "contracts/golden/index-job-provider-input-v1.json").read_text(
        encoding="utf-8"
    )
)


def test_embedding_provider_input_matches_shared_v1_fixture() -> None:
    queue_row = {
        **_FIXTURE,
        "claim_token": "must-not-cross-provider-boundary",
        "lease_until": 123,
        "retry_count": 2,
        "accepted_result_json": "must-not-cross-provider-boundary",
    }

    result = embedding_job_provider_input(queue_row)

    assert result == _FIXTURE
    assert "claim_token" not in result
    assert "lease_until" not in result
    assert "retry_count" not in result
    assert "accepted_result_json" not in result
