from __future__ import annotations

import json
from pathlib import Path

import pytest

from kogwistar.conversation.agentic_answering import AnswerWithCitations

pytestmark = pytest.mark.ci


def test_answer_provider_extra_fields_match_shared_rust_contract() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts/golden/answer-provider-extra-fields-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["contract_version"] == 1
    answer = AnswerWithCitations.model_validate(fixture["input"])
    assert answer.model_dump(mode="json") == fixture["expected"]
