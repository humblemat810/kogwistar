from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from kogwistar.agent.streaming import ProviderEvent

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "golden"
    / "provider-events-v1.json"
)
_EVENT_ADAPTER = TypeAdapter(ProviderEvent)


def test_python_provider_events_match_shared_rust_golden_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert len(fixture["events"]) == 8
    for expected in fixture["events"]:
        event = _EVENT_ADAPTER.validate_python(expected)
        assert event.model_dump(mode="json") == expected


def test_python_provider_events_reject_the_shared_invalid_cases() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for invalid in fixture["invalid_events"]:
        with pytest.raises(ValidationError):
            _EVENT_ADAPTER.validate_python(invalid)
