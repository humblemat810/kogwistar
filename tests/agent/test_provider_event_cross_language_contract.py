from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from kogwistar.agent.event_journal import (
    DurableProviderEvent,
    DurableProviderEventPublication,
)
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


def test_python_durable_provider_event_matches_shared_rust_golden_fixture() -> None:
    fixture_path = _FIXTURE.with_name("agent-durable-provider-event-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    envelope = DurableProviderEvent.model_validate(fixture["envelope"])
    assert envelope.model_dump(mode="json") == fixture["envelope"]
    for invalid in fixture["invalid_envelopes"]:
        with pytest.raises(ValidationError):
            DurableProviderEvent.model_validate(invalid)
    publication = DurableProviderEventPublication.model_validate(fixture["publication"])
    assert publication.model_dump(mode="json") == fixture["publication"]
