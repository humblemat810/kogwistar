from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from kogwistar.agent.event_journal import (
    DurableProviderEvent,
    DurableProviderEventPublication,
    _event_identity,
)
from kogwistar.agent.streaming import ProviderEvent

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "golden"
    / "provider-events-v1.json"
)
_EVENT_ADAPTER = TypeAdapter(ProviderEvent)
_REQUEST_FIXTURE = _FIXTURE.with_name("provider-requests-v1.json")


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


def test_python_provider_requests_match_shared_rust_golden_fixture() -> None:
    from kogwistar.agent.streaming import ProviderRequest

    fixture = json.loads(_REQUEST_FIXTURE.read_text(encoding="utf-8"))
    assert len(fixture["requests"]) == 3
    for expected in fixture["requests"]:
        request = ProviderRequest.model_validate(expected)
        assert request.model_dump(mode="json") == expected
    for invalid in fixture["invalid_requests"]:
        with pytest.raises(ValidationError):
            ProviderRequest.model_validate(invalid)


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
    envelope = DurableProviderEvent.model_validate(fixture["envelope"])
    assert f"agent-event:{_event_identity(envelope.trace, envelope.event)}" == fixture[
        "computed_event_id"
    ]
    sequence_event = _EVENT_ADAPTER.validate_python(
        fixture["envelope"]["event"] | {"provider_event_id": None}
    )
    assert (
        f"agent-event:{_event_identity(envelope.trace, sequence_event)}"
        == fixture["computed_sequence_event_id"]
    )
