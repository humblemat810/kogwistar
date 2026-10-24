from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kogwistar.engine_core.conversation_contracts import (
    ConversationSummaryV1,
    ConversationTranscriptV1,
)

ROOT = Path(__file__).resolve().parents[2]


def test_conversation_summary_matches_shared_golden() -> None:
    payload = json.loads(
        (ROOT / "contracts" / "golden" / "conversation-wire-v1.json").read_text(
            encoding="utf-8"
        )
    )
    summary = ConversationSummaryV1.model_validate(payload)
    assert summary.model_dump(mode="json") == payload


def test_conversation_transcript_rejects_unknown_fields_and_unordered_turns() -> None:
    with pytest.raises(ValidationError):
        ConversationTranscriptV1.model_validate(
            {
                "schema_version": 1,
                "conversation_id": "conversation:demo",
                "turns": [
                    {
                        "schema_version": 1,
                        "node_id": "node:2",
                        "turn_index": 2,
                        "role": "assistant",
                        "content": "later",
                        "entity_type": "assistant_turn",
                    },
                    {
                        "schema_version": 1,
                        "node_id": "node:1",
                        "turn_index": 1,
                        "role": "user",
                        "content": "first",
                        "entity_type": "conversation_turn",
                    },
                ],
                "unknown": True,
            }
        )


def test_conversation_transcript_matches_shared_golden() -> None:
    payload = json.loads(
        (
            ROOT
            / "contracts"
            / "golden"
            / "conversation-transcript-wire-v1.json"
        ).read_text(encoding="utf-8")
    )
    transcript = ConversationTranscriptV1.model_validate(payload)
    assert transcript.model_dump(mode="json") == payload
