from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kogwistar.engine_core.conversation_contracts import ToolAuditResponseV1

ROOT = Path(__file__).resolve().parents[2]


def test_tool_audit_matches_shared_golden() -> None:
    payload = json.loads(
        (ROOT / "contracts" / "golden" / "tool-audit-v1.json").read_text(
            encoding="utf-8"
        )
    )
    audit = ToolAuditResponseV1.model_validate(payload)
    assert audit.model_dump(mode="json") == payload


def test_tool_audit_rejects_unknown_fields_and_cross_conversation_items() -> None:
    with pytest.raises(ValidationError):
        ToolAuditResponseV1.model_validate(
            {"schema_version": 1, "conversation_id": None, "items": [], "extra": True}
        )
    with pytest.raises(ValidationError):
        ToolAuditResponseV1.model_validate(
            {
                "schema_version": 1,
                "conversation_id": "conversation:demo",
                "items": [
                    {
                        "schema_version": 1,
                        "node_id": "tool-call:other",
                        "entity_type": "tool_call",
                        "tool_name": "kg_neighbors",
                        "conversation_id": "conversation:other",
                        "turn_index": 1,
                        "summary": "wrong scope",
                        "properties": {},
                    }
                ],
            }
        )
