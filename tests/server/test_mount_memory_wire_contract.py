from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kogwistar.server.chat_service_conversation_queries import _ConversationQueryService


class _SnapshotService:
    def latest_context_snapshot_node(
        self, *, conversation_id: str, run_id: str | None, stage: str | None
    ) -> SimpleNamespace:
        assert (conversation_id, run_id, stage) == ("conversation-1", "run-1", "answer")
        return SimpleNamespace(id="snapshot-1")

    def get_context_snapshot_payload(self, *, snapshot_node_id: str) -> dict[str, Any]:
        assert snapshot_node_id == "snapshot-1"
        return {
            "metadata": {"stage": "answer", "run_id": "run-1"},
            "properties": {"summary": "checkpoint payload", "step_seq": 12},
        }


def test_python_mount_memory_result_matches_shared_wire_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/mount-memory-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    owner = SimpleNamespace(_conversation_service=lambda: _SnapshotService())
    queries = _ConversationQueryService(cast(Any, owner))

    result = queries.latest_snapshot(
        "conversation-1", run_id="run-1", stage="answer"
    )

    assert result == fixture
