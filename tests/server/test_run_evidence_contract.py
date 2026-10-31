from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kogwistar.server.chat_service import ChatRunService


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "contracts" / "golden" / "run-evidence-v1.json"


def test_python_run_evidence_matches_shared_v1_contract() -> None:
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    class ServiceStub:
        def _require_capability(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {}

        def get_run(self, run_id: str) -> dict[str, Any]:
            assert run_id == expected["run_id"]
            return {"conversation_id": expected["conversation_id"]}

        def latest_snapshot(
            self, conversation_id: str, *, run_id: str
        ) -> dict[str, Any]:
            assert conversation_id == expected["conversation_id"]
            assert run_id == expected["run_id"]
            return {
                "snapshot_node_id": expected["snapshot_node_id"],
                "metadata": expected["snapshot_metadata"],
                "properties": {
                    "evidence_pack_digest": json.dumps(
                        expected["source_summary"]["evidence_pack_digest"]
                    )
                },
            }

    assert ChatRunService.run_evidence(ServiceStub(), expected["run_id"]) == expected
