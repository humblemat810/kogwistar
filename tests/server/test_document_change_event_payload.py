from __future__ import annotations

import json
from pathlib import Path

from kogwistar.engine_core.models import Document
from kogwistar.engine_core.subsystems.write import _entity_payload


def test_ocr_change_event_payload_matches_shared_python_model_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = json.loads(
        (root / "contracts/golden/document-ocr-change-event-payload-v1.json")
        .read_text(encoding="utf-8")
    )
    document = Document.from_ocr(
        id=fixture["input"]["doc_id"],
        ocr_content=json.loads(fixture["input"]["content"]),
        type="ocr",
    )
    document.metadata["insertion_method"] = fixture["input"]["insertion_method"]

    assert _entity_payload(document) == fixture["expected_payload"]
