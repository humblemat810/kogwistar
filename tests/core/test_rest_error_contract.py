from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from kogwistar.server_mcp_with_admin import DocumentGraphProposal

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "golden"
    / "document-graph-validation-error-v1.json"
)


def test_document_graph_validation_required_field_error_matches_shared_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))

    try:
        DocumentGraphProposal.model_validate(fixture["request"])
    except ValidationError as error:
        details = error.errors(include_url=False)
    else:
        raise AssertionError("fixture request must omit required nodes")

    python_detail = [
        {**detail, "loc": ["body", *detail["loc"]]}
        for detail in details
    ]
    assert python_detail == fixture["python_detail"]
