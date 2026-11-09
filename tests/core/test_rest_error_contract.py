from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from kogwistar.server_mcp_with_admin import app

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "golden"
    / "document-graph-validation-error-v1.json"
)


def test_python_document_graph_validation_http_error_matches_shared_fixture() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == fixture["python_detail"]
