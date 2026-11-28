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


def test_python_document_graph_validation_wrong_type_matches_shared_fixture() -> None:
    fixture_path = _FIXTURE.with_name("document-graph-validation-wrong-type-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == fixture["python_detail"]


def test_python_document_graph_validation_ignores_extra_top_level_field() -> None:
    fixture_path = _FIXTURE.with_name("document-graph-validation-extra-field-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == 200
    assert response.json() == fixture["response"]


def test_python_document_graph_validation_rejects_null_nodes() -> None:
    fixture_path = _FIXTURE.with_name("document-graph-validation-null-nodes-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == 422
    assert response.json()["detail"] == fixture["python_detail"]


def test_python_document_graph_validation_rejects_non_object_items() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-non-object-items-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    client = TestClient(app)

    for case in fixture["cases"]:
        response = client.post("/api/document.validate_graph", json=case["request"])
        assert response.status_code == case["status"], case["name"]
        assert response.headers["content-type"].startswith("application/json")


def test_python_document_graph_validation_accepts_multimodal_only_grounding() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-multimodal-grounding-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]


def test_python_document_graph_validation_rejects_malformed_multimodal_evidence() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-multimodal-grounding-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    invalid_requests = []
    for field, value in (
        ("content_sha256", "not-a-digest"),
        ("modality", "image"),
        ("spans", "not-a-list"),
    ):
        request = json.loads(json.dumps(fixture["request"]))
        mention = request["nodes"][0]["mentions"][0]
        span = mention["multimodal_spans"][0]
        if field == "modality":
            span["modality"] = value
            span["locator"] = {"kind": "temporal_interval", "start_ms": 0, "end_ms": 1}
        elif field == "spans":
            mention[field] = value
        else:
            span[field] = value
        invalid_requests.append(request)

    client = TestClient(app)
    for request in invalid_requests:
        response = client.post("/api/document.validate_graph", json=request)
        assert response.status_code == fixture["status"]
        body = response.json()
        assert body["ok"] is False
        assert "node-image-grounded" in body["node_errors"]


def test_python_document_graph_validation_defaults_omitted_edges_to_empty() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-empty-edges-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]

def test_python_document_graph_validation_rejects_incomplete_pointer_without_500() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-invalid-pointer-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    body = response.json()
    assert body["ok"] is fixture["ok"]
    assert list(body["node_errors"]) == fixture["node_error_ids"]


def test_python_document_graph_validation_preserves_integer_string_pointer_coercion() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-integer-string-pointer-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]


def test_python_document_graph_validation_accepts_u64_max_pointer_offsets() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-u64-offset-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]

    out_of_range_request = json.loads(json.dumps(fixture["request"]))
    pointer = out_of_range_request["nodes"][0]["metadata"]["pointers"][0]
    pointer["start_char"] = "18446744073709551616"
    pointer["end_char"] = "18446744073709551617"
    rejected = TestClient(app).post(
        "/api/document.validate_graph",
        json=out_of_range_request,
    )
    assert rejected.status_code == fixture["status"]
    assert rejected.json()["ok"] is False
    assert list(rejected.json()["node_errors"]) == ["node-u64-offset"]


def test_python_document_graph_validation_coerces_boolean_pointer_offset() -> None:
    fixture_path = _FIXTURE.with_name("document-validate-graph-bool-offset-v1.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    response = TestClient(app).post(
        "/api/document.validate_graph",
        json=fixture["request"],
    )

    assert response.status_code == fixture["status"]
    assert response.json() == fixture["response"]
