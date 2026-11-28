from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from typing import cast
from pathlib import Path
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from kogwistar.server.auth_middleware import claims_ctx
from tests.auth_env import TEST_JWT_ALG, TEST_JWT_SECRET


@pytest.mark.ci
@pytest.mark.parametrize(
    "route_name",
    ["document_upsert", "document_upsert_tree", "admin_delete_doc"],
)
@pytest.mark.parametrize("claims", [None, {"role": "ro"}])
def test_document_write_routes_deny_without_rw_before_engine_access(
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
    claims: dict[str, str] | None,
) -> None:
    import kogwistar.server_mcp_with_admin as server

    def engine_access_denied() -> object:
        pytest.fail("unauthorized document write accessed the engine")

    monkeypatch.setattr(
        server,
        "engine",
        SimpleNamespace(get=engine_access_denied),
    )
    route = cast(Callable[[object], object], getattr(server, route_name))
    token = claims_ctx.set(claims)
    try:
        with pytest.raises(HTTPException) as error:
            route(None)
    finally:
        claims_ctx.reset(token)

    assert error.value.status_code == 403


@pytest.mark.ci
def test_document_upsert_keeps_rw_write_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import kogwistar.server_mcp_with_admin as server

    written: list[object] = []
    fake_engine = SimpleNamespace(write=SimpleNamespace(add_document=written.append))
    monkeypatch.setattr(server, "engine", SimpleNamespace(get=lambda: fake_engine))
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "document-upsert-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    http_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/document-upsert-http-v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = server.DocumentUpsert.model_validate(fixture)

    token = claims_ctx.set({"role": "rw"})
    try:
        result = server.document_upsert(payload)
    finally:
        claims_ctx.reset(token)

    assert result.status == "ok"
    assert len(written) == 1
    assert written[0].id == fixture["doc_id"]
    assert written[0].content == fixture["content"]
    assert written[0].metadata["insertion_method"] == fixture["insertion_method"]

    ocr_content = {
        "scan.pdf": [
            {
                "pdf_page_num": 1,
                "OCR_text_clusters": [{"text": "Hello"}],
                "non_text_objects": [],
            }
        ]
    }
    ocr_payload = server.DocumentUpsert(
        doc_id="doc-ocr",
        doc_type="ocr",
        content=json.dumps(ocr_content),
        insertion_method="ocr-parser",
    )
    token = claims_ctx.set({"role": "rw"})
    try:
        ocr_result = server.document_upsert(ocr_payload)
    finally:
        claims_ctx.reset(token)

    assert ocr_result.status == "ok"
    assert len(written) == 2
    ocr_document = written[1]
    assert ocr_document.type == "ocr_document"
    assert ocr_document.content["scan.pdf"][0]["OCR_text_clusters"][0]["id"] == "p1_c0"
    assert ocr_document.source_map == {
        "p1_c0": {"text": "Hello", "id": "p1_c0"}
    }
    assert ocr_document.metadata["insertion_method"] == "ocr-parser"

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(
        server.app.router, "lifespan_context", noop_lifespan, raising=False
    )
    # ???? server ????? JWT ??,???? TEST_JWT_SECRET / iss="local"
    # ??????? env ??,?? stale ? state.auth_jwt_settings?
    _saved_jwt_settings = getattr(server.app.state, "auth_jwt_settings", None)
    server.app.state.auth_jwt_settings = None
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)
    monkeypatch.setenv("JWT_ALG", TEST_JWT_ALG)
    monkeypatch.delenv("JWT_ISS", raising=False)
    monkeypatch.delenv("JWT_AUD", raising=False)
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "document-writer",
            "role": "rw",
            "ns": ["docs"],
            "iat": now,
            "exp": now + 3600,
            "iss": "local",
        },
        TEST_JWT_SECRET,
        algorithm=TEST_JWT_ALG,
    )
    with TestClient(server.app) as client:
        response = client.request(
            http_contract["method"],
            http_contract["path"],
            json=fixture,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == http_contract["success_status"]
    assert response.json() == http_contract["success_response"]
    assert len(written) == 3
    assert written[2].id == fixture["doc_id"]
    server.app.state.auth_jwt_settings = _saved_jwt_settings
