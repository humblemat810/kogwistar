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

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(
        server.app.router, "lifespan_context", noop_lifespan, raising=False
    )
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
        response = client.post(
            "/api/document",
            json=fixture,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(written) == 2
    assert written[1].id == fixture["doc_id"]
