from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from kogwistar.engine_core.search_index.models import (
    AddIndexEntriesInput,
    IndexingItem,
)
from kogwistar.server.auth_middleware import claims_ctx
from tests.auth_env import TEST_JWT_ALG, TEST_JWT_SECRET


@pytest.mark.ci
def test_add_index_entries_requires_write_role_and_keeps_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kogwistar.server_mcp_with_admin as server

    submitted: list[list[IndexingItem]] = []
    fake_engine = SimpleNamespace(
        search_index=SimpleNamespace(upsert_entries=submitted.append)
    )
    monkeypatch.setattr(
        server,
        "engine",
        SimpleNamespace(get=lambda: fake_engine),
    )
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "search-index-entry-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload = AddIndexEntriesInput.model_validate(fixture["request"])

    for claims in (None, {"role": "ro"}):
        token = claims_ctx.set(claims)
        try:
            with pytest.raises(HTTPException) as error:
                server.add_index_entries(payload)
        finally:
            claims_ctx.reset(token)
        assert error.value.status_code == 403

    assert submitted == []

    token = claims_ctx.set({"role": "rw"})
    try:
        assert server.add_index_entries(payload) == {"ok": True}
    finally:
        claims_ctx.reset(token)

    assert submitted == [payload.index]
    route = next(
        route
        for route in server.app.routes
        if getattr(route, "path", None) == "/api/add_index_entries"
    )
    assert route.status_code in (None, 200)

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    monkeypatch.setattr(
        server.app.router, "lifespan_context", noop_lifespan, raising=False
    )
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "index-writer",
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
            "/api/add_index_entries",
            json=fixture["request"],
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(submitted) == 2
    assert submitted[1] == payload.index
