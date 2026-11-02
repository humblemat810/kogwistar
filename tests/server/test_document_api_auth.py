from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException

from kogwistar.server.auth_middleware import claims_ctx


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
    payload = server.DocumentUpsert(
        doc_id="auth-regression-doc",
        doc_type="text",
        insertion_method="api-test",
        content="authorized content",
    )

    token = claims_ctx.set({"role": "rw"})
    try:
        result = server.document_upsert(payload)
    finally:
        claims_ctx.reset(token)

    assert result.status == "ok"
    assert len(written) == 1
    assert written[0].id == "auth-regression-doc"
