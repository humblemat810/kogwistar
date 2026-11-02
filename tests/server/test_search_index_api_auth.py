from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kogwistar.engine_core.search_index.models import (
    AddIndexEntriesInput,
    IndexingItem,
)
from kogwistar.server.auth_middleware import claims_ctx


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
    payload = AddIndexEntriesInput(
        index=[
            IndexingItem(
                node_id="node-1",
                canonical_title="Index contract",
                provision="write-only route test",
            )
        ]
    )

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
