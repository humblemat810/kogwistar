from __future__ import annotations

from typing import Any, Callable

from jose import jwt


def auth_header_from_token(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def mint_dev_token(
    client,
    *,
    role: str,
    ns: str,
    username: str = "tester",
    capabilities: str | None = None,
) -> str:
    payload = {"username": username, "role": role, "ns": ns}
    if capabilities is not None:
        payload["capabilities"] = capabilities
    resp = client.post("/auth/dev-token", json=payload)
    resp.raise_for_status()
    return str(resp.json()["token"])


def token_header(
    client,
    *,
    role: str,
    ns: str,
    username: str = "tester",
    capabilities: str | None = None,
) -> dict[str, str]:
    return auth_header_from_token(
        mint_dev_token(
            client,
            role=role,
            ns=ns,
            username=username,
            capabilities=capabilities,
        )
    )


def mint_dev_token_http(
    session,
    base_url: str,
    *,
    role: str,
    ns: str,
    username: str = "tester",
    capabilities: str | None = None,
    timeout: float = 10.0,
) -> str:
    payload = {"username": username, "role": role, "ns": ns}
    if capabilities is not None:
        payload["capabilities"] = capabilities
    resp = session.post(f"{base_url}/auth/dev-token", json=payload, timeout=timeout)
    resp.raise_for_status()
    return str(resp.json()["token"])


def token_header_http(
    session,
    base_url: str,
    *,
    role: str,
    ns: str,
    username: str = "tester",
    capabilities: str | None = None,
    timeout: float = 10.0,
) -> dict[str, str]:
    return auth_header_from_token(
        mint_dev_token_http(
            session,
            base_url,
            role=role,
            ns=ns,
            username=username,
            capabilities=capabilities,
            timeout=timeout,
        )
    )


def decode_bearer_claims(headers: dict[str, str]) -> dict[str, Any]:
    auth = headers.get("Authorization", "")
    token = auth.split(" ", 1)[1].strip() if auth.startswith("Bearer ") else ""
    if not token:
        return {}
    try:
        return dict(jwt.get_unverified_claims(token))
    except Exception:  # noqa: BLE001
        return {"_decode_error": True}


def decode_token_subject(token: str, *, default: str = "") -> str:
    try:
        claims = jwt.get_unverified_claims(token)
    except Exception:  # noqa: BLE001
        return default
    return str(claims.get("sub") or default)


def register_looping_sleep_workflow_http(
    session,
    base_url: str,
    *,
    workflow_id: str,
    headers: dict[str, str],
    designer_id: str = "tester",
    timeout: float = 20.0,
    on_error: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    start_id = f"wf|{workflow_id}|start"
    sleep_id = f"wf|{workflow_id}|sleep"
    end_id = f"wf|{workflow_id}|end"

    for payload in (
        {
            "designer_id": designer_id,
            "node_id": start_id,
            "label": "Start",
            "op": "start",
            "start": True,
        },
        {
            "designer_id": designer_id,
            "node_id": sleep_id,
            "label": "Sleep",
            "op": "sleep",
        },
        {
            "designer_id": designer_id,
            "node_id": end_id,
            "label": "End",
            "op": "end",
            "terminal": True,
        },
    ):
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/nodes",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if not resp.ok and on_error is not None:
            on_error(
                {
                    "stage": "workflow_design_node_upsert_fail",
                    "workflow_id": workflow_id,
                    "endpoint": f"/api/workflow/design/{workflow_id}/nodes",
                    "status_code": resp.status_code,
                    "response_text": resp.text[:500],
                    "headers_claims": decode_bearer_claims(headers),
                    "payload": payload,
                }
            )
        assert resp.ok, (
            f"workflow node upsert failed for workflow_id={workflow_id}: "
            f"status={resp.status_code} body={resp.text}"
        )

    for payload in (
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|start_sleep",
            "src": start_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_end",
            "src": sleep_id,
            "dst": end_id,
            "relation": "wf_next",
            "predicate": "always_false",
            "priority": 0,
        },
        {
            "designer_id": designer_id,
            "edge_id": f"wf|{workflow_id}|e|sleep_sleep",
            "src": sleep_id,
            "dst": sleep_id,
            "relation": "wf_next",
            "is_default": True,
            "priority": 1,
        },
    ):
        resp = session.post(
            f"{base_url}/api/workflow/design/{workflow_id}/edges",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        assert resp.ok, (
            f"workflow edge upsert failed for workflow_id={workflow_id}: "
            f"status={resp.status_code} body={resp.text}"
        )
