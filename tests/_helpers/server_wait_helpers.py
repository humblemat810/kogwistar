from __future__ import annotations

import time
from typing import Any

import requests


def wait_for_health(
    session: requests.Session, base_url: str, timeout_s: float = 5.0
) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = session.get(f"{base_url}/health", timeout=1.0)
            if resp.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"Server at {base_url} not healthy. "
        f"Last error: {last_error}"
    )


def wait_for_run_terminal(
    session: requests.Session,
    base_url: str,
    run_id: str,
    headers: dict[str, str],
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = session.get(
            f"{base_url}/api/runs/{run_id}",
            headers=headers,
            timeout=(20.0, 10000.0),
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.5)
    raise AssertionError(f"Run {run_id} did not reach a terminal status within {timeout_s}s")
