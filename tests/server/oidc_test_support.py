from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx


OIDC_TEST_IDENTITY = {
    "user_id": "dev-user-id",
    "username": "dev",
    "password": "dev",
    "email": "dev@example.com",
    "display_name": "Dev User",
    "role": "ro",
    "namespaces": "docs,conversation,workflow,wisdom",
}


def oidc_seed_json() -> str:
    return json.dumps(
        [
            {
                "user_id": OIDC_TEST_IDENTITY["user_id"],
                "email": OIDC_TEST_IDENTITY["email"],
                "display_name": OIDC_TEST_IDENTITY["display_name"],
                "global_role": OIDC_TEST_IDENTITY["role"],
                "global_ns": OIDC_TEST_IDENTITY["namespaces"],
                "identities": [
                    {
                        "issuer": "dev",
                        "subject": OIDC_TEST_IDENTITY["username"],
                    }
                ],
            }
        ]
    )


def run_subprocess(
    args: list[str], *, timeout: float = 60.0, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = run_subprocess(["docker", "info"], timeout=20.0, check=False)
    except Exception:
        return False
    return proc.returncode == 0


def start_keycloak_container() -> dict[str, str]:
    realm_path = Path(__file__).resolve().parents[2] / "keycloak" / "realm-kge.json"
    bind_src = realm_path.resolve().as_posix()
    mount_arg = (
        f"type=bind,src={bind_src},"
        "dst=/opt/keycloak/data/import/realm-kge.json,readonly"
    )

    proc = run_subprocess(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--mount",
            mount_arg,
            "-e",
            "KC_BOOTSTRAP_ADMIN_USERNAME=admin",
            "-e",
            "KC_BOOTSTRAP_ADMIN_PASSWORD=admin",
            "-p",
            "127.0.0.1::8080",
            "quay.io/keycloak/keycloak:24.0",
            "start-dev",
            "--import-realm",
        ],
        timeout=120.0,
    )
    container_id = proc.stdout.strip()

    port_proc = run_subprocess(
        [
            "docker",
            "inspect",
            "-f",
            "{{(index (index .NetworkSettings.Ports \"8080/tcp\") 0).HostPort}}",
            container_id,
        ],
        timeout=20.0,
    )
    host_port = port_proc.stdout.strip()
    base_url = f"http://127.0.0.1:{host_port}"
    discovery_url = f"{base_url}/realms/kge/.well-known/openid-configuration"

    deadline = time.time() + 120.0
    last_error: str | None = None
    while time.time() < deadline:
        try:
            resp = httpx.get(discovery_url, timeout=5.0)
            if resp.status_code == 200 and "authorization_endpoint" in resp.text:
                config = resp.json()
                return {
                    "container_id": container_id,
                    "base_url": base_url,
                    "discovery_url": discovery_url,
                    "issuer": str(config.get("issuer") or ""),
                }
            last_error = f"unexpected status {resp.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2.0)

    raise AssertionError(
        f"Keycloak did not become ready in time: {last_error or 'unknown error'}"
    )


def stop_keycloak_container(container_id: str) -> None:
    run_subprocess(["docker", "rm", "-f", container_id], timeout=30.0, check=False)


def extract_login_action(html_text: str, page_url: str) -> str:
    match = re.search(
        r'<form[^>]*id="kc-form-login"[^>]*action="([^"]+)"',
        html_text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r'<form[^>]*action="([^"]+)"', html_text, flags=re.IGNORECASE)
    if not match:
        raise AssertionError("Could not find Keycloak login form action")
    return urljoin(page_url, html.unescape(match.group(1)))
