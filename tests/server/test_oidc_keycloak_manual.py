from __future__ import annotations

import os
import csv
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import httpx
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("fastmcp")
pytest.importorskip("sqlalchemy")
from sqlalchemy.orm import sessionmaker

from kogwistar.server.auth.db import create_auth_engine
from kogwistar.server.auth.models import ExternalIdentity, User
from tests.server.oidc_test_support import oidc_provider_json, oidc_seed_json


pytestmark = [pytest.mark.manual]


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _port_owner_details(port: int) -> str | None:
    try:
        netstat = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except Exception:
        return None

    pid: str | None = None
    for line in netstat.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        proto, local_addr, _, state, owner_pid = parts[:5]
        if proto.upper() != "TCP":
            continue
        if not local_addr.endswith(f":{port}"):
            continue
        if state.upper() != "LISTENING":
            continue
        pid = owner_pid
        break

    if not pid:
        return None

    try:
        tasklist = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
        row = next(csv.reader(tasklist.stdout.splitlines()), None)
        if row and len(row) >= 2:
            return f"PID {pid} ({row[0]})"
    except Exception:
        pass
    return f"PID {pid}"


@pytest.fixture
def manual_oidc_server(
    tmp_path,
    keycloak_container: dict[str, str],
    oidc_test_identity,
):
    host = "127.0.0.1"
    port = 28110
    if not _port_available(host, port):
        owner = _port_owner_details(port)
        detail = f" by {owner}" if owner else ""
        pytest.skip(f"manual OIDC app port {port} is already in use{detail}")

    data_root = tmp_path / "manual_oidc"
    data_root.mkdir(parents=True, exist_ok=True)
    auth_db_path = (data_root / "auth.sqlite").resolve()

    env = {
        **os.environ,
        "AUTH_MODE": "oidc",
        "AUTH_DB_URL": f"sqlite:///{auth_db_path.as_posix()}",
        "UI_URL": "http://localhost:28110/__test__/auth/success",
        "OIDC_PROVIDERS_JSON": oidc_provider_json(
            discovery_url=keycloak_container["discovery_url"],
            redirect_uri="http://localhost:28110/api/auth/callback",
            issuer=keycloak_container["issuer"],
        ),
        "JWT_SECRET": "dev-secret",
        "GKE_BACKEND": "chroma",
        "GKE_PERSIST_DIRECTORY": str((data_root / "gke-data").resolve()),
        "ANONYMIZED_TELEMETRY": "FALSE",
        "DEV_AUTH_SEED_JSON": oidc_seed_json(),
    }

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "tests.server.manual_oidc_test_app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    log_buf: deque[str] = deque(maxlen=400)
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            log_buf.append(line.rstrip())

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    base_url = "http://localhost:28110"
    deadline = time.time() + 90.0
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            joined = "\n".join(log_buf)
            raise RuntimeError(
                f"OIDC app server exited before becoming healthy (exit={proc.returncode}).\n{joined}"
            )
        try:
            health = httpx.get(f"{base_url}/health", timeout=2.0)
            if health.status_code == 200:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    else:
        joined = "\n".join(log_buf)
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            proc.kill()
            proc.wait(timeout=5.0)
        raise RuntimeError(
            f"OIDC app server did not become healthy at {base_url}: {last_error}\n{joined}"
        )

    try:
        yield {
            "base_url": base_url,
            "success_url": f"{base_url}/__test__/auth/success",
            "result_url": f"{base_url}/__test__/auth/result",
            "auth_db_url": env["AUTH_DB_URL"],
            "auth_db_path": auth_db_path,
            "logs": log_buf,
        }
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                proc.kill()
                proc.wait(timeout=5.0)


def test_oidc_keycloak_browser_manual(
    manual_oidc_server,
    keycloak_container: dict[str, str],
    oidc_test_identity,
):
    login_url = f"{manual_oidc_server['base_url']}/api/auth/login"

    print("")
    print("Manual OIDC browser flow")
    print(f"- App login URL: {login_url}")
    print(f"- Keycloak discovery: {keycloak_container['discovery_url']}")
    print(f"- Final backend success page: {manual_oidc_server['success_url']}")
    print(f"- Sign in with username: {oidc_test_identity['username']}")
    print(f"- Sign in with password: {oidc_test_identity['password']}")
    print("- After the browser shows email, role, and namespaces, return here.")
    print("")

    opened = webbrowser.open(login_url)
    if not opened:
        print("Browser was not opened automatically. Open this URL manually:")
        print(login_url)

    deadline = time.time() + 300.0
    result_payload = None
    while time.time() < deadline:
        try:
            result_resp = httpx.get(manual_oidc_server["result_url"], timeout=5.0)
            if result_resp.status_code == 200:
                payload = result_resp.json()
                if payload.get("ready"):
                    result_payload = payload
                    break
        except Exception:
            pass
        time.sleep(1.0)

    assert result_payload is not None, (
        "Timed out waiting for manual OIDC login completion. "
        "Open the login URL again and finish the Keycloak sign-in."
    )

    token = result_payload["token"]
    assert token

    me_resp = httpx.get(
        f"{manual_oidc_server['base_url']}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    assert me_resp.status_code == 200
    me = me_resp.json()
    assert me["email"] == oidc_test_identity["email"]
    assert me["role"] == oidc_test_identity["role"]
    assert result_payload["email"] == oidc_test_identity["email"]
    assert result_payload["role"] == oidc_test_identity["role"]
    assert oidc_test_identity["role"] == me["role"]

    engine = create_auth_engine(manual_oidc_server["auth_db_url"])
    session = sessionmaker(bind=engine)()
    try:
        user = session.query(User).filter(User.email == oidc_test_identity["email"]).first()
        assert user is not None

        identity = (
            session.query(ExternalIdentity)
            .filter(ExternalIdentity.user_id == user.user_id)
            .filter(ExternalIdentity.issuer == keycloak_container["issuer"])
            .first()
        )
        assert identity is not None
        assert identity.email == oidc_test_identity["email"]
    finally:
        session.close()
        engine.dispose()
