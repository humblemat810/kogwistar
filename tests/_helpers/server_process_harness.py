from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from tests.net_helpers import pick_free_port


def start_cdc_bridge(
    *,
    monkeypatch: pytest.MonkeyPatch | None,
    backend_kind: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> str:
    host = "127.0.0.1"
    port = pick_free_port()
    oplog_file = tmp_path / f"{backend_kind}.cdc_oplog.jsonl"
    log_file = Path.cwd() / ".tmp_runtime_sse_debug" / f"{backend_kind}.cdc_bridge.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    cmd = [
        sys.executable,
        "-m",
        "kogwistar.cdc.change_bridge",
        "--host",
        host,
        "--port",
        str(port),
        "--oplog-file",
        str(oplog_file),
        "--reset-oplog",
        "--log-level",
        "warning",
    ]
    log_buf: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _reader() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stripped = line.rstrip()
            log_buf.append(stripped)
            try:
                with log_file.open("a", encoding="utf-8") as fh:
                    fh.write(stripped + "\n")
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)
        t.join(timeout=5.0)
        if proc.stdout is not None:
            proc.stdout.close()

    request.addfinalizer(_cleanup)

    endpoint = f"http://{host}:{port}"
    deadline = time.time() + 30.0
    last_err: Exception | None = None
    with requests.Session() as session:
        while time.time() < deadline:
            if proc.poll() is not None:
                if proc.stdout is not None:
                    proc.stdout.close()
                raise RuntimeError(
                    "CDC bridge exited before becoming healthy.\n"
                    f"exit={proc.returncode}\n"
                    + "\n".join(log_buf[-80:])
                )
            try:
                health = session.get(f"{endpoint}/openapi.json", timeout=1.5)
                if health.ok:
                    if monkeypatch is not None:
                        monkeypatch.setenv("CDC_PUBLISH_ENDPOINT", endpoint)
                    return endpoint
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(0.2)

    if proc.stdout is not None:
        proc.stdout.close()
    raise RuntimeError(
        f"CDC bridge at {endpoint} did not become healthy.\n"
        f"last_error={last_err!r}\n"
        f"log_tail=\n" + "\n".join(log_buf[-80:])
    )


def wait_for_cdc_oplog_entries(
    oplog_file: Path, *, min_entries: int = 1, timeout_s: float = 20.0
) -> int:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if oplog_file.exists():
            try:
                with oplog_file.open("r", encoding="utf-8") as fh:
                    line_count = sum(1 for line in fh if line.strip())
                entry_count = max(0, line_count - 1)
                if entry_count >= min_entries:
                    return entry_count
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        time.sleep(0.1)
    raise AssertionError(
        f"CDC oplog did not record {min_entries} entries in time: {oplog_file}\n"
        f"last_error={last_error!r}"
    )


def count_cdc_oplog_entries(oplog_file: Path) -> int:
    if not oplog_file.exists():
        return 0
    with oplog_file.open("r", encoding="utf-8") as fh:
        line_count = sum(1 for line in fh if line.strip())
    return max(0, line_count - 1)


@contextmanager
def real_server_base_url(
    *,
    backend_kind: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    runtime_runner_import: str | None = None,
):
    import requests

    port = pick_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    debug_dir = Path.cwd() / ".tmp_runtime_sse_debug"
    debug_log = debug_dir / f"{backend_kind}.jsonl"
    server_log = debug_dir / f"{backend_kind}-server.log"
    if debug_log.exists():
        debug_log.unlink()
    if server_log.exists():
        server_log.unlink()

    env = os.environ.copy()
    env["GKE_PERSIST_DIRECTORY"] = str(tmp_path / "server-data")
    env["GKE_INDEX_DIR"] = str(tmp_path / "index")
    env["AUTH_MODE"] = "dev"
    env["JWT_ALG"] = "HS256"
    env["JWT_SECRET"] = "kogwistar-test-secret"
    env["KOGWISTAR_RUNTIME_SSE_DEBUG_LOG"] = str(debug_log)
    if runtime_runner_import:
        env["KOGWISTAR_TEST_RUNTIME_RUNNER_IMPORT"] = str(runtime_runner_import)
    env["CDC_PUBLISH_ENDPOINT"] = start_cdc_bridge(
        monkeypatch=None,
        backend_kind=backend_kind,
        tmp_path=tmp_path,
        request=request,
    )
    if backend_kind == "chroma":
        pytest.importorskip("chromadb")
        chroma_server = request.getfixturevalue("real_chroma_server")
        env["GKE_BACKEND"] = "chroma"
        env["GKE_CHROMA_ASYNC"] = "1"
        env["GKE_CHROMA_HOST"] = str(chroma_server.host)
        env["GKE_CHROMA_PORT"] = str(chroma_server.port)
    elif backend_kind == "pg":
        pg_dsn = request.getfixturevalue("pg_dsn")
        if not pg_dsn:
            pytest.skip("async pg fixtures are unavailable in this environment")
        env["GKE_BACKEND"] = "pg"
        env["GKE_PG_ASYNC"] = "1"
        env["GKE_PG_URL"] = str(pg_dsn)
        env["KOGWISTAR_TEST_EMBEDDING_DIM"] = "384"
    else:
        raise ValueError(f"unknown backend_kind: {backend_kind!r}")

    port_file = tmp_path / f"{backend_kind}.port"
    env["PORT"] = str(port)
    env["HOST"] = host

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "kogwistar.server_mcp_with_admin:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    log_buf: list[str] = []
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
            stripped = line.rstrip()
            log_buf.append(stripped)
            try:
                with server_log.open("a", encoding="utf-8") as fh:
                    fh.write(stripped + "\n")
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        deadline = time.time() + 90.0
        last_err: Exception | None = None
        ready = False
        with requests.Session() as session:
            while time.time() < deadline:
                if proc.poll() is not None:
                    if proc.stdout is not None:
                        proc.stdout.close()
                    joined = "\n".join(log_buf)
                    raise RuntimeError(
                        f"server exited before becoming healthy (exit={proc.returncode}).\n{joined}"
                    )
                try:
                    r = session.get(f"{base_url}/health", timeout=1.5)
                    if r.ok:
                        ready = True
                        break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                time.sleep(0.5)
        if not ready:
            raise RuntimeError(
                "server did not become healthy in time.\n"
                f"last_error={last_err!r}\n"
                + "\n".join(log_buf[-80:])
            )
        if proc.poll() is not None:
            joined = "\n".join(log_buf)
            raise RuntimeError(
                f"server exited before becoming healthy (exit={proc.returncode}).\n{joined}"
            )

        with requests.Session() as session:
            port_value = session.get(f"{base_url}/health", timeout=1.5)
            port_value.raise_for_status()
        port_file.write_text(str(port), encoding="utf-8")
        yield base_url
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15.0)
        t.join(timeout=5.0)
        if proc.stdout is not None:
            proc.stdout.close()
