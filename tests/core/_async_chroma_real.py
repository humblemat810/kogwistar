from __future__ import annotations
# '''_async_chroma_real.py'''
import contextlib
import dataclasses
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from kogwistar.engine_core.async_compat import run_awaitable_blocking
from kogwistar.engine_core.chroma_backend import AsyncChromaBackend
from kogwistar.engine_core.storage_backend import AsyncNoopUnitOfWork


@dataclasses.dataclass(slots=True)
class RealChromaServer:
    proc: subprocess.Popen[str]
    host: str
    port: int
    persist_dir: Path
    log_path: Path


_ASYNC_CHROMA_SERVER_CLIENTS: dict[int, Any] = {}
_LIVE_REAL_CHROMA_SERVERS: dict[int, RealChromaServer] = {}


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _tail_text(path: Path, *, limit: int = 16_000) -> str:
    with contextlib.suppress(Exception):
        data = path.read_bytes()
        return data[-limit:].decode(errors="replace")
    return ""


def _terminate_process_tree(proc: subprocess.Popen[str], *, timeout: float) -> None:
    if sys.platform == "win32":
        # Chroma can hand off to a child while its launcher has already
        # exited.  Still issue ``taskkill /T`` for the recorded root; returning
        # early on ``poll()`` would strand that serving child between tests.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        with contextlib.suppress(Exception):
            proc.wait(timeout=timeout)
        return

    if proc.poll() is not None:
        return
    proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout)
    if proc.poll() is None:
        proc.kill()


def _register_async_chroma_server(client: Any) -> None:
    server = getattr(client, "_server", None)
    if server is not None:
        _ASYNC_CHROMA_SERVER_CLIENTS[id(server)] = server


def _close_async_chroma_server_clients() -> None:
    for server in list(_ASYNC_CHROMA_SERVER_CLIENTS.values()):
        clients = getattr(server, "_clients", None)
        if not isinstance(clients, dict):
            continue
        for http_client in list(clients.values()):
            close = getattr(http_client, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    run_awaitable_blocking(asyncio.wait_for(close(), timeout=5.0))
        clients.clear()
    _ASYNC_CHROMA_SERVER_CLIENTS.clear()


@pytest.fixture(autouse=True)
def _cleanup_async_chroma_clients_after_test():
    try:
        yield
    finally:
        _close_async_chroma_server_clients()
        # Chroma CLI may fork its serving process.  Keep an explicit registry
        # so every test gets a final cleanup pass even if a fixture setup or
        # async teardown fails before its own finalizer is reached.
        for server in list(_LIVE_REAL_CHROMA_SERVERS.values()):
            stop_real_chroma_server(server)


def start_real_chroma_server(tmp_path: Path) -> RealChromaServer:
    pytest.importorskip("chromadb")

    host = "127.0.0.1"
    port = _free_port()
    persist_dir = tmp_path / "pytest-async-chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tests._helpers.chroma_server_entry",
        "run",
        "--path",
        str(persist_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_path = tmp_path / "chroma-server.log"
    log_file = log_path.open("w", encoding="utf-8")
    repo_root = str(Path(__file__).resolve().parents[2])
    child_env = os.environ.copy()
    executable = sys.executable
    python_path_parts = [repo_root]
    if sys.platform == "win32":
        # Windows venv ``python.exe`` is itself a launcher that starts the
        # base interpreter and exits.  Use that interpreter directly, while
        # preserving this venv's packages, so ``proc`` owns real server.
        executable = getattr(sys, "_base_executable", sys.executable)
        python_path_parts.append(str(Path(sys.prefix) / "Lib" / "site-packages"))
    python_path_parts.append(child_env.get("PYTHONPATH", ""))
    child_env["PYTHONPATH"] = os.pathsep.join(filter(None, python_path_parts))
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [executable, *cmd[1:]],
        cwd=str(tmp_path),
        env=child_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    log_file.close()
    server = RealChromaServer(
        proc=proc, host=host, port=port, persist_dir=persist_dir, log_path=log_path
    )
    _LIVE_REAL_CHROMA_SERVERS[proc.pid] = server

    deadline = time.monotonic() + 60.0
    heartbeat_url = f"http://{host}:{port}/api/v2/heartbeat"
    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = _tail_text(log_path)
                raise RuntimeError(
                    "Chroma server exited before it became ready "
                    f"(code={proc.returncode}).\n{output}"
                )
            with contextlib.suppress(Exception):
                res = client.get(heartbeat_url)
                if res.status_code == 200:
                    return server
            time.sleep(0.25)

    output = _tail_text(log_path)
    _terminate_process_tree(proc, timeout=5)
    raise TimeoutError(
        "Timed out waiting for Chroma server to become ready.\n" + output
    )


def stop_real_chroma_server(server: RealChromaServer) -> None:
    try:
        _terminate_process_tree(server.proc, timeout=10)
    finally:
        _LIVE_REAL_CHROMA_SERVERS.pop(server.proc.pid, None)


async def _await_chroma_setup(awaitable: Any, *, operation: str, server: RealChromaServer) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=15.0)
    except TimeoutError as exc:
        output = _tail_text(server.log_path)
        stop_real_chroma_server(server)
        raise TimeoutError(
            f"Timed out during Chroma {operation} on {server.host}:{server.port}.\n"
            f"Chroma log tail:\n{output}"
        ) from exc


@pytest.fixture
def real_chroma_server(tmp_path: Path):
    server = start_real_chroma_server(tmp_path)
    try:
        yield server
    finally:
        stop_real_chroma_server(server)


async def make_real_async_chroma_backend(
    server: RealChromaServer, *, collection_prefix: str
) -> tuple[Any, AsyncChromaBackend, dict[str, Any]]:
    import chromadb

    client = await _await_chroma_setup(
        chromadb.AsyncHttpClient(host=server.host, port=server.port),
        operation="AsyncHttpClient",
        server=server,
    )
    _register_async_chroma_server(client)
    collections: dict[str, Any] = {}
    for key in (
        "node_index",
        "node",
        "edge",
        "edge_endpoints",
        "document",
        "domain",
        "node_docs",
        "node_refs",
        "edge_refs",
    ):
        collection_name = f"{collection_prefix}_{key}"
        collections[key] = await _await_chroma_setup(
            client.get_or_create_collection(name=collection_name),
            operation=f"get_or_create_collection({collection_name!r})",
            server=server,
        )

    backend = AsyncChromaBackend(
        node_index_collection=collections["node_index"],
        node_collection=collections["node"],
        edge_collection=collections["edge"],
        edge_endpoints_collection=collections["edge_endpoints"],
        document_collection=collections["document"],
        domain_collection=collections["domain"],
        node_docs_collection=collections["node_docs"],
        node_refs_collection=collections["node_refs"],
        edge_refs_collection=collections["edge_refs"],
    )
    return client, backend, collections


def make_real_async_chroma_uow() -> AsyncNoopUnitOfWork:
    return AsyncNoopUnitOfWork()
