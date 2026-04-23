from __future__ import annotations
# '''_async_chroma_real.py'''
import contextlib
import dataclasses
import os
import shutil
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


_ASYNC_CHROMA_SERVER_CLIENTS: dict[int, Any] = {}


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _read_proc_output(proc: subprocess.Popen[str]) -> str:
    if proc.stdout is None:
        return ""
    with contextlib.suppress(Exception):
        return proc.stdout.read() or ""
    return ""


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
                    run_awaitable_blocking(close())
        clients.clear()
    _ASYNC_CHROMA_SERVER_CLIENTS.clear()


@pytest.fixture(autouse=True)
def _cleanup_async_chroma_clients_after_test():
    yield
    _close_async_chroma_server_clients()


def start_real_chroma_server(tmp_path: Path) -> RealChromaServer:
    pytest.importorskip("chromadb")

    chroma_cli = shutil.which("chroma")
    if chroma_cli is None:
        pytest.skip("real async Chroma tests require the `chroma` CLI")

    host = "127.0.0.1"
    port = _free_port()
    persist_dir = tmp_path / "pytest-async-chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        chroma_cli,
        "run",
        "--path",
        str(persist_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = RealChromaServer(
        proc=proc, host=host, port=port, persist_dir=persist_dir
    )

    deadline = time.monotonic() + 60.0
    heartbeat_url = f"http://{host}:{port}/api/v2/heartbeat"
    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = _read_proc_output(proc)
                raise RuntimeError(
                    "Chroma server exited before it became ready "
                    f"(code={proc.returncode}).\n{output}"
                )
            with contextlib.suppress(Exception):
                res = client.get(heartbeat_url)
                if res.status_code == 200:
                    return server
            time.sleep(0.25)

    output = _read_proc_output(proc)
    proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)
    if proc.poll() is None:
        proc.kill()
    with contextlib.suppress(Exception):
        if proc.stdout is not None:
            proc.stdout.close()
    raise TimeoutError(
        "Timed out waiting for Chroma server to become ready.\n" + output
    )


def stop_real_chroma_server(server: RealChromaServer) -> None:
    proc = server.proc
    if proc.poll() is not None:
        with contextlib.suppress(Exception):
            if proc.stdout is not None:
                proc.stdout.close()
        return
    proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=10)
    if proc.poll() is None:
        proc.kill()
    with contextlib.suppress(Exception):
        if proc.stdout is not None:
            proc.stdout.close()


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

    client = await chromadb.AsyncHttpClient(host=server.host, port=server.port)
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
        collections[key] = await client.get_or_create_collection(
            name=collection_name
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
