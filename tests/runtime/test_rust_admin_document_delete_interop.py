from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from kogwistar.engine_core.canonical_events import read_canonical_entity_revision
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from tests.runtime.test_rust_runtime_authority import (
    RUST_SERVER,
    _test_hs256_token,
)

pytestmark = [pytest.mark.ci, pytest.mark.integration]


def test_rust_admin_delete_is_readable_by_python_sqlite_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not RUST_SERVER.exists():
        pytest.skip("build kogwistar-server before live native-server test")

    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/document-delete-http-v1.json"
        ).read_text(encoding="utf-8")
    )

    database_path = tmp_path / "shared-graph.sqlite"
    metadata = EngineSQLite(tmp_path, database_path.name)
    metadata.ensure_initialized()

    doc_id = "interop-delete-document"
    node_id = "interop-delete-node"
    node_payload = {
        "id": node_id,
        "label": "Interop node",
        "type": "entity",
        "summary": "Node created through Python's canonical event contract",
        "doc_id": doc_id,
        "domain_id": None,
        "level_from_root": 0,
        "properties": {},
        "embedding": None,
        "mentions": [{"spans": [{"doc_id": doc_id}], "multimodal_spans": []}],
        "metadata": {
            "entity_type": "node",
            "label": "Interop node",
            "namespace": "docs",
            "doc_id": doc_id,
            "lifecycle_status": "active",
        },
    }
    metadata.append_entity_event(
        namespace="docs",
        event_id="interop-delete-add-event",
        entity_kind="node",
        entity_id=node_id,
        op="ADD",
        payload_json=json.dumps(node_payload, separators=(",", ":")),
    )
    projection_key = json.dumps(
        ["docs", None, None, doc_id], separators=(",", ":")
    )
    metadata.replace_named_projection(
        "kogwistar.source_documents.v1",
        projection_key,
        {
            "namespace": "docs",
            "workspace_id": None,
            "graph_space": None,
            "record": {
                "id": doc_id,
                "document": "Interop source document",
                "metadata": {"namespace": "docs"},
            },
            "embedding_dim": 0,
        },
        last_authoritative_seq=0,
        last_materialized_seq=0,
        projection_schema_version=1,
        materialization_status="source_document",
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    env = os.environ.copy()
    for name in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD", "AUTH_MODE"):
        env.pop(name, None)
    env.update(
        {
            "KOGWISTAR_BACKEND": "sqlite",
            "KOGWISTAR_META_SQLITE_PATH": str(database_path),
            "KOGWISTAR_SERVER_HOST": "127.0.0.1",
            "KOGWISTAR_SERVER_PORT": str(port),
            "KOGWISTAR_SERVER_REQUIRED_ROLES": "",
        }
    )
    secret = f"admin-delete-{os.urandom(24).hex()}"
    env.update({"JWT_SECRET": secret, "JWT_ALG": "HS256", "AUTH_MODE": "jwt"})
    token = _test_hs256_token(
        secret,
        {
            "sub": "admin-delete-interop-test",
            "role": "rw",
            "ns": ["docs"],
            "storage_ns": "docs",
            "exp": int(time.time()) + 3600,
        },
    )
    headers = {"authorization": f"Bearer {token}"}
    process = subprocess.Popen(
        [str(RUST_SERVER)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise AssertionError(f"Rust server exited before ready: {stderr}")
            try:
                if httpx.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            pytest.fail("Rust server did not become ready")

        response = httpx.request(
            contract["method"],
            f"{base_url}{contract['path'].replace('{doc_id}', doc_id)}",
            headers=headers,
            timeout=10,
        )
        assert response.status_code == contract["success_status"], response.text
        payload = response.json()
        assert set(payload) == set(contract["response_fields"])
        assert payload["ok"] is True
        assert payload["doc_id"] == doc_id
        assert set(payload["deleted"]) == set(contract["deleted_fields"])
        assert payload["deleted"] == {"nodes": 1, "edges": 0}
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if process.stderr is not None:
            process.stderr.close()

    # Python's supported event reader must observe the native server's commit.
    events = list(metadata.iter_entity_events(namespace="docs"))
    assert [str(event[3]) for event in events] == ["ADD", "TOMBSTONE"]
    tombstone = json.loads(str(events[-1][4]))
    assert tombstone["lifecycle_patch"]["lifecycle_status"] == "tombstoned"
    revision = read_canonical_entity_revision(
        events=(tuple(event) for event in events),
        namespace="docs",
        entity_kind="node",
        entity_id=node_id,
    )
    assert revision is not None and revision.state == "deleted"
    assert metadata.get_named_projection(
        "kogwistar.source_documents.v1", projection_key
    ) is None
