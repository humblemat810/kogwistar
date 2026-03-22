from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import pytest
import requests
import websocket  # pip install websocket-client


from fastapi.testclient import TestClient

from kogwistar.cdc.change_bridge import create_app
from kogwistar.cdc.change_event import ChangeEvent
from kogwistar.cdc.oplog import OplogWriter


_SAMPLE_CHANGESET: tuple[ChangeEvent, ...] = (
    ChangeEvent(
        seq=1,
        op="node.upsert",
        ts_unix_ms=1700000000001,
        entity={
            "kind": "node",
            "id": "conv-node-1",
            "kg_graph_type": "conversation",
            "url": None,
        },
        payload={
            "id": "conv-node-1",
            "label": "User turn 1",
            "type": "message",
            "props": {"role": "user", "text": "Hello"},
        },
    ),
    ChangeEvent(
        seq=2,
        op="node.upsert",
        ts_unix_ms=1700000000002,
        entity={
            "kind": "node",
            "id": "conv-node-2",
            "kg_graph_type": "conversation",
            "url": None,
        },
        payload={
            "id": "conv-node-2",
            "label": "Assistant turn 1",
            "type": "message",
            "props": {"role": "assistant", "text": "Hi there"},
        },
        run_id="sample-run-1",
        step_id="step-1",
    ),
    ChangeEvent(
        seq=3,
        op="edge.upsert",
        ts_unix_ms=1700000000003,
        entity={
            "kind": "edge",
            "id": "conv-edge-1",
            "kg_graph_type": "conversation",
            "url": None,
        },
        payload={
            "id": "conv-edge-1",
            "source": "conv-node-1",
            "target": "conv-node-2",
            "type": "replies_to",
            "props": {},
        },
        run_id="sample-run-1",
        step_id="step-1",
    ),
    ChangeEvent(
        seq=4,
        op="doc.upsert",
        ts_unix_ms=1700000000004,
        entity={
            "kind": "doc_node",
            "id": "conv-doc-1",
            "kg_graph_type": "conversation",
            "url": "doc://conversation/sample-1",
        },
        payload={
            "id": "conv-doc-1",
            "title": "Conversation sample 1",
            "text": "Hello\nHi there",
        },
        run_id="sample-run-1",
        step_id="step-2",
    ),
    ChangeEvent(
        seq=5,
        op="search_index.upsert",
        ts_unix_ms=1700000000005,
        entity={
            "kind": "search_index",
            "id": "conv-index-1",
            "kg_graph_type": "conversation",
            "url": None,
        },
        payload={
            "id": "conv-index-1",
            "text": "hello hi there",
            "doc_id": "conv-doc-1",
        },
        run_id="sample-run-1",
        step_id="step-2",
    ),
)


def _export_sample_oplog(path: Path) -> Path:
    writer = OplogWriter(path, fsync=False)
    for event in _SAMPLE_CHANGESET:
        writer.append(event)
    return path


@pytest.fixture
def replay_oplog_path(tmp_path: Path) -> Path:
    repo_path = Path("tests/input_test_data/changes.jsonl")
    if repo_path.exists():
        return repo_path
    return _export_sample_oplog(tmp_path / "changes.jsonl")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Minimal JSONL reader that skips non-JSON/header lines."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # header or corrupted line
                continue


def _load_filtered_events(
    *,
    oplog_path: Path,
    since_seq: int = 0,
    kg_graph_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[Dict[str, Any]]:
    events: list[Dict[str, Any]] = []
    for ev in _iter_jsonl(oplog_path):
        try:
            seq = int(ev.get("seq", -1))
        except Exception:
            continue

        if seq <= since_seq:
            continue

        if kg_graph_type is not None:
            ent = ev.get("entity") or {}
            if ent.get("kg_graph_type") != kg_graph_type:
                continue

        events.append(ev)
        if limit is not None and len(events) >= limit:
            break
    return events


@pytest.mark.ci_full
def test_replay_oplog_broadcasts_to_websocket(
    tmp_path: Path, replay_oplog_path: Path
) -> None:
    """
    End-to-end: replay JSONL -> POST /ingest -> WS /changes/ws receives.

    This matches the CLI flow:
      replay_oplog_to_bridge.py --oplog ... --bridge ... --kg-graph-type conversation
    """
    # Your test data path
    oplog_path = replay_oplog_path
    assert oplog_path.exists(), f"Missing test data: {oplog_path.resolve()}"

    # If your bridge module stores global subscriber state, it's safer to reset it here.
    # Only do this if change_bridge.py defines a global `subscribers` set.
    # try:
    #     import kogwistar.cdc.change_bridge as bridge_mod  # adjust if needed
    #     if hasattr(bridge_mod, "subscribers"):
    #         bridge_mod.subscribers.clear()
    # except Exception:
    #     pass

    expected_events = _load_filtered_events(
        oplog_path=oplog_path,
        since_seq=0,
        kg_graph_type="conversation",
        limit=10,
    )
    assert expected_events, "No conversation events were available in the oplog fixture"

    with TestClient(create_app(oplog_file=tmp_path / "bridge_oplog.jsonl")) as client:
        # Subscribe to the conversation stream only so unrelated bridge state cannot pollute the assertion.
        with client.websocket_connect("/changes/ws?stream=conversation") as ws:
            received = []
            for ev in expected_events:
                r = client.post("/ingest", json=ev)
                assert r.status_code == 200, r.text
                received.append(json.loads(ws.receive_text()))

    assert [e["seq"] for e in received] == [e["seq"] for e in expected_events]
    assert all(
        (e.get("entity") or {}).get("kg_graph_type") == "conversation" for e in received
    ), (
        f"Received non-conversation events: {[(e.get('entity') or {}).get('kg_graph_type') for e in received]}"
    )


@pytest.mark.manual
def test_replay_oplog_to_real_bridge(replay_oplog_path: Path):
    """
    Requires:
      - FastAPI bridge running at http://127.0.0.1:8787
      - WS endpoint at ws://127.0.0.1:8787/changes/ws
    """
    bridge_http = "http://127.0.0.1:8787/ingest"
    bridge_ws = "ws://127.0.0.1:8787/changes/ws"

    oplog_path = replay_oplog_path
    assert oplog_path.exists()

    # Connect websocket first
    ws = websocket.create_connection(bridge_ws, timeout=5)

    sent = 0
    received = []

    # Replay oplog
    for ev in _iter_jsonl(oplog_path):
        seq = ev.get("seq")
        if not isinstance(seq, int):
            continue

        ent = ev.get("entity") or {}
        if ent.get("kg_graph_type") != "conversation":
            continue

        r = requests.post(bridge_http, json=ev, timeout=1)
        assert r.status_code == 200
        sent += 1

        # try to receive immediately
        try:
            msg = ws.recv()
            received.append(json.loads(msg))
        except Exception as _e:
            pass

        # if sent >= 5:
        #     break

    ws.close()

    assert sent > 0, "No events sent"
    assert received, "No events received from websocket"

    assert any(
        (e.get("entity") or {}).get("kg_graph_type") == "conversation" for e in received
    )
