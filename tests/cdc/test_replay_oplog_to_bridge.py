
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import threading
import pytest
import requests
import websocket  # pip install websocket-client



from fastapi.testclient import TestClient

# Import the bridge app from your codebase
# Adjust import path to match your project layout:
from graph_knowledge_engine.cdc.change_bridge import app  # or wherever change_bridge.py lives


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


def _replay_into_bridge(
    client: TestClient,
    *,
    oplog_path: Path,
    since_seq: int = 0,
    kg_graph_type: Optional[str] = None,
    sleep_ms: int = 0,
) -> int:
    """POST events to /ingest; returns count posted successfully."""
    sent = 0
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

        r = client.post("/ingest", json=ev)
        assert r.status_code == 200, r.text
        sent += 1

        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)
    return sent


@pytest.mark.integration
def test_replay_oplog_broadcasts_to_websocket(tmp_path: Path) -> None:
    """
    End-to-end: replay JSONL -> POST /ingest -> WS /changes/ws receives.

    This matches the CLI flow:
      replay_oplog_to_bridge.py --oplog ... --bridge ... --kg-graph-type conversation
    """
    # Your test data path
    oplog_path = Path("tests/input_test_data/changes.jsonl")
    assert oplog_path.exists(), f"Missing test data: {oplog_path.resolve()}"

    # If your bridge module stores global subscriber state, it's safer to reset it here.
    # Only do this if change_bridge.py defines a global `subscribers` set.
    # try:
    #     import graph_knowledge_engine.cdc.change_bridge as bridge_mod  # adjust if needed
    #     if hasattr(bridge_mod, "subscribers"):
    #         bridge_mod.subscribers.clear()
    # except Exception:
    #     pass

    client = TestClient(app)

    # Connect WS first (bridge only broadcasts live).
    with client.websocket_connect("/changes/ws") as ws:
        # Start replay in background thread (so ws.receive_text() can block safely).
        sent_holder = {"sent": 0}

        def _bg_send():
            sent_holder["sent"] = _replay_into_bridge(
                client,
                oplog_path=oplog_path,
                since_seq=0,
                kg_graph_type="conversation",
                sleep_ms=0,  # keep test fast; you can set to 2 if you want
            )

        t = threading.Thread(target=_bg_send, daemon=True)
        t.start()

        # Collect a few events to prove broadcast works.
        received = []
        deadline = time.time() + 5.0  # seconds

        while time.time() < deadline and len(received) < 10:
            msg = ws.receive_text()
            ev = json.loads(msg)
            received.append(ev)

        t.join(timeout=2.0)

    assert sent_holder["sent"] > 0, "No events were posted from the oplog (filter too strict?)"
    assert len(received) > 0, "No events received on websocket"

    # Validate that at least one received event is the intended topic.
    assert any(
        (e.get("entity") or {}).get("kg_graph_type") == "conversation"
        for e in received
    ), f"Did not receive any conversation events. Got: {[ (e.get('entity') or {}).get('kg_graph_type') for e in received[:10] ]}"

@pytest.mark.manual
def test_replay_oplog_to_real_bridge():
    """
    Requires:
      - FastAPI bridge running at http://127.0.0.1:8787
      - WS endpoint at ws://127.0.0.1:8787/changes/ws
    """
    bridge_http = "http://127.0.0.1:8787/ingest"
    bridge_ws = "ws://127.0.0.1:8787/changes/ws"

    oplog_path = Path("tests/input_test_data/changes.jsonl")
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
        (e.get("entity") or {}).get("kg_graph_type") == "conversation"
        for e in received
    )