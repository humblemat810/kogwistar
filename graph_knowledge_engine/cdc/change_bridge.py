from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

# Canonical ChangeEvent type (used by engine).
from .change_event import ChangeEvent
from .oplog import OplogReader, OplogWriter
import sys

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())


def create_app(*, oplog_file: Path, fsync: bool = False) -> FastAPI:
    """CDC bridge server with replay.

    Responsibilities:
      - Accept ChangeEvent JSON via POST /ingest (single event or {"events": [...]}).
      - Append every event to a file oplog (durable).
      - Broadcast live events to websocket subscribers (/changes/ws).
      - On websocket connect, replay oplog events with seq > ?since, then tail live.

    Notes:
      - The engine remains "core/db only" and simply posts events.
      - Replay is entirely bridge-owned.
    """

    app = FastAPI()

    oplog_writer = OplogWriter(oplog_file, fsync=fsync)
    oplog_reader = OplogReader(oplog_file)

    subscribers: set[WebSocket] = set()
    subs_lock = asyncio.Lock()
    ingest_lock = asyncio.Lock()

    async def _broadcast_jsonable(evj: dict[str, Any]) -> None:
        # Snapshot subscriber list to avoid holding lock across network I/O.
        async with subs_lock:
            targets = list(subscribers)

        dead: list[WebSocket] = []
        msg = json.dumps(evj, ensure_ascii=False)
        for ws in targets:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        if dead:
            async with subs_lock:
                for ws in dead:
                    subscribers.discard(ws)

    @app.post("/ingest")
    async def ingest(body: dict) -> dict[str, Any]:
        """Ingest a single event or a batch.

        Accepts either:
          - a ChangeEvent JSON dict
          - {"events": [ChangeEvent JSON dict, ...]}

        Persists-before-broadcast: live clients only ever observe events that
        have already been appended to the oplog.
        """

        events = body.get("events")
        if isinstance(events, list):
            ev_dicts = events
        else:
            ev_dicts = [body]

        accepted = 0
        last_seq: Optional[int] = None

        # Serialize ingest to preserve oplog ordering.
        async with ingest_lock:
            for d in ev_dicts:
                ev = ChangeEvent.from_jsonable(d)
                oplog_writer.append(ev)
                accepted += 1
                last_seq = ev.seq
                await _broadcast_jsonable(ev.to_jsonable())

        if accepted == 1:
            logger.info("ingest seq=%s op=%s", last_seq, ev_dicts[0].get("op"))
        else:
            logger.info("ingest batch n=%s last_seq=%s", accepted, last_seq)

        return {"ok": True, "accepted": accepted, "last_seq": last_seq}

    @app.websocket("/changes/ws")
    async def ws_changes(
        websocket: WebSocket,
        since: int = Query(
            default=0,
            description="Replay events with seq > since before switching to live tail.",
        ),
    ) -> None:
        await websocket.accept()

        # Determine a replay watermark under ingest_lock so we can avoid missing
        # events that arrive during the replay window.
        async with ingest_lock:
            watermark = since
            for ev in oplog_reader.iter_since(since_seq=0, limit=None):
                if ev.seq > watermark:
                    watermark = ev.seq

        # Replay up to watermark
        for ev in oplog_reader.iter_since(since_seq=since, limit=None):
            if ev.seq <= watermark:
                await websocket.send_text(json.dumps(ev.to_jsonable(), ensure_ascii=False))
            else:
                break

        # Subscribe for live tail.
        async with subs_lock:
            subscribers.add(websocket)

        # Catch-up in case events arrived between watermark scan and subscribe.
        for ev in oplog_reader.iter_since(since_seq=watermark, limit=None):
            await websocket.send_text(json.dumps(ev.to_jsonable(), ensure_ascii=False))

        try:
            while True:
                # Keepalive / optional client control messages (ignored for now).
                await websocket.receive_text()
        except WebSocketDisconnect:
            async with subs_lock:
                subscribers.discard(websocket)

    return app


# Default app instance for uvicorn:
#   uvicorn graph_knowledge_engine.cdc.change_bridge:app
app = create_app(oplog_file=Path("data/cdc_oplog.jsonl"))


# uvicorn graph_knowledge_engine.cdc.change_bridge:app --host 127.0.0.1 --port 8787
"""python - <<'PY'
import asyncio,websockets
async def m(): async with websockets.connect('ws://localhost:8787/changes/ws') as w: async for x in w: print(x)
asyncio.run(m())
PY"""