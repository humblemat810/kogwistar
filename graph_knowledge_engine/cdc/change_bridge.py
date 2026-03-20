from __future__ import annotations

import sys
import pathlib

if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).absolute().parent.parent.parent))

import argparse
import asyncio
import json
import logging

from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

# Canonical ChangeEvent type (used by engine).
from graph_knowledge_engine.cdc.change_event import ChangeEvent
from graph_knowledge_engine.cdc.oplog import OplogReader, OplogWriter

logger = logging.getLogger(__name__)
log_path = Path(".cdc_debug") / "cdc_bridge.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

# Configure handler (append mode by default)
handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Optional: prevent duplicate logs if root logger also configured
logger.propagate = False


# Example usage
logger.info("CDC bridge started")


def _event_graph_type(evj: dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of graph type from a ChangeEvent jsonable dict.

    We primarily rely on `evj["entity"]["kg_graph_type"]` (the canonical field
    used across the repo), but keep fallbacks for older / custom emitters.
    """
    ent = (evj.get("entity") or {}) if isinstance(evj, dict) else {}
    g = (
        ent.get("kg_graph_type")
        or ent.get("graph_type")
        or evj.get("kg_graph_type")
        or evj.get("graph_type")
    )
    return str(g) if g is not None else None


def _stream_match(evj: dict[str, Any], stream: Optional[str]) -> bool:
    """Return True if event should be sent to a subscriber with `stream` filter.

    Backward compatible behavior:
      - stream=None -> accept all events
    """
    if not stream:
        return True
    gt = _event_graph_type(evj)
    return gt == stream


def create_app(*, oplog_file: Path, fsync: bool = False) -> FastAPI:
    """CDC bridge server with replay + optional stream filtering.

    Responsibilities:
      - Accept ChangeEvent JSON via POST /ingest (single event or {"events": [...]}).
      - Append every event to a file oplog (durable).
      - Broadcast live events to websocket subscribers (/changes/ws).
      - On websocket connect, replay oplog events with seq > ?since, then tail live.

    Optional filtering:
      - WebSocket subscribers may pass `?stream=<graph_type>` to only receive events
        for that graph_type (e.g. conversation/workflow/knowledge).
      - If omitted, behavior is unchanged: subscriber receives all events.

    Notes:
      - The engine remains "core/db only" and simply posts events.
      - Replay is entirely bridge-owned.
    """

    app = FastAPI()

    oplog_writer = OplogWriter(oplog_file, fsync=fsync)
    oplog_reader = OplogReader(oplog_file)

    # Each subscriber can optionally request a stream filter.
    subscribers: dict[WebSocket, Optional[str]] = {}
    subs_lock = asyncio.Lock()
    ingest_lock = asyncio.Lock()

    async def _broadcast_jsonable(evj: dict[str, Any]) -> None:
        # Snapshot subscriber list to avoid holding lock across network I/O.
        async with subs_lock:
            targets = list(subscribers.items())

        dead: list[WebSocket] = []
        msg = json.dumps(evj, ensure_ascii=False)

        for ws, stream in targets:
            if not _stream_match(evj, stream):
                continue
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        if dead:
            async with subs_lock:
                for ws in dead:
                    subscribers.pop(ws, None)

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
        stream: Optional[str] = Query(
            default=None,
            description="Optional graph-type filter (conversation/workflow/knowledge). Backward compatible when omitted.",
        ),
    ) -> None:
        """WebSocket endpoint for real-time change events with replay.
        
        ### Connection Lifecycle & Race-Condition Prevention:
        To ensure a seamless transition from historical replay to live tail without
        missing or duplicating events, we use a 'watermark' strategy:
        
        1. **Watermark Acquisition**: We lock ingestion and find the current maximum
           sequence number in the oplog. This is our 'watermark'.
        2. **Initial Replay**: We replay all events from `since` up to the `watermark`.
        3. **Live Subscription**: we add the client to the active subscribers list.
        4. **Gap Catch-up**: We perform a final scan of the oplog for any events with
           `seq > watermark` that might have arrived between steps 1 and 3.
        """
        await websocket.accept()

        # Determine a replay watermark under ingest_lock so we can avoid missing
        # events that arrive during the replay window.
        async with ingest_lock:
            watermark = since
            for ev in oplog_reader.iter_since(since_seq=0, limit=None):
                if ev.seq > watermark:
                    watermark = ev.seq

        # Replay up to watermark (Historical Phase)
        for ev in oplog_reader.iter_since(since_seq=since, limit=None):
            if ev.seq <= watermark:
                evj = ev.to_jsonable()
                if _stream_match(evj, stream):
                    await websocket.send_text(json.dumps(evj, ensure_ascii=False))
            else:
                break

        # Subscribe for live tail (Live Phase)
        async with subs_lock:
            subscribers[websocket] = stream

        # Catch-up in case events arrived between watermark scan and subscribe.
        # This ensures that no events are lost in the 'handoff' between replay and live tail.
        for ev in oplog_reader.iter_since(since_seq=watermark, limit=None):
            evj = ev.to_jsonable()
            if _stream_match(evj, stream):
                await websocket.send_text(json.dumps(evj, ensure_ascii=False))

        try:
            while True:
                # Keepalive / optional client control messages (ignored for now).
                await websocket.receive_text()
        except WebSocketDisconnect:
            async with subs_lock:
                subscribers.pop(websocket, None)

    return app


def reset_oplog(oplog_file: Path) -> None:
    """Delete past oplog file (best-effort), used by --reset-oplog at launch."""
    try:
        if oplog_file.exists():
            oplog_file.unlink()
            logger.info("reset_oplog removed %s", oplog_file)
    except Exception as e:
        logger.exception("reset_oplog failed for %s: %s", oplog_file, e)


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="CDC change bridge launcher")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument(
        "--oplog-file", type=Path, default=Path(".cdc_debug/data/cdc_oplog.jsonl")
    )
    p.add_argument(
        "--fsync",
        action="store_true",
        help="fsync oplog on each append (safer, slower)",
    )
    p.add_argument(
        "--reset-oplog", action="store_true", help="delete oplog file before starting"
    )
    p.add_argument(
        "--log-level",
        default="info",
        help="uvicorn log level (debug/info/warning/error/critical)",
    )
    p.add_argument("--access-log", action="store_true")
    p.add_argument("--reload", action="store_true")
    p.add_argument("--app-dir")

    args = p.parse_args(argv)

    # Ensure parent dir exists for oplog
    args.oplog_file.parent.mkdir(parents=True, exist_ok=True)

    if args.reset_oplog:
        reset_oplog(args.oplog_file)

    app = create_app(oplog_file=args.oplog_file, fsync=args.fsync)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
        reload=args.reload,
    )
    return 0


if __name__ != "__main__":
    app = create_app(oplog_file=Path(".cdc_debug/data/cdc_oplog.jsonl"))
# Default app instance for uvicorn import-style launch (backward compatible)

"""
Normal usage:

python -m graph_knowledge_engine.cdc.change_bridge --host 127.0.0.1 --port 8787

Reset oplog on launch:

python -m graph_knowledge_engine.cdc.change_bridge --reset-oplog

Custom oplog path + fsync:

python -m graph_knowledge_engine.cdc.change_bridge --oplog-file .cdc_debug/data/my_oplog.jsonl --fsync

python -m graph_knowledge_engine.cdc.change_bridge --host 127.0.0.1 --port 8787 --oplog-file .cdc_debug/data/cdc_oplog.jsonl --reset-oplog
"""
if __name__ == "__main__":
    import sys

    raise SystemExit(main())


# uvicorn graph_knowledge_engine.cdc.change_bridge:app --host 127.0.0.1 --port 8787 --log-level info --access-log
