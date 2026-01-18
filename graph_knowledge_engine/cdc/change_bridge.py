from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from knowledge_graph_engine.changes.change_bus import ChangeBus
from knowledge_graph_engine.changes.oplog import OplogReader

router = APIRouter()


@dataclass(frozen=True, slots=True)
class BridgeDeps:
    bus: ChangeBus
    oplog: Optional[OplogReader] = None  # optional durable replay


def _need_snapshot(reason: str) -> str:
    return json.dumps({"type": "need_snapshot", "reason": reason})


def _event_frame(ev) -> str:
    return json.dumps({"type": "event", "event": ev.to_jsonable()}, ensure_ascii=False)


def _batch_frame(events) -> str:
    return json.dumps(
        {"type": "batch", "events": [e.to_jsonable() for e in events]},
        ensure_ascii=False,
    )


@router.get("/changes")
def http_changes(
    since_seq: int = Query(0, ge=0),
    limit: int = Query(2000, ge=1, le=20000),
):
    """
    Oplog-backed catch-up via HTTP (for clients that can't WS, or for initial catch-up).
    Requires deps.oplog to be configured.
    """
    deps: BridgeDeps = router.dependency_overrides.get(BridgeDeps)  # type: ignore[assignment]
    if deps is None or deps.oplog is None:
        return JSONResponse({"error": "oplog not configured"}, status_code=503)

    evs = list(deps.oplog.iter_since(since_seq=since_seq, max_events=limit))
    last = evs[-1].seq if evs else since_seq
    return {"since_seq": since_seq, "last_seq": last, "events": [e.to_jsonable() for e in evs]}


@router.websocket("/changes/ws")
async def ws_changes(
    ws: WebSocket,
    since_seq: int = 0,
):
    """
    WebSocket live stream with buffer replay; falls back to "need_snapshot" if too far behind.
    """
    await ws.accept()

    deps: BridgeDeps = router.dependency_overrides.get(BridgeDeps)  # type: ignore[assignment]
    if deps is None:
        await ws.send_text(_need_snapshot("bridge not configured"))
        await ws.close(code=1011)
        return

    bus = deps.bus
    q = bus.subscribe()

    try:
        # 1) replay from in-memory buffer
        rr = bus.replay_into(q, since_seq=since_seq)
        if not rr.ok:
            # 1b) optional: replay from oplog if configured
            if deps.oplog is not None:
                # push oplog events into q (bounded)
                for ev in deps.oplog.iter_since(since_seq=since_seq, max_events=50_000):
                    try:
                        q.put_nowait(ev)
                    except asyncio.QueueFull:
                        await ws.send_text(_need_snapshot("client too slow during oplog replay"))
                        return
            else:
                await ws.send_text(_need_snapshot(rr.reason or "cannot replay"))
                return

        # 2) stream forever, batching a little to reduce overhead
        while True:
            first = await q.get()
            batch = [first]

            # small batch window (no timers; opportunistic)
            for _ in range(200):
                try:
                    batch.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(batch) == 1:
                await ws.send_text(_event_frame(batch[0]))
            else:
                await ws.send_text(_batch_frame(batch))

    except WebSocketDisconnect:
        return
    finally:
        bus.unsubscribe(q)
