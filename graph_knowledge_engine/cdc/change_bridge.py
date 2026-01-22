import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import logging
app = FastAPI()
subscribers: set[WebSocket] = set()
def _summarize_event(ev: dict) -> str:
    seq = ev.get("seq")
    op = ev.get("op")

    entity = ev.get("entity") or {}
    kind = entity.get("kind")
    eid = entity.get("id")
    gtype = entity.get("kg_graph_type")

    payload = ev.get("payload") or {}
    label = (
        payload.get("label")
        or payload.get("name")
        or payload.get("relation")
        or payload.get("type")
    )

    return (
        f"seq={seq} "
        f"op={op} "
        f"kind={kind} "
        f"id={eid} "
        f"label={label!r} "
        f"graph={gtype}"
    )
@app.post("/ingest")
async def ingest(ev: dict):
    dead = []
    log_msg = _summarize_event(ev)
    logging.info(log_msg)
    msg_json_dumped = json.dumps(ev)
    print(msg_json_dumped[:200])
    print(log_msg)
    for ws in subscribers:
        try:
            await ws.send_text(json.dumps(ev))
        except Exception:
            dead.append(ws)

    for ws in dead:
        subscribers.discard(ws)

    return {"ok": True}

@app.websocket("/changes/ws")
async def changes_ws(ws: WebSocket):
    await ws.accept()
    subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive, ignore content
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.discard(ws)
        
# uvicorn change_bridge:app --host 127.0.0.1 --port 8787
# uvicorn graph_knowledge_engine.cdc.change_bridge:app --host 127.0.0.1 --port 8787
"""python - <<'PY'
import asyncio,websockets
async def m(): async with websockets.connect('ws://localhost:8787/changes/ws') as w: async for x in w: print(x)
asyncio.run(m())
PY"""