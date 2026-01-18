import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()
subscribers: set[WebSocket] = set()

@app.post("/ingest")
async def ingest(ev: dict):
    dead = []
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