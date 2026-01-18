import queue
import threading
import json
import time
import requests

class DebugEventProducer:
    def __init__(self, bridge_url: str, *, max_queue: int = 5000):
        self.bridge_url = bridge_url.rstrip("/")
        self.q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, event: dict) -> None:
        # NEVER block engine
        try:
            self.q.put_nowait(event)
        except queue.Full:
            # drop silently (debug-only channel)
            pass

    def _run(self) -> None:
        session = requests.Session()
        endpoint = f"{self.bridge_url}/ingest"

        while True:
            ev = self.q.get()
            try:
                session.post(endpoint, json=ev, timeout=0.5)
            except Exception:
                # bridge down → drop event
                pass