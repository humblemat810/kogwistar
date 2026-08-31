from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class JsonlEventSink:
    """Append event dictionaries to a JSONL file and optionally mirror downstream."""

    def __init__(self, *, jsonl_path: Path, downstream_sink: Any | None = None) -> None:
        self.jsonl_path = jsonl_path
        self.downstream_sink = downstream_sink
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        if self.downstream_sink is not None:
            self.downstream_sink.emit(event)
        line = json.dumps(
            {"observed_at_ms": int(time.time() * 1000), **dict(event)},
            default=str,
            sort_keys=True,
        )
        with self._lock:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def close(self) -> None:
        """Satisfy telemetry sink lifecycle; JSONL opens per write."""
        return None
