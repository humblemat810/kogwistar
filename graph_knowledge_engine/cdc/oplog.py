# knowledge_graph_engine/changes/oplog.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterator, Optional
from .change_event import ChangeEvent


class OplogWriter:
    def __init__(self, path: Path, *, fsync: bool = False):
        self.path = path
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.write_text(
                json.dumps({"format": "kge-oplog", "version": 1}) + "\n",
                encoding="utf-8",
            )

    def append(self, ev: ChangeEvent) -> None:
        line = json.dumps(ev.to_jsonable(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())


class OplogReader:
    def __init__(self, path: Path):
        self.path = path

    def iter_since(
        self, *, since_seq: int, limit: Optional[int] = None
    ) -> Iterator[ChangeEvent]:
        if not self.path.exists():
            return
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            _header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ev = ChangeEvent(
                    seq=int(d["seq"]),
                    op=d["op"],
                    ts_unix_ms=int(d["ts_unix_ms"]),
                    entity=d.get("entity"),
                    payload=d.get("payload"),
                    run_id=d.get("run_id"),
                    step_id=d.get("step_id"),
                )
                if ev.seq > since_seq:
                    yield ev
                    n += 1
                    if limit is not None and n >= limit:
                        return
