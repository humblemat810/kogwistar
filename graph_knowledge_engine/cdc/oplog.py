from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .change_event import ChangeEvent


@dataclass(frozen=True, slots=True)
class OplogHeader:
    format: str = "kge-oplog"
    version: int = 1


class OplogWriter:
    """
    Append-only JSONL oplog.

    File layout:
      line 1: {"format":"kge-oplog","version":1}
      subsequent lines: ChangeEvent JSON objects (one per line)

    Replay:
      - simplest: scan and filter seq > since_seq
      - fast seek optional later with a sidecar index file
    """

    def __init__(self, path: Path, *, fsync: bool = False) -> None:
        self.path = path
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(OplogHeader().__dict__) + "\n")

    def append(self, ev: ChangeEvent) -> None:
        line = json.dumps(ev.to_jsonable(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())


class OplogReader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def iter_events(self) -> Iterator[ChangeEvent]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            first = f.readline()
            # ignore header for now
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield ChangeEvent.from_jsonable(d)

    def iter_since(self, *, since_seq: int, max_events: Optional[int] = None) -> Iterator[ChangeEvent]:
        n = 0
        for ev in self.iter_events():
            if ev.seq > since_seq:
                yield ev
                n += 1
                if max_events is not None and n >= max_events:
                    return
