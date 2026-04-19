from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CostLedger:
    workspace_id: str
    events: list[dict[str, object]] = field(default_factory=list)

    def add_event(self, *, kind: str, amount: int, source: str) -> dict[str, object]:
        evt = {
            "workspace_id": self.workspace_id,
            "kind": kind,
            "amount": int(amount),
            "source": source,
        }
        self.events.append(evt)
        return evt

    def ingest(self, event: Any) -> dict[str, object]:
        evt = {
            "workspace_id": self.workspace_id,
            "kind": str(getattr(event, "kind", "unknown")),
            "amount": int(getattr(event, "amount", 0) or 0),
            "source": str(getattr(event, "source", "unknown")),
            "unit": str(getattr(event, "unit", "")),
            "scope": str(getattr(event, "scope", "run")),
        }
        self.events.append(evt)
        return evt

    def snapshot(self) -> dict[str, object]:
        total_amount = 0
        by_kind: dict[str, int] = {}
        for evt in self.events:
            amt = int(evt.get("amount", 0) or 0)
            total_amount += amt
            kind = str(evt.get("kind") or "unknown")
            by_kind[kind] = by_kind.get(kind, 0) + 1
        return {
            "workspace_id": self.workspace_id,
            "event_count": len(self.events),
            "total_amount": total_amount,
            "by_kind": by_kind,
        }

    def history(self, *, limit: int | None = None) -> list[dict[str, object]]:
        rows = list(self.events)
        if limit is not None:
            rows = rows[-max(0, int(limit)) :]
        return rows
