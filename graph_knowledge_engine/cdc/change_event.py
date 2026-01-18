from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, TypedDict


Op = Literal[
    node.upsert,
    node.remove,
    edge.upsert,
    edge.remove,
    checkpoint,
    snapshot,  # optional if you ever want to send snapshots through the stream
]


class EntityRef(TypedDict, total=False)
    kind Literal[node, edge, checkpoint, other]
    id str
    short_id str


@dataclass(frozen=True, slots=True)
class ChangeEvent
    seq int
    op Op
    ts_unix_ms int

    entity Optional[EntityRef] = None
    payload Any = None

    # Optional provenancedebug fields
    run_id Optional[str] = None
    step_id Optional[str] = None

    def to_jsonable(self) - dict[str, Any]
        return {
            seq self.seq,
            op self.op,
            ts_unix_ms self.ts_unix_ms,
            entity self.entity,
            payload self.payload,
            run_id self.run_id,
            step_id self.step_id,
        }

    @staticmethod
    def from_jsonable(d Mapping[str, Any]) - ChangeEvent
        return ChangeEvent(
            seq=int(d[seq]),
            op=d[op],
            ts_unix_ms=int(d[ts_unix_ms]),
            entity=d.get(entity),
            payload=d.get(payload),
            run_id=d.get(run_id),
            step_id=d.get(step_id),
        )
