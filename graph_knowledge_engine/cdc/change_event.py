from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, TypedDict
from pydantic import BaseModel


# ---- Operation types -------------------------------------------------

Op = Literal[
    "node.upsert",  # tombstoning = delete, updating
    "node.remove",  # never delete in practise so far
    "doc.upsert",  # tombstoning = delete, updating
    "doc.remove",  # never delete in practise so far
    "edge.upsert",  # tombstoning = delete, updating
    "edge.remove",  # never delete in practise so far
    "search_index.upsert",  # search index entries updated
    "checkpoint",  # unused
    "snapshot",  # unused
]


# ---- Entity reference -------------------------------------------------


class EntityRef(TypedDict, total=False):
    kind: Literal["node", "edge", "doc_node", "search_index"]
    id: str
    kg_graph_type: str
    url: str


class EntityRefModel(BaseModel):
    kind: Literal["node", "edge", "doc_node", "search_index"]
    id: str
    kg_graph_type: str
    url: str | None

    def model_dump_entity_ref(self, *arg, **kwarg):
        return EntityRef(super().model_dump(*arg, **kwarg))


# ---- Change event -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    seq: int
    op: Op
    ts_unix_ms: int

    entity: Optional[EntityRef] = None
    payload: Any = None

    # Optional provenance / debug fields
    run_id: Optional[str] = None
    step_id: Optional[str] = None

    # ---- Serialization ------------------------------------------------

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "op": self.op,
            "ts_unix_ms": self.ts_unix_ms,
            "entity": self.entity,
            "payload": self.payload,
            "run_id": self.run_id,
            "step_id": self.step_id,
        }

    @staticmethod
    def from_jsonable(d: Mapping[str, Any]) -> ChangeEvent:
        return ChangeEvent(
            seq=int(d["seq"]),
            op=d["op"],  # type: ignore[arg-type]
            ts_unix_ms=int(d["ts_unix_ms"]),
            entity=d.get("entity"),
            payload=d.get("payload"),
            run_id=d.get("run_id"),
            step_id=d.get("step_id"),
        )
