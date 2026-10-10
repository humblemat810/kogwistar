from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from pydantic import BaseModel

from ..json_types import JsonValue

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

    def model_dump_entity_ref(self, *args: object, **kwargs: object) -> EntityRef:
        return cast(EntityRef, super().model_dump(*args, **kwargs))


# ---- Change event -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    seq: int
    op: Op
    ts_unix_ms: int

    entity: EntityRef | None = None
    payload: JsonValue | None = None

    # Optional provenance / debug fields
    run_id: str | None = None
    step_id: str | None = None

    # ---- Serialization ------------------------------------------------

    def to_jsonable(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
            "seq": self.seq,
            "op": self.op,
            "ts_unix_ms": self.ts_unix_ms,
            "entity": cast(JsonValue, self.entity),
            "payload": self.payload,
            "run_id": self.run_id,
            "step_id": self.step_id,
            },
        )

    @staticmethod
    def from_jsonable(d: Mapping[str, object]) -> ChangeEvent:
        entity_value = d.get("entity")
        if entity_value is None:
            entity = None
        elif isinstance(entity_value, Mapping):
            entity = cast(EntityRef, dict(entity_value))
        else:
            raise TypeError("entity must be a JSON object or null")

        seq_value = d.get("seq")
        ts_value = d.get("ts_unix_ms")
        if not isinstance(seq_value, (int, float, str)) or isinstance(seq_value, bool):
            raise TypeError("seq must be an integer-compatible JSON scalar")
        if not isinstance(ts_value, (int, float, str)) or isinstance(ts_value, bool):
            raise TypeError("ts_unix_ms must be an integer-compatible JSON scalar")

        return ChangeEvent(
            seq=int(seq_value),
            op=cast(Op, d["op"]),
            ts_unix_ms=int(ts_value),
            entity=entity,
            payload=cast(JsonValue | None, d.get("payload")),
            run_id=cast(str | None, d.get("run_id")),
            step_id=cast(str | None, d.get("step_id")),
        )
