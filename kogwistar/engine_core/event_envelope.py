"""Lossless event envelopes used by portable archive tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EntityEventEnvelope:
    """The immutable, portable representation of one authoritative event."""

    namespace: str
    seq: int
    event_id: str
    entity_kind: str
    entity_id: str
    op: str
    payload_json: str
    created_at: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EntityEventEnvelope":
        return cls(
            namespace=str(value["namespace"]),
            seq=int(value["seq"]),
            event_id=str(value["event_id"]),
            entity_kind=str(value["entity_kind"]),
            entity_id=str(value["entity_id"]),
            op=str(value["op"]),
            payload_json=str(value["payload_json"]),
            created_at=int(value["created_at"]),
        )
