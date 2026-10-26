from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict


class EmbeddingJobProviderInput(TypedDict):
    """Minimum queue data an embedding batch adapter needs.

    Claim tokens, leases, retry counters, and acknowledgement state are
    intentionally excluded from provider-facing input.
    """

    schema_version: Literal[1]
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    coalesce_key: str
    op: str
    payload_json: str | None


def embedding_job_provider_input(job: object) -> EmbeddingJobProviderInput:
    """Project a claimed row into a non-authoritative embedding input."""

    def field(name: str) -> object:
        if isinstance(job, Mapping):
            return job.get(name)
        return getattr(job, name, None)

    values: dict[str, str] = {}
    for name in (
        "job_id",
        "namespace",
        "entity_kind",
        "entity_id",
        "index_kind",
        "coalesce_key",
        "op",
    ):
        value = field(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"claimed index job has invalid {name}")
        values[name] = value

    payload_json = field("payload_json")
    return {
        "schema_version": 1,
        **values,
        "payload_json": payload_json if isinstance(payload_json, str) else None,
    }
