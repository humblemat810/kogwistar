from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

import pytest

from scripts.postgres_authority_benchmark import WORKLOADS, Workload, _run_workload


class _MemoryStore:
    def __init__(self) -> None:
        self.events: dict[str, list[tuple[int, str, str, str, str]]] = {}
        self.projections: dict[str, list[dict[str, Any]]] = {}

    def transaction(self) -> Any:
        return nullcontext()

    def append_entity_event(
        self,
        *,
        namespace: str,
        event_id: str,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str,
    ) -> None:
        del event_id
        rows = self.events.setdefault(namespace, [])
        rows.append((len(rows) + 1, entity_kind, entity_id, op, payload_json))

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        *,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> None:
        self.projections.setdefault(namespace, []).append(
            {
                "key": key,
                "payload": payload,
                "last_authoritative_seq": last_authoritative_seq,
                "last_materialized_seq": last_materialized_seq,
                "projection_schema_version": projection_schema_version,
                "materialization_status": materialization_status,
            }
        )

    def iter_entity_events(
        self, *, namespace: str, from_seq: int, batch_size: int
    ) -> list[tuple[int, str, str, str, str]]:
        return self.events.get(namespace, [])[from_seq - 1 : from_seq - 1 + batch_size]

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        return sorted(self.projections.get(namespace, []), key=lambda row: row["key"])


@pytest.mark.parametrize("workload", WORKLOADS)
def test_single_workload_runner_produces_matching_digest(workload: Workload) -> None:
    python = _MemoryStore()
    rust = _MemoryStore()

    python_metrics, python_digest = _run_workload(
        python,
        workload=workload,
        suffix="sample-python",
        items=3,
        canonical_prefix="sample",
    )
    rust_metrics, rust_digest = _run_workload(
        rust,
        workload=workload,
        suffix="sample-rust",
        items=3,
        canonical_prefix="sample",
    )

    assert python_metrics["operations"] == rust_metrics["operations"] == 3
    assert python_digest == rust_digest


def test_event_digest_rejects_wrong_payload() -> None:
    store = _MemoryStore()
    _run_workload(
        store,
        workload="event_append_individual",
        suffix="sample",
        items=1,
    )
    namespace = "sample-event_append_individual"
    row = store.events[namespace][0]
    store.events[namespace][0] = (*row[:4], json.dumps({"id": "forged"}))

    with pytest.raises(RuntimeError, match="event payload/ordering mismatch"):
        _run_workload(
            store,
            workload="event_append_individual",
            suffix="sample",
            items=1,
        )
