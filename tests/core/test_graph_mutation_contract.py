from __future__ import annotations

import json
from pathlib import Path

import pytest

from kogwistar.engine_core import (
    GraphMutationContractError,
    validate_graph_mutation_batch,
)


def _fixture() -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "contracts/golden/graph-mutation-batch-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_python_graph_mutation_contract_accepts_rust_golden_fixture() -> None:
    batch = validate_graph_mutation_batch(_fixture())

    assert batch.contract_version == 1
    assert batch.atomic is True
    assert batch.mutations[0].record.embedding == [1.0, 0.0]
    assert batch.index_jobs == []


@pytest.mark.parametrize("mutation_change", [
    {"table": "gke_edges"},
    {"op": "UNKNOWN"},
    {"embedding_dim": 3},
    {"payload": ["not", "an", "object"]},
])
def test_python_graph_mutation_contract_rejects_semantically_invalid_mutations(
    mutation_change: dict[str, object],
) -> None:
    payload = _fixture()
    mutation = payload["mutations"][0]
    assert isinstance(mutation, dict)
    mutation.update(mutation_change)

    with pytest.raises(GraphMutationContractError):
        validate_graph_mutation_batch(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("contract_version", 2), ("atomic", False), ("mutations", [])],
)
def test_python_graph_mutation_contract_rejects_invalid_batch_header(
    field: str, value: object
) -> None:
    payload = _fixture()
    payload[field] = value

    with pytest.raises(GraphMutationContractError):
        validate_graph_mutation_batch(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace", "other-tenant"),
        ("op", "REPLACE"),
        ("max_retries", -1),
    ],
)
def test_python_graph_mutation_contract_rejects_invalid_index_intents(
    field: str, value: object
) -> None:
    payload = _fixture()
    payload["index_jobs"] = [
        {
            "job_id": "index-node-1",
            "namespace": "tenant-a",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "max_retries": 3,
        }
    ]
    jobs = payload["index_jobs"]
    assert isinstance(jobs, list)
    job = jobs[0]
    assert isinstance(job, dict)
    job[field] = value

    with pytest.raises(GraphMutationContractError):
        validate_graph_mutation_batch(payload)

