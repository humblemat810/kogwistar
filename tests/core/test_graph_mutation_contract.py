from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from kogwistar.engine_core import (
    GraphMutationContractError,
    validate_graph_mutation_batch,
    validate_graph_mutation_batch_v2,
)


def _fixture() -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "contracts/golden/graph-mutation-batch-v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_graph_mutation_shared_invalid_cases_are_rejected_by_python() -> None:
    root = Path(__file__).resolve().parents[2]
    invalid_path = root / "contracts/golden/graph-mutation-invalid-v1.json"
    invalid_fixture = json.loads(invalid_path.read_text(encoding="utf-8"))
    assert invalid_fixture["schema_version"] == 1

    for case_id in invalid_fixture["cases"]:
        payload = _fixture()
        if case_id == "payload_record_id_mismatch":
            payload["mutations"][0]["record"]["id"] = "different-node"
        elif case_id == "embedding_boolean":
            payload["mutations"][0]["record"]["embedding"] = [True, 0]
        elif case_id == "embedding_numeric_string":
            payload["mutations"][0]["record"]["embedding"] = ["1", 0]
        elif case_id == "embedding_non_finite":
            payload["mutations"][0]["record"]["embedding"] = [math.nan, 0]
        elif case_id == "embedding_dim_boolean":
            payload["mutations"][0]["embedding_dim"] = True
        elif case_id in {
            "index_retries_negative",
            "index_retries_above_postgres_integer",
            "index_retries_boolean",
        }:
            payload["index_jobs"] = [
                {
                    "job_id": "index-node-1",
                    "namespace": "tenant-a",
                    "entity_kind": "node",
                    "entity_id": "node-1",
                    "index_kind": "node_embedding",
                    "op": "UPSERT",
                    "max_retries": (
                        -1
                        if case_id == "index_retries_negative"
                        else 2_147_483_648
                        if case_id == "index_retries_above_postgres_integer"
                        else True
                    ),
                }
            ]
        else:
            raise AssertionError(f"unknown graph mutation invalid case: {case_id}")

        with pytest.raises(GraphMutationContractError):
            validate_graph_mutation_batch(payload)


def test_python_graph_mutation_contract_accepts_rust_golden_fixture() -> None:
    batch = validate_graph_mutation_batch(_fixture())

    assert batch.contract_version == 1
    assert batch.atomic is True
    assert batch.mutations[0].record.embedding == [1.0, 0.0]
    assert batch.index_jobs == []


def test_python_acl_aware_graph_batch_v2_matches_rust_fixture() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "contracts/golden/graph-mutation-batch-v2.json").read_text(
            encoding="utf-8"
        )
    )

    batch = validate_graph_mutation_batch_v2(payload)

    assert batch.contract_version == 2
    assert batch.default_acl.truth_graph == "conversation"
    assert batch.mutations[0].record.id == "node-1"
    with pytest.raises(GraphMutationContractError):
        validate_graph_mutation_batch(payload)


@pytest.mark.parametrize("truth_graph", ["", "   "])
def test_python_acl_aware_graph_batch_v2_rejects_empty_truth_graph(
    truth_graph: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "contracts/golden/graph-mutation-batch-v2.json").read_text(
            encoding="utf-8"
        )
    )
    payload["default_acl"]["truth_graph"] = truth_graph

    with pytest.raises(GraphMutationContractError, match="truth_graph"):
        validate_graph_mutation_batch_v2(payload)


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


def test_canonical_graph_mutation_rejects_physical_delete_operation() -> None:
    payload = _fixture()
    mutation = payload["mutations"][0]
    assert isinstance(mutation, dict)
    mutation["op"] = "DELETE"

    with pytest.raises(GraphMutationContractError):
        validate_graph_mutation_batch(payload)


def test_graph_mutation_rejects_payload_and_projection_identity_drift() -> None:
    payload = _fixture()
    mutation = payload["mutations"][0]
    assert isinstance(mutation, dict)
    mutation["record"]["id"] = "different-node"

    with pytest.raises(GraphMutationContractError, match="payload id"):
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
        ("max_retries", 2_147_483_648),
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


def test_python_graph_mutation_contract_accepts_postgres_integer_retry_limit() -> None:
    payload = _fixture()
    payload["index_jobs"] = [
        {
            "job_id": "index-node-1",
            "namespace": "tenant-a",
            "entity_kind": "node",
            "entity_id": "node-1",
            "index_kind": "node_embedding",
            "op": "UPSERT",
            "max_retries": 2_147_483_647,
        }
    ]

    batch = validate_graph_mutation_batch(payload)

    assert batch.index_jobs[0].max_retries == 2_147_483_647

