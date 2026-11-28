from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext

import pytest

from scripts.native_graph_backend_benchmark import (
    build_backend_trace,
    build_backend_trace_document,
    PostgresGraphTraceAdapter,
    run_trace_sample,
    verify_backend_report,
)


class TraceAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def write_record(self, operation: dict[str, object]) -> None:
        self.calls.append("write_record")

    def read_event(self, operation: dict[str, object]) -> dict[str, object]:
        self.calls.append("read_event")
        return {
            "node_id": operation["expected_entity_id"],
            "edge_id": operation["expected_edge_id"],
            "source_id": operation["expected_source_id"],
            "target_id": operation["expected_entity_id"],
        }

    def read_projection(self, operation: dict[str, object]) -> dict[str, object]:
        self.calls.append("read_projection")
        return {
            "node_id": operation["expected_entity_id"],
            "edge_id": operation["expected_edge_id"],
            "source_id": operation["expected_source_id"],
            "target_id": operation["expected_entity_id"],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "state_digest": "same-state",
            "canonical_event_count": 1,
            "projection_converged": True,
            "peak_rss_bytes": 1024,
        }


def test_run_trace_sample_executes_each_operation_and_records_native_evidence() -> None:
    adapter = TraceAdapter()
    trace = [
        {
            "operation": "write_record", "index": 0, "expected_entity_id": "node-0",
            "expected_edge_id": "edge-0", "expected_source_id": "node-root",
        },
        {
            "operation": "read_event", "index": 0, "expected_entity_id": "node-0",
            "expected_edge_id": "edge-0", "expected_source_id": "node-root",
        },
        {
            "operation": "read_projection", "index": 0, "expected_entity_id": "node-0",
            "expected_edge_id": "edge-0", "expected_source_id": "node-root",
        },
    ]

    sample = run_trace_sample(
        adapter,
        trace,
        trace_id="trace-01",
        trace_sha256="trace-sha",
        resource_limits_verified=True,
        ffi_calls=0,
    )

    assert adapter.calls == ["write_record", "read_event", "read_projection"]
    assert sample["trace_id"] == "trace-01"
    assert sample["trace_sha256"] == "trace-sha"
    assert sample["canonical_event_count"] == 1
    assert sample["state_digest"] == "same-state"
    assert sample["resource_limits_verified"] is True
    assert sample["peak_rss_bytes"] == 1024
    assert sample["ffi_calls"] == 0
    assert set(sample["operation_p95_ms"]) == {"write_record", "read_event", "read_projection"}
    assert sample["elapsed_ms"] > 0
    assert sample["throughput_ops_per_sec"] > 0


def test_run_trace_sample_fails_when_read_after_write_result_differs() -> None:
    class WrongReadAdapter(TraceAdapter):
        def read_event(self, operation: dict[str, object]) -> dict[str, object]:
            self.calls.append("read_event")
            return {"node_id": "wrong-node"}

    with pytest.raises(AssertionError, match="read_event"):
        run_trace_sample(
            WrongReadAdapter(),
            [
                {"operation": "write_record", "index": 0, "expected_entity_id": "node-0"},
                {"operation": "read_event", "index": 0, "expected_entity_id": "node-0"},
            ],
            trace_id="trace-01",
            trace_sha256="trace-sha",
            resource_limits_verified=True,
            ffi_calls=0,
        )


@pytest.mark.parametrize("operation_name", ["read_event", "read_projection"])
def test_run_trace_sample_rejects_graph_topology_mismatch(operation_name: str) -> None:
    class WrongTopologyAdapter(TraceAdapter):
        def read_event(self, operation: dict[str, object]) -> dict[str, object]:
            self.calls.append("read_event")
            return {
                "node_id": operation["expected_entity_id"],
                "edge_id": operation["expected_edge_id"],
                "source_id": "wrong-source",
                "target_id": operation["expected_entity_id"],
            }

        def read_projection(self, operation: dict[str, object]) -> dict[str, object]:
            self.calls.append("read_projection")
            return {
                "node_id": operation["expected_entity_id"],
                "edge_id": operation["expected_edge_id"],
                "source_id": "wrong-source",
                "target_id": operation["expected_entity_id"],
            }

    trace = {
        "operation": operation_name,
        "index": 0,
        "expected_entity_id": "node-0",
        "expected_edge_id": "edge-0",
        "expected_source_id": "node-root",
    }
    with pytest.raises(AssertionError, match="source_id"):
        run_trace_sample(
            WrongTopologyAdapter(),
            [trace],
            trace_id="trace-01",
            trace_sha256="trace-sha",
            resource_limits_verified=True,
            ffi_calls=0,
        )


def test_run_trace_sample_rejects_unknown_operation_before_backend_call() -> None:
    adapter = TraceAdapter()
    with pytest.raises(ValueError, match="unsupported"):
        run_trace_sample(
            adapter,
            [{"operation": "skip_write", "index": 0, "expected_entity_id": "node-0"}],
            trace_id="trace-01",
            trace_sha256="trace-sha",
            resource_limits_verified=True,
            ffi_calls=0,
        )
    assert adapter.calls == []


def test_build_backend_trace_is_repeatable_and_contains_graph_and_vector_payloads() -> None:
    first = build_backend_trace(2)
    second = build_backend_trace(2)

    assert first == second
    assert [row["operation"] for row in first] == [
        "write_record", "read_event", "read_projection",
        "write_record", "read_event", "read_projection",
    ]
    write = first[0]
    assert write["node_payload"]["id"] == "mixed-node-00000"
    assert write["edge_payload"]["id"] == "mixed-edge-00000"
    assert write["edge_payload"]["source_ids"] == ["mixed-node-root"]
    assert len(write["vector"]) == 384
    assert all(isinstance(value, float) for value in write["vector"])


def test_build_backend_trace_rejects_empty_workload() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_backend_trace(0)


def test_backend_trace_document_is_deterministic_and_has_a_shared_root() -> None:
    first = build_backend_trace_document(2)
    second = build_backend_trace_document(2)

    assert first == second
    assert first["schema"] == "kogwistar-native-graph-backend-trace/v1"
    assert first["trace_sha256"]
    assert first["root_payload"]["id"] == "mixed-node-root"
    assert len(first["operations"]) == 6
    assert first["operations"][0]["node_payload"]["id"] == "mixed-node-00000"


def test_backend_trace_document_rejects_empty_workload() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_backend_trace_document(0)


def test_python_postgres_trace_adapter_matches_shared_event_and_projection_contract() -> None:
    class FakeMeta:
        def __init__(self) -> None:
            self.events: list[tuple[int, str, str, str, str]] = []

        def transaction(self) -> object:
            return nullcontext()

        def append_entity_event(self, **values: str) -> int:
            seq = len(self.events) + 1
            self.events.append(
                (
                    seq,
                    values["entity_kind"],
                    values["entity_id"],
                    values["event_id"],
                    values["payload_json"],
                )
            )
            return seq

        def iter_entity_events(
            self, *, namespace: str, from_seq: int, batch_size: int, to_seq: int | None = None
        ) -> list[tuple[int, str, str, str, str]]:
            del namespace, batch_size
            return [
                row
                for row in self.events
                if row[0] >= from_seq and (to_seq is None or row[0] <= to_seq)
            ]

    class FakeVectorBackend:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, dict[str, object]]] = {"node": {}, "edge": {}}
            self.endpoint_deletes: list[dict[str, object]] = []

        def _upsert(
            self,
            kind: str,
            ids: list[str],
            documents: list[str],
            metadatas: list[dict[str, object]],
            embeddings: list[list[float]] | None = None,
        ) -> None:
            for index, entity_id in enumerate(ids):
                self.rows[kind][entity_id] = {
                    "document": documents[index],
                    "metadata": metadatas[index],
                    "embedding": embeddings[index] if embeddings else None,
                }

        def node_upsert(self, **values: object) -> None:
            self._upsert("node", **values)  # type: ignore[arg-type]

        def edge_upsert(self, **values: object) -> None:
            self._upsert("edge", **values)  # type: ignore[arg-type]

        def edge_endpoints_delete(self, **values: object) -> None:
            self.endpoint_deletes.append(values)

        def _get(self, kind: str, ids: list[str]) -> dict[str, object]:
            rows = [self.rows[kind][entity_id] for entity_id in ids if entity_id in self.rows[kind]]
            return {
                "ids": [entity_id for entity_id in ids if entity_id in self.rows[kind]],
                "documents": [row["document"] for row in rows],
                "metadatas": [row["metadata"] for row in rows],
            }

        def node_get(self, *, ids: list[str], include: list[str]) -> dict[str, object]:
            del include
            return self._get("node", ids)

        def edge_get(self, *, ids: list[str], include: list[str]) -> dict[str, object]:
            del include
            return self._get("edge", ids)

    trace = build_backend_trace_document(1)
    operation = trace["operations"][0]
    meta = FakeMeta()
    vector_backend = FakeVectorBackend()
    adapter = PostgresGraphTraceAdapter(
        meta,
        vector_backend,
        namespace="benchmark-native-graph",
        trace_id=trace["trace_id"],
    )

    adapter.seed_root(trace["root_payload"])
    adapter.write_record(operation)
    event_result = adapter.read_event(trace["operations"][1])
    projection_result = adapter.read_projection(trace["operations"][2])
    snapshot = adapter.snapshot()

    assert [row[1:3] for row in meta.events] == [
        ("node", "mixed-node-root"),
        ("node", "mixed-node-00000"),
        ("edge", "mixed-edge-00000"),
    ]
    assert event_result == projection_result == {
        "node_id": "mixed-node-00000",
        "edge_id": "mixed-edge-00000",
        "source_id": "mixed-node-root",
        "target_id": "mixed-node-00000",
    }
    payloads = {
        "node:mixed-node-root": trace["root_payload"],
        "node:mixed-node-00000": operation["node_payload"],
        "edge:mixed-edge-00000": operation["edge_payload"],
    }
    expected_digest = hashlib.sha256(
        json.dumps(payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert snapshot["canonical_event_count"] == 3
    assert snapshot["projection_converged"] is True
    assert snapshot["state_digest"] == expected_digest
    assert vector_backend.endpoint_deletes == [{"where": {"edge_id": "mixed-edge-00000"}}]


def test_python_trace_adapter_supports_sqlite_metadata_with_chroma_projection(tmp_path) -> None:
    from kogwistar.engine_core.engine_sqlite import EngineSQLite

    class FakeChroma:
        def __init__(self) -> None:
            self.records: dict[str, dict[str, dict[str, object]]] = {
                "node": {},
                "edge": {},
            }

        def upsert(self, kind: str, **values: object) -> None:
            ids = values["ids"]
            docs = values["documents"]
            metadata = values["metadatas"]
            assert isinstance(ids, list) and isinstance(docs, list)
            assert isinstance(metadata, list)
            for index, entity_id in enumerate(ids):
                self.records[kind][entity_id] = {
                    "document": docs[index],
                    "metadata": metadata[index],
                }

        def get(self, kind: str, *, ids: list[str], include: list[str]) -> dict[str, object]:
            del include
            found = [entity_id for entity_id in ids if entity_id in self.records[kind]]
            return {
                "ids": found,
                "documents": [self.records[kind][entity_id]["document"] for entity_id in found],
                "metadatas": [self.records[kind][entity_id]["metadata"] for entity_id in found],
            }

        def node_upsert(self, **values: object) -> None:
            self.upsert("node", **values)

        def edge_upsert(self, **values: object) -> None:
            self.upsert("edge", **values)

        def node_get(self, **values: object) -> dict[str, object]:
            return self.get("node", **values)  # type: ignore[arg-type]

        def edge_get(self, **values: object) -> dict[str, object]:
            return self.get("edge", **values)  # type: ignore[arg-type]

        def edge_endpoints_delete(self, **values: object) -> None:
            del values

    trace = build_backend_trace_document(1)
    metadata = EngineSQLite(tmp_path)
    metadata.ensure_initialized()
    chroma = FakeChroma()
    adapter = PostgresGraphTraceAdapter(
        metadata,
        chroma,
        namespace="benchmark-native-graph",
        trace_id=trace["trace_id"],
    )

    adapter.seed_root(trace["root_payload"])
    adapter.write_record(trace["operations"][0])
    snapshot = adapter.snapshot()
    assert adapter.read_event(trace["operations"][1]) == adapter.read_projection(
        trace["operations"][2]
    )

    expected_payloads = {
        "node:mixed-node-root": trace["root_payload"],
        "node:mixed-node-00000": trace["operations"][0]["node_payload"],
        "edge:mixed-edge-00000": trace["operations"][0]["edge_payload"],
    }
    expected_digest = hashlib.sha256(
        json.dumps(expected_payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert snapshot["canonical_event_count"] == 3
    assert snapshot["projection_converged"] is True
    assert snapshot["state_digest"] == expected_digest


@pytest.fixture
def valid_report() -> dict[str, object]:
    sample = {
        "trace_id": "trace-01",
        "trace_sha256": "trace-sha",
        "state_digest": "graph-sha",
        "canonical_event_count": 6,
        "projection_converged": True,
        "elapsed_ms": 100.0,
        "throughput_ops_per_sec": 60.0,
        "peak_rss_bytes": 1024,
        "operation_p95_ms": {"write": 2.0, "read": 1.0},
        "resource_limits_verified": True,
        "ffi_calls": 0,
    }
    return {
        "schema": "kogwistar-native-graph-backend-benchmark/v1",
        "backend": "postgres_pgvector",
        "samples_per_mode": 2,
        "warmup_samples_per_mode": 1,
        "trace": {"trace_id": "trace-01", "trace_sha256": "trace-sha"},
        "modes": {
            mode: {
                "warmup": copy.deepcopy(sample),
                "samples": [copy.deepcopy(sample), copy.deepcopy(sample)],
            }
            for mode in ("python", "rust_native")
        },
    }


def test_backend_report_accepts_equivalent_python_and_rust_traces(valid_report) -> None:
    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is True
    assert result["performance_gate"]["passed"] is True


@pytest.mark.parametrize("backend", ["sqlite_event_projection", "chroma_sqlite_metadata"])
def test_backend_report_rejects_backend_identity_mismatch(valid_report, backend) -> None:
    result = verify_backend_report(
        valid_report,
        expected_backend=backend,
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("backend" in error for error in result["errors"])


def test_backend_report_rejects_different_trace_identity(valid_report) -> None:
    valid_report["modes"]["rust_native"]["samples"][0]["trace_sha256"] = "other"

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("trace" in error for error in result["errors"])


def test_chroma_report_requires_eventual_projection_convergence(valid_report) -> None:
    valid_report["backend"] = "chroma_sqlite_metadata"
    valid_report["modes"]["rust_native"]["samples"][0]["projection_converged"] = False

    result = verify_backend_report(
        valid_report,
        expected_backend="chroma_sqlite_metadata",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("converg" in error for error in result["errors"])


def test_chroma_backend_report_accepts_equivalent_converged_traces(valid_report) -> None:
    valid_report["backend"] = "chroma_sqlite_metadata"

    result = verify_backend_report(
        valid_report,
        expected_backend="chroma_sqlite_metadata",
        expected_samples=2,
    )

    assert result["passed"] is True
    assert result["performance_gate"]["passed"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ffi_calls", 1, "ffi"),
        ("resource_limits_verified", False, "resource"),
        ("state_digest", "different", "digest"),
    ],
)
def test_backend_report_rejects_invalid_native_or_parity_evidence(
    valid_report, field, value, message
) -> None:
    valid_report["modes"]["rust_native"]["samples"][0][field] = value

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any(message in error.lower() for error in result["errors"])


def test_backend_report_applies_declared_throughput_and_tail_thresholds(valid_report) -> None:
    for sample in valid_report["modes"]["rust_native"]["samples"]:
        sample["throughput_ops_per_sec"] = 50.0
        sample["operation_p95_ms"]["write"] = 2.3

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert result["performance_gate"]["passed"] is False
    assert any("throughput" in error for error in result["errors"])
    assert any("p95" in error for error in result["errors"])


def test_backend_report_rejects_rust_peak_rss_above_declared_ratio(valid_report) -> None:
    for sample in valid_report["modes"]["rust_native"]["samples"]:
        sample["peak_rss_bytes"] = 1127

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert result["performance_gate"]["passed"] is False
    assert result["performance_gate"]["peak_rss_ratio"] > 1.10
    assert any("rss" in error.lower() for error in result["errors"])


def test_backend_report_rejects_missing_peak_rss_measurement(valid_report) -> None:
    del valid_report["modes"]["rust_native"]["samples"][0]["peak_rss_bytes"]

    result = verify_backend_report(
        valid_report,
        expected_backend="postgres_pgvector",
        expected_samples=2,
    )

    assert result["passed"] is False
    assert any("rss" in error.lower() for error in result["errors"])
