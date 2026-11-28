"""Verification contracts for resource-matched native graph backend traces."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping
from time import perf_counter, process_time
from typing import Any, Protocol


MIN_RUST_THROUGHPUT_RATIO = 0.95
MAX_RUST_P95_LATENCY_RATIO = 1.10
MAX_RUST_PEAK_RSS_RATIO = 1.10
BACKENDS = frozenset({"postgres_pgvector", "chroma_sqlite_metadata"})
MODES = frozenset({"python", "rust_native"})

try:
    from .mixed_workload import (
        VECTOR_DIMENSION,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        build_read_after_write_trace,
        normalized_vector_digest,
    )
except ImportError:
    from mixed_workload import (
        VECTOR_DIMENSION,
        build_graph_payloads,
        build_root_graph_payload,
        build_normalized_vector,
        build_read_after_write_trace,
        normalized_vector_digest,
    )


class TraceBackendAdapter(Protocol):
    def write_record(self, operation: Mapping[str, Any]) -> None: ...

    def read_event(self, operation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def read_projection(self, operation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def snapshot(self) -> Mapping[str, Any]: ...


class PostgresGraphTraceAdapter:
    """Python graph trace adapter over EnginePostgresMetaStore + PgVectorBackend."""

    def __init__(
        self,
        metadata_store: Any,
        vector_backend: Any,
        *,
        namespace: str,
        trace_id: str,
        eventual_projection: bool = False,
    ) -> None:
        if not namespace or not trace_id:
            raise ValueError("trace namespace and id are required")
        self.metadata_store = metadata_store
        self.vector_backend = vector_backend
        self.namespace = namespace
        self.trace_id = trace_id
        self.eventual_projection = eventual_projection
        self._node_ids: set[str] = set()
        self._edge_ids: set[str] = set()

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _append_entity_event(
        self,
        *,
        kind: str,
        entity_id: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if payload.get("id") != entity_id:
            raise ValueError(f"{kind} payload id differs from requested id")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{kind} payload metadata must be an object")
        payload_json = self._canonical_json(payload)
        self.metadata_store.append_entity_event(
            namespace=self.namespace,
            event_id=f"{self.trace_id}:{kind}:{entity_id}",
            entity_kind=kind,
            entity_id=entity_id,
            op="ADD",
            payload_json=payload_json,
        )
        return payload_json, metadata

    def _project_entity(
        self,
        *,
        kind: str,
        entity_id: str,
        payload_json: str,
        metadata: dict[str, Any],
        vector: list[float] | None,
    ) -> None:
        upsert = (
            self.vector_backend.node_upsert
            if kind == "node"
            else self.vector_backend.edge_upsert
        )
        projection_metadata = {**metadata, "namespace": self.namespace}
        values: dict[str, Any] = {
            "ids": [entity_id],
            "documents": [payload_json],
            "metadatas": [projection_metadata],
        }
        if vector is not None:
            values["embeddings"] = [vector]
        upsert(**values)
        if kind == "edge":
            self.vector_backend.edge_endpoints_delete(where={"edge_id": entity_id})
        (self._node_ids if kind == "node" else self._edge_ids).add(entity_id)

    def _write_entity(
        self,
        *,
        kind: str,
        entity_id: str,
        payload: Mapping[str, Any],
        vector: list[float] | None,
    ) -> None:
        payload_json, metadata = self._append_entity_event(
            kind=kind, entity_id=entity_id, payload=payload
        )
        self._project_entity(
            kind=kind,
            entity_id=entity_id,
            payload_json=payload_json,
            metadata=metadata,
            vector=vector,
        )

    def seed_root(
        self,
        root_payload: Mapping[str, Any],
        *,
        vector: list[float] | None = None,
    ) -> None:
        root_id = root_payload.get("id")
        if not isinstance(root_id, str) or not root_id:
            raise ValueError("root payload id is required")
        if self.eventual_projection:
            with self.metadata_store.transaction():
                payload_json, metadata = self._append_entity_event(
                    kind="node", entity_id=root_id, payload=root_payload
                )
            self._project_entity(
                kind="node",
                entity_id=root_id,
                payload_json=payload_json,
                metadata=metadata,
                vector=vector,
            )
            return
        with self.metadata_store.transaction():
            self._write_entity(
                kind="node", entity_id=root_id, payload=root_payload, vector=vector
            )

    def write_record(self, operation: Mapping[str, Any]) -> None:
        node = operation.get("node_payload")
        edge = operation.get("edge_payload")
        vector = operation.get("vector")
        if not isinstance(node, Mapping) or not isinstance(edge, Mapping):
            raise ValueError("write_record requires node and edge payloads")
        if not isinstance(vector, list) or not vector:
            raise ValueError("write_record requires a non-empty precomputed vector")
        node_id = operation.get("expected_entity_id")
        edge_id = operation.get("expected_edge_id")
        if not isinstance(node_id, str) or not isinstance(edge_id, str):
            raise ValueError("write_record requires expected node and edge ids")
        if self.eventual_projection:
            with self.metadata_store.transaction():
                node_payload_json, node_metadata = self._append_entity_event(
                    kind="node", entity_id=node_id, payload=node
                )
                edge_payload_json, edge_metadata = self._append_entity_event(
                    kind="edge", entity_id=edge_id, payload=edge
                )
            self._project_entity(
                kind="node",
                entity_id=node_id,
                payload_json=node_payload_json,
                metadata=node_metadata,
                vector=vector,
            )
            self._project_entity(
                kind="edge",
                entity_id=edge_id,
                payload_json=edge_payload_json,
                metadata=edge_metadata,
                vector=vector,
            )
            return
        with self.metadata_store.transaction():
            self._write_entity(kind="node", entity_id=node_id, payload=node, vector=vector)
            self._write_entity(kind="edge", entity_id=edge_id, payload=edge, vector=vector)

    def read_event(self, operation: Mapping[str, Any]) -> Mapping[str, Any]:
        index = operation.get("index")
        if not isinstance(index, int) or index < 0:
            raise ValueError("read_event requires a non-negative trace index")
        first_seq = 2 + 2 * index
        rows = list(
            self.metadata_store.iter_entity_events(
                namespace=self.namespace,
                from_seq=first_seq,
                to_seq=first_seq + 1,
                batch_size=2,
            )
        )
        if len(rows) != 2:
            raise AssertionError("read_event did not return node and edge events")
        node_row, edge_row = rows
        node_payload = json.loads(node_row[4])
        edge_payload = json.loads(edge_row[4])
        expected_node = operation.get("expected_entity_id")
        expected_edge = operation.get("expected_edge_id")
        expected_source = operation.get("expected_source_id")
        if (
            (node_row[1], node_row[2], node_payload.get("id"))
            != ("node", expected_node, expected_node)
            or (edge_row[1], edge_row[2], edge_payload.get("id"))
            != ("edge", expected_edge, expected_edge)
            or edge_payload.get("source_ids") != [expected_source]
            or edge_payload.get("target_ids") != [expected_node]
        ):
            raise AssertionError("canonical event read differs from shared graph trace")
        return {
            "node_id": expected_node,
            "edge_id": expected_edge,
            "source_id": expected_source,
            "target_id": expected_node,
        }

    def _read_payload(self, kind: str, entity_id: str) -> dict[str, Any]:
        getter = (
            self.vector_backend.node_get
            if kind == "node"
            else self.vector_backend.edge_get
        )
        result = getter(ids=[entity_id], include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        if not ids or ids[0] != entity_id or not documents or not isinstance(documents[0], str):
            raise AssertionError(f"{kind} projection {entity_id!r} is missing")
        payload = json.loads(documents[0])
        if not isinstance(payload, dict) or payload.get("id") != entity_id:
            raise AssertionError(f"{kind} projection {entity_id!r} has invalid document")
        return payload

    def read_projection(self, operation: Mapping[str, Any]) -> Mapping[str, Any]:
        node_id = operation.get("expected_entity_id")
        edge_id = operation.get("expected_edge_id")
        source_id = operation.get("expected_source_id")
        if not all(isinstance(value, str) for value in (node_id, edge_id, source_id)):
            raise ValueError("read_projection requires graph ids")
        node_payload = self._read_payload("node", node_id)
        edge_payload = self._read_payload("edge", edge_id)
        if edge_payload.get("source_ids") != [source_id] or edge_payload.get(
            "target_ids"
        ) != [node_id]:
            raise AssertionError("edge projection endpoints differ from shared graph trace")
        return {
            "node_id": node_id,
            "edge_id": edge_id,
            "source_id": source_id,
            "target_id": node_id,
        }

    def snapshot(self) -> Mapping[str, Any]:
        records: dict[str, Any] = {}
        converged = True
        for kind, entity_ids in (
            ("node", sorted(self._node_ids)),
            ("edge", sorted(self._edge_ids)),
        ):
            for entity_id in entity_ids:
                try:
                    payload = self._read_payload(kind, entity_id)
                except (AssertionError, KeyError, ValueError, json.JSONDecodeError):
                    converged = False
                    continue
                records[f"{kind}:{entity_id}"] = payload
        events = list(
            self.metadata_store.iter_entity_events(
                namespace=self.namespace,
                from_seq=1,
                batch_size=max(1, len(self._node_ids) + len(self._edge_ids) + 1),
            )
        )
        serialized = json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "state_digest": hashlib.sha256(serialized).hexdigest(),
            "canonical_event_count": len(events),
            "peak_rss_bytes": _process_peak_rss_bytes(),
            "projection_converged": converged
            and len(records) == len(self._node_ids) + len(self._edge_ids),
        }


def _process_peak_rss_bytes() -> int | None:
    """Return process high-water RSS where the OS exposes it; fail closed otherwise."""
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)]


def build_backend_trace(items: int) -> list[dict[str, Any]]:
    """Build one repeatable graph trace before either implementation is timed."""
    if items < 1:
        raise ValueError("items must be positive")
    trace: list[dict[str, Any]] = []
    for operation in build_read_after_write_trace(items):
        row = dict(operation)
        if operation["operation"] == "write_record":
            index = operation["index"]
            vector = build_normalized_vector(index)
            node, edge = build_graph_payloads(
                index,
                vector_sha256=normalized_vector_digest(vector),
            )
            row.update(
                {
                    "node_payload": node,
                    "edge_payload": edge,
                    "vector": vector,
                    "vector_dimension": VECTOR_DIMENSION,
                }
            )
        trace.append(row)
    return trace


def build_backend_trace_document(items: int) -> dict[str, Any]:
    """Serialize one versioned trace contract shared by Python and Rust runners."""
    operations = build_backend_trace(items)
    body: dict[str, Any] = {
        "schema": "kogwistar-native-graph-backend-trace/v1",
        "trace_id": f"native-graph-{items}",
        "root_payload": build_root_graph_payload(),
        "operations": operations,
    }
    body["trace_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return body


def run_trace_sample(
    adapter: TraceBackendAdapter,
    trace: list[Mapping[str, Any]],
    *,
    trace_id: str,
    trace_sha256: str,
    resource_limits_verified: bool,
    ffi_calls: int,
) -> dict[str, Any]:
    """Execute fixed read-after-write operations and collect measured evidence."""
    if not trace:
        raise ValueError("trace must contain operations")
    if not trace_id or not trace_sha256:
        raise ValueError("trace identity and digest are required")
    if not resource_limits_verified or ffi_calls < 0:
        raise ValueError("verified resource limits and non-negative FFI count are required")

    elapsed_by_operation: dict[str, list[float]] = {}
    wall_start = perf_counter()
    cpu_start = process_time()
    for operation in trace:
        name = operation.get("operation")
        operation_start = perf_counter()
        if name == "write_record":
            adapter.write_record(operation)
        elif name == "read_event":
            actual = adapter.read_event(operation)
            _verify_graph_read(actual, operation, operation_name="read_event")
        elif name == "read_projection":
            actual = adapter.read_projection(operation)
            _verify_graph_read(actual, operation, operation_name="read_projection")
        else:
            raise ValueError(f"unsupported trace operation: {name!r}")
        elapsed_by_operation.setdefault(str(name), []).append(
            (perf_counter() - operation_start) * 1000
        )

    snapshot = adapter.snapshot()
    elapsed_ms = (perf_counter() - wall_start) * 1000
    cpu_ms = (process_time() - cpu_start) * 1000
    state_digest = snapshot.get("state_digest")
    event_count = snapshot.get("canonical_event_count")
    if not isinstance(state_digest, str) or not state_digest:
        raise ValueError("backend snapshot must provide a canonical state digest")
    if not isinstance(event_count, int) or event_count < 0:
        raise ValueError("backend snapshot must provide a non-negative event count")
    peak_rss_bytes = snapshot.get("peak_rss_bytes")
    if not isinstance(peak_rss_bytes, int) or peak_rss_bytes <= 0:
        raise ValueError("backend snapshot must provide a positive peak RSS measurement")
    projection_converged = snapshot.get("projection_converged")
    if not isinstance(projection_converged, bool):
        raise ValueError("backend snapshot must report projection convergence")

    return {
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "state_digest": state_digest,
        "canonical_event_count": event_count,
        "peak_rss_bytes": peak_rss_bytes,
        "projection_converged": projection_converged,
        "elapsed_ms": elapsed_ms,
        "cpu_ms": cpu_ms,
        "throughput_ops_per_sec": len(trace) * 1000 / elapsed_ms,
        "operation_p95_ms": {
            name: _p95(values) for name, values in sorted(elapsed_by_operation.items())
        },
        "resource_limits_verified": resource_limits_verified,
        "ffi_calls": ffi_calls,
    }


def _verify_graph_read(
    actual: Mapping[str, Any],
    operation: Mapping[str, Any],
    *,
    operation_name: str,
) -> None:
    expected = {
        "node_id": operation.get("expected_entity_id"),
        "edge_id": operation.get("expected_edge_id"),
        "source_id": operation.get("expected_source_id"),
        "target_id": operation.get("expected_entity_id"),
    }
    for field, expected_value in expected.items():
        if actual.get(field) != expected_value:
            raise AssertionError(
                f"{operation_name} {field} returned {actual.get(field)!r}, "
                f"expected {expected_value!r}"
            )


def verify_backend_report(
    report: Mapping[str, Any],
    *,
    expected_backend: str,
    expected_samples: int,
) -> dict[str, Any]:
    """Fail closed unless both pure modes prove one equivalent backend trace."""

    errors: list[str] = []
    if expected_backend not in BACKENDS:
        errors.append(f"unsupported expected backend: {expected_backend}")
    if report.get("schema") != "kogwistar-native-graph-backend-benchmark/v1":
        errors.append("report schema is missing or unsupported")
    if report.get("backend") != expected_backend:
        errors.append("backend identity differs from requested backend")
    if report.get("samples_per_mode") != expected_samples or expected_samples < 2:
        errors.append("sample count differs from requested benchmark")
    if report.get("warmup_samples_per_mode") != 1:
        errors.append("exactly one warm-up per mode is required")

    trace = report.get("trace")
    if not isinstance(trace, Mapping):
        errors.append("trace identity is missing")
        trace_id = trace_sha = None
    else:
        trace_id = trace.get("trace_id")
        trace_sha = trace.get("trace_sha256")
        if not isinstance(trace_id, str) or not trace_id:
            errors.append("trace id is missing")
        if not isinstance(trace_sha, str) or not trace_sha:
            errors.append("trace digest is missing")

    modes = report.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != MODES:
        errors.append("report must contain exactly Python and Rust-native modes")
        modes = {}

    valid_samples: dict[str, list[Mapping[str, Any]]] = {}
    digests: set[str] = set()
    operation_p95: dict[str, dict[str, list[float]]] = {
        mode: {} for mode in MODES
    }
    median_throughput: dict[str, float] = {}
    median_peak_rss: dict[str, float] = {}
    for mode in sorted(MODES):
        mode_report = modes.get(mode)
        if not isinstance(mode_report, Mapping):
            errors.append(f"{mode}: mode report is missing")
            continue
        samples = mode_report.get("samples")
        warmup = mode_report.get("warmup")
        if not isinstance(samples, list) or len(samples) != expected_samples:
            errors.append(f"{mode}: measured sample count differs")
            samples = []
        if not isinstance(warmup, Mapping):
            errors.append(f"{mode}: warm-up sample is missing")
            warmup = {}
        rows = [warmup, *(row for row in samples if isinstance(row, Mapping))]
        if len(rows) != expected_samples + 1:
            errors.append(f"{mode}: one or more samples are malformed")
        valid_samples[mode] = [row for row in rows if isinstance(row, Mapping)]

        throughputs: list[float] = []
        for sample_index, sample in enumerate(rows):
            label = "warm-up" if sample_index == 0 else f"sample {sample_index}"
            if sample.get("trace_id") != trace_id or sample.get("trace_sha256") != trace_sha:
                errors.append(f"{mode}/{label}: trace identity differs")
            digest = sample.get("state_digest")
            if not isinstance(digest, str) or not digest:
                errors.append(f"{mode}/{label}: state digest is missing")
            else:
                digests.add(digest)
            event_count = sample.get("canonical_event_count")
            if not isinstance(event_count, int) or event_count < 0:
                errors.append(f"{mode}/{label}: canonical event count is invalid")
            peak_rss_bytes = sample.get("peak_rss_bytes")
            if not isinstance(peak_rss_bytes, (int, float)) or peak_rss_bytes <= 0:
                errors.append(f"{mode}/{label}: peak RSS measurement is invalid")
            if expected_backend == "chroma_sqlite_metadata" and sample.get(
                "projection_converged"
            ) is not True:
                errors.append(f"{mode}/{label}: Chroma projection did not converge")
            if sample.get("resource_limits_verified") is not True:
                errors.append(f"{mode}/{label}: resource limits are not verified")
            ffi_calls = sample.get("ffi_calls")
            if mode == "rust_native" and ffi_calls != 0:
                errors.append(f"{mode}/{label}: Rust-native path reports nonzero FFI calls")

            elapsed = sample.get("elapsed_ms")
            throughput = sample.get("throughput_ops_per_sec")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                errors.append(f"{mode}/{label}: elapsed time is invalid")
            if not isinstance(throughput, (int, float)) or throughput <= 0:
                errors.append(f"{mode}/{label}: throughput is invalid")
            elif sample_index > 0:
                throughputs.append(float(throughput))

            latencies = sample.get("operation_p95_ms")
            if not isinstance(latencies, Mapping) or not latencies:
                errors.append(f"{mode}/{label}: operation p95 metrics are missing")
                continue
            for operation, value in latencies.items():
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"{mode}/{label}/{operation}: p95 latency is invalid")
                    continue
                if sample_index > 0:
                    operation_p95[mode].setdefault(str(operation), []).append(float(value))
        if throughputs:
            median_throughput[mode] = statistics.median(throughputs)
        rss_values = [
            float(row["peak_rss_bytes"])
            for row in rows
            if isinstance(row.get("peak_rss_bytes"), (int, float))
            and row["peak_rss_bytes"] > 0
        ]
        if len(rss_values) == expected_samples + 1:
            median_peak_rss[mode] = statistics.median(rss_values[1:])

    if len(digests) != 1:
        errors.append("Python/Rust trace state digests differ or are incomplete")
    if set(valid_samples) == MODES:
        python_counts = {
            row.get("canonical_event_count") for row in valid_samples["python"]
        }
        rust_counts = {
            row.get("canonical_event_count") for row in valid_samples["rust_native"]
        }
        if python_counts != rust_counts:
            errors.append("canonical event counts differ between modes")

    performance_errors: list[str] = []
    throughput_ratio: float | None = None
    python_rate = median_throughput.get("python")
    rust_rate = median_throughput.get("rust_native")
    if python_rate is None or rust_rate is None or python_rate <= 0:
        performance_errors.append("throughput comparison is incomplete")
    else:
        throughput_ratio = rust_rate / python_rate
        if throughput_ratio < MIN_RUST_THROUGHPUT_RATIO:
            performance_errors.append(
                f"throughput ratio {throughput_ratio:.3f} is below {MIN_RUST_THROUGHPUT_RATIO:.2f}"
            )

    python_operations = operation_p95.get("python", {})
    rust_operations = operation_p95.get("rust_native", {})
    if not python_operations or set(python_operations) != set(rust_operations):
        performance_errors.append("operation p95 comparison is incomplete")
    p95_ratios: dict[str, float | None] = {}
    for operation in sorted(set(python_operations) & set(rust_operations)):
        python_values = python_operations[operation]
        rust_values = rust_operations[operation]
        if len(python_values) != expected_samples or len(rust_values) != expected_samples:
            performance_errors.append(f"{operation}: operation p95 samples are incomplete")
            continue
        python_p95 = statistics.median(python_values)
        rust_p95 = statistics.median(rust_values)
        ratio = (
            1.0 if python_p95 == 0 and rust_p95 == 0
            else None if python_p95 == 0
            else rust_p95 / python_p95
        )
        p95_ratios[operation] = ratio
        if ratio is None or ratio > MAX_RUST_P95_LATENCY_RATIO:
            performance_errors.append(
                f"{operation}: p95 latency ratio exceeds {MAX_RUST_P95_LATENCY_RATIO:.2f}"
            )

    python_peak_rss = median_peak_rss.get("python")
    rust_peak_rss = median_peak_rss.get("rust_native")
    peak_rss_ratio = (
        None
        if python_peak_rss is None or rust_peak_rss is None or python_peak_rss <= 0
        else rust_peak_rss / python_peak_rss
    )
    if peak_rss_ratio is None or peak_rss_ratio > MAX_RUST_PEAK_RSS_RATIO:
        performance_errors.append(
            f"peak RSS ratio {peak_rss_ratio!r} exceeds {MAX_RUST_PEAK_RSS_RATIO:.2f} or is incomplete"
        )

    errors.extend(performance_errors)
    return {
        "passed": not errors,
        "errors": errors,
        "performance_gate": {
            "passed": not performance_errors,
            "throughput_ratio": throughput_ratio,
            "operation_p95_latency_ratios": p95_ratios,
            "peak_rss_ratio": peak_rss_ratio,
        },
    }
