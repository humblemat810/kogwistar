"""Versioned deterministic traces shared by storage rewrite benchmarks."""

from __future__ import annotations

import json
import hashlib
import math
import struct
from pathlib import Path
from typing import Literal, TypedDict

from kogwistar.engine_core.models import Edge, Grounding, Node
from kogwistar.engine_core.span_models import Span


class WorkloadOperation(TypedDict):
    operation: Literal["write_record", "read_event", "read_projection"]
    index: int
    expected_entity_id: str
    expected_edge_id: str
    expected_source_id: str


class ConcurrentRead(TypedDict):
    reader_id: int
    visibility: Literal["before_commit", "after_commit"]
    minimum_revision: int
    maximum_revision: int


class ConcurrentStep(TypedDict):
    write_index: int
    reads: list[ConcurrentRead]


class ConcurrentPhase(TypedDict):
    phase: int
    steps: list[ConcurrentStep]


VECTOR_SEED = "kogwistar-mixed-rewrite-v1"
VECTOR_DIMENSION = 384
SOURCE_DOC_ID = "mixed-rewrite-source-v1"
ROOT_NODE_ID = "mixed-node-root"


def graph_node_id(index: int) -> str:
    if index < 0:
        raise ValueError("graph node index must be non-negative")
    return f"mixed-node-{index:05d}"


def graph_edge_id(index: int) -> str:
    if index < 0:
        raise ValueError("graph edge index must be non-negative")
    return f"mixed-edge-{index:05d}"


def previous_graph_node_id(index: int) -> str:
    if index < 0:
        raise ValueError("graph node index must be non-negative")
    return ROOT_NODE_ID if index == 0 else graph_node_id(index - 1)


def _root_source_line() -> str:
    return f"Root node {ROOT_NODE_ID}.\n"


def _record_source_line(index: int) -> str:
    return (
        f"Record {index:05d}: {previous_graph_node_id(index)} precedes "
        f"{graph_node_id(index)}.\n"
    )


def build_source_text(items: int) -> str:
    if items < 1:
        raise ValueError("items must be positive")
    return _root_source_line() + "".join(_record_source_line(index) for index in range(items))


def _node_payload(
    *, node_id: str, label: str, summary: str, excerpt: str, start_char: int,
    properties: dict[str, object], metadata: dict[str, object],
) -> dict[str, object]:
    span = Span(
        collection_page_url="fixture/mixed-rewrite",
        document_page_url="fixture/mixed-rewrite/source-v1",
        doc_id=SOURCE_DOC_ID,
        insertion_method="benchmark_fixture",
        page_number=1,
        start_char=start_char,
        end_char=start_char + len(excerpt),
        excerpt=excerpt,
    )
    node = Node(
        id=node_id,
        label=label,
        type="entity",
        summary=summary,
        doc_id=SOURCE_DOC_ID,
        properties=properties,
        mentions=[Grounding(spans=[span])],
        metadata=metadata,
    )
    payload = json.loads(node.model_dump_json(field_mode="backend"))
    if not isinstance(payload, dict):
        raise TypeError("Kogwistar Node serialization must produce an object")
    return payload


def _edge_payload(
    *, edge_id: str, source_id: str, target_id: str, excerpt: str, start_char: int,
    index: int,
) -> dict[str, object]:
    span = Span(
        collection_page_url="fixture/mixed-rewrite",
        document_page_url="fixture/mixed-rewrite/source-v1",
        doc_id=SOURCE_DOC_ID,
        insertion_method="benchmark_fixture",
        page_number=1,
        start_char=start_char,
        end_char=start_char + len(excerpt),
        excerpt=excerpt,
    )
    edge = Edge(
        id=edge_id,
        label="precedes",
        type="relationship",
        summary=f"{source_id} precedes {target_id}",
        doc_id=SOURCE_DOC_ID,
        mentions=[Grounding(spans=[span])],
        source_ids=[source_id],
        target_ids=[target_id],
        relation="precedes",
        source_edge_ids=[],
        target_edge_ids=[],
        properties={"ordinal": index},
        metadata={"benchmark": True, "ordinal": index},
    )
    payload = json.loads(edge.model_dump_json(field_mode="backend"))
    if not isinstance(payload, dict):
        raise TypeError("Kogwistar Edge serialization must produce an object")
    return payload


def build_root_graph_payload() -> dict[str, object]:
    excerpt = _root_source_line().rstrip("\n")
    return _node_payload(
        node_id=ROOT_NODE_ID,
        label="Root node",
        summary=f"Deterministic graph root {ROOT_NODE_ID}",
        excerpt=excerpt,
        start_char=0,
        properties={"kind": "benchmark_root"},
        metadata={"benchmark": True, "root": True},
    )


def build_graph_payloads(
    index: int, *, vector_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    node_id = graph_node_id(index)
    edge_id = graph_edge_id(index)
    source_id = previous_graph_node_id(index)
    line = _record_source_line(index)
    excerpt = line.rstrip("\n")
    first_record_offset = len(_root_source_line())
    first_record_length = len(_record_source_line(0))
    later_record_length = len(_record_source_line(1))
    start_char = (
        first_record_offset
        if index == 0
        else first_record_offset + first_record_length + (index - 1) * later_record_length
    )
    node = _node_payload(
        node_id=node_id,
        label=f"Benchmark node {index:05d}",
        summary=f"Deterministic mixed-workload graph node {index:05d}",
        excerpt=excerpt,
        start_char=start_char,
        properties={"ordinal": index, "vector_sha256": vector_sha256},
        metadata={"benchmark": True, "ordinal": index},
    )
    edge = _edge_payload(
        edge_id=edge_id,
        source_id=source_id,
        target_id=node_id,
        excerpt=excerpt,
        start_char=start_char,
        index=index,
    )
    return node, edge


def build_normalized_vector(index: int, *, dimension: int = VECTOR_DIMENSION) -> list[float]:
    """Precompute a stable float32 vector without model inference."""
    if index < 0 or dimension < 1:
        raise ValueError("index must be non-negative and dimension must be positive")
    raw = hashlib.shake_256(f"{VECTOR_SEED}:{index}".encode("ascii")).digest(dimension * 4)
    values = [
        (int.from_bytes(raw[offset : offset + 4], "little") / 0xFFFFFFFF) * 2.0 - 1.0
        for offset in range(0, len(raw), 4)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise RuntimeError("deterministic vector unexpectedly has zero magnitude")
    return [
        struct.unpack("<f", struct.pack("<f", value / norm))[0]
        for value in values
    ]


def normalized_vector_digest(vector: list[float]) -> str:
    packed = b"".join(struct.pack("<f", value) for value in vector)
    return hashlib.sha256(packed).hexdigest()


def build_read_after_write_trace(items: int) -> list[WorkloadOperation]:
    """Create a stable trace with a committed write before its two reads."""
    if items < 1:
        raise ValueError("items must be positive")
    trace: list[WorkloadOperation] = []
    for index in range(items):
        entity_id = graph_node_id(index)
        for operation in ("write_record", "read_event", "read_projection"):
            trace.append(
                {
                    "operation": operation,
                    "index": index,
                    "expected_entity_id": entity_id,
                    "expected_edge_id": graph_edge_id(index),
                    "expected_source_id": previous_graph_node_id(index),
                }
            )
    return trace


def write_trace(path: Path, items: int) -> list[WorkloadOperation]:
    trace = build_read_after_write_trace(items)
    path.write_text(
        json.dumps(
            {
                "schema": "kogwistar-mixed-rewrite-trace/v2",
                "profile": "read_after_write",
                "operations": trace,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return trace


def read_trace(path: Path) -> list[WorkloadOperation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "kogwistar-mixed-rewrite-trace/v2":
        raise ValueError("unsupported mixed workload trace schema")
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("workload trace operations must be a list")
    expected_operations = build_read_after_write_trace(len(operations) // 3)
    if operations != expected_operations:
        raise ValueError("workload trace is malformed or not canonical")
    return expected_operations


def build_concurrent_trace(
    items: int, *, mix: Literal["balanced", "read_heavy"], readers: int = 4
) -> list[ConcurrentPhase]:
    """Build a deterministic phase schedule with explicit visibility bounds.

    Each write is one commit. All four readers observe the prior committed
    revision while the new write is uncommitted, then the new revision after
    commit; this exercises overlap and gives every observation exact bounds.
    """
    if items < 1:
        raise ValueError("items must be positive")
    if readers != 4:
        raise ValueError("the v1 concurrent workload requires exactly four readers")
    writes_per_phase = readers if mix == "balanced" else 1
    if mix not in {"balanced", "read_heavy"}:
        raise ValueError(f"unsupported concurrent mix: {mix}")

    phases: list[ConcurrentPhase] = []
    first_index = 0
    phase_number = 0
    while first_index < items:
        write_indexes = list(range(first_index, min(first_index + writes_per_phase, items)))
        steps: list[ConcurrentStep] = []
        for index in write_indexes:
            before_readers = (
                list(range(readers))
                if mix == "read_heavy"
                else [index % readers, (index + 1) % readers]
            )
            after_readers = (
                list(range(readers))
                if mix == "read_heavy"
                else [
                    reader_id
                    for reader_id in range(readers)
                    if reader_id not in before_readers
                ]
            )
            reads: list[ConcurrentRead] = []
            reads.extend(
                {
                    "reader_id": reader_id,
                    "visibility": "before_commit",
                    "minimum_revision": index,
                    "maximum_revision": index,
                }
                for reader_id in before_readers
            )
            reads.extend(
                {
                    "reader_id": reader_id,
                    "visibility": "after_commit",
                    "minimum_revision": index + 1,
                    "maximum_revision": index + 1,
                }
                for reader_id in after_readers
            )
            steps.append({"write_index": index, "reads": reads})
        phases.append(
            {
                "phase": phase_number,
                "steps": steps,
            }
        )
        first_index += len(write_indexes)
        phase_number += 1
    return phases


def write_concurrent_trace(
    path: Path,
    items: int,
    *,
    mix: Literal["balanced", "read_heavy"],
    readers: int = 4,
) -> list[ConcurrentPhase]:
    phases = build_concurrent_trace(items, mix=mix, readers=readers)
    path.write_text(
        json.dumps(
            {
                "schema": "kogwistar-mixed-rewrite-concurrent-trace/v3",
                "profile": f"concurrent_{mix}",
                "readers": readers,
                "phases": phases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return phases


def read_concurrent_trace(path: Path) -> tuple[str, list[ConcurrentPhase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workload trace must be an object")
    if payload.get("schema") != "kogwistar-mixed-rewrite-concurrent-trace/v3":
        raise ValueError("unsupported mixed workload trace schema")
    profile = payload.get("profile")
    if profile not in {"concurrent_balanced", "concurrent_read_heavy"}:
        raise ValueError("unsupported concurrent workload profile")
    mix: Literal["balanced", "read_heavy"] = (
        "balanced" if profile == "concurrent_balanced" else "read_heavy"
    )
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("concurrent workload phases must be a non-empty list")
    item_count = sum(
        len(phase.get("steps", []))
        for phase in raw_phases
        if isinstance(phase, dict)
    )
    expected = build_concurrent_trace(item_count, mix=mix)
    if payload.get("readers") != 4 or raw_phases != expected:
        raise ValueError("concurrent workload trace is malformed or not canonical")
    return mix, expected
