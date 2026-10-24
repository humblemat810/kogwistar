"""Measure the Python/PyO3/Rust cost of graph-model field slicing.

This is a microbenchmark for the typed graph-model seam, not a storage or
end-to-end service benchmark. It keeps the canonical payload and projection
contract identical across paths, then reports the cost of each boundary stage
so a Rust-native caller can be compared without blaming the wrong layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Prefer the checkout when the benchmark is launched from an environment that
# also has an older published kogwistar installed.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kogwistar._rust_bridge import project_graph_model_slice, project_graph_model_slices
from kogwistar.engine_core.graph_contracts import build_graph_model_slice_envelope
from kogwistar.engine_core.graph_slice_manifest import project_with_manifest
from kogwistar.engine_core.models import Edge, Node


JsonObject = dict[str, Any]


def _mentions(excerpt: str) -> list[JsonObject]:
    return [
        {
            "spans": [
                {
                    "collection_page_url": "collection/benchmark",
                    "document_page_url": "document/benchmark",
                    "doc_id": "benchmark",
                    "insertion_method": "benchmark",
                    "start_char": 0,
                    "end_char": len(excerpt),
                    "excerpt": excerpt,
                }
            ]
        }
    ]


def _models(count: int) -> list[Node | Edge]:
    result: list[Node | Edge] = []
    for index in range(count):
        result.append(
            Node(
                id=f"node-{index}",
                label=f"node-{index}",
                type="entity",
                summary=f"Node {index} used for seam measurement",
                mentions=_mentions(f"node {index}"),
            )
        )
        result.append(
            Edge(
                id=f"edge-{index}",
                label=f"edge-{index}",
                type="relationship",
                summary=f"Edge {index} used for seam measurement",
                relation="relates_to",
                source_ids=[f"node-{index}"],
                target_ids=[f"node-{(index + 1) % count}"],
                source_edge_ids=None,
                target_edge_ids=None,
                mentions=_mentions(f"edge {index}"),
            )
        )
    return result


def _timed(call: Callable[[], Any], iterations: int) -> list[float]:
    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        call()
        values.append((time.perf_counter_ns() - start) / 1_000_000)
    return values


def _summary(values: list[float]) -> JsonObject:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run(*, model_count: int, iterations: int, field_mode: str) -> JsonObject:
    models = _models(model_count)
    envelopes = [
        build_graph_model_slice_envelope(model, field_mode=field_mode)
        for model in models
    ]
    encoded = [
        json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        for envelope in envelopes
    ]

    # Warm up model and extension loading before recording measurements.
    python_result = [
        project_with_manifest(type(model), model, field_mode) for model in models
    ]
    native = __import__("kogwistar._rust", fromlist=["project_graph_model_slice"])
    native_result = [json.loads(native.project_graph_model_slice(item)) for item in encoded]
    if [item["payload"] for item in native_result] != python_result:
        raise RuntimeError("Python and Rust slice projections differ")

    python_projection = _timed(
        lambda: [
            project_with_manifest(type(model), model, field_mode) for model in models
        ],
        iterations,
    )
    python_encode = _timed(
        lambda: [
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
            for envelope in envelopes
        ],
        iterations,
    )
    rust_call = _timed(
        lambda: [native.project_graph_model_slice(item) for item in encoded],
        iterations,
    )
    rust_decode = _timed(
        lambda: [json.loads(native.project_graph_model_slice(item)) for item in encoded],
        iterations,
    )
    bridge_total = _timed(
        lambda: [project_graph_model_slice(envelope) for envelope in envelopes],
        iterations,
    )
    bridge_batch = _timed(
        lambda: project_graph_model_slices(envelopes),
        iterations,
    )

    bridge_result = [project_graph_model_slice(envelope) for envelope in envelopes]
    bridge_batch_result = project_graph_model_slices(envelopes)
    if [item["payload"] for item in bridge_batch_result] != [
        item["payload"] for item in bridge_result
    ]:
        raise RuntimeError("batch and per-envelope bridge projections differ")
    return {
        "schema": "kogwistar-graph-slice-seam-benchmark/v1",
        "model_count": model_count,
        "envelope_count": len(envelopes),
        "iterations": iterations,
        "field_mode": field_mode,
        "payload_bytes": sum(len(item.encode("utf-8")) for item in encoded),
        "python_projection_digest": _digest(python_result),
        "rust_projection_digest": _digest([item["payload"] for item in native_result]),
        "bridge_projection_digest": _digest([item["payload"] for item in bridge_result]),
        "bridge_batch_projection_digest": _digest(
            [item["payload"] for item in bridge_batch_result]
        ),
        "stages": {
            "python_model_projection": _summary(python_projection),
            "python_json_encode": _summary(python_encode),
            "rust_pyo3_call_and_json_encode": _summary(rust_call),
            "rust_pyo3_call_decode": _summary(rust_decode),
            "python_bridge_end_to_end": _summary(bridge_total),
            "python_bridge_batch_end_to_end": _summary(bridge_batch),
        },
        "interpretation": {
            "python_projection": "Pydantic-extension governed Python projection only",
            "rust_pyo3_call_and_json_encode": "Rust projection plus PyO3 entry and output serialization",
            "rust_pyo3_call_decode": "Rust projection plus PyO3 and Python JSON decoding",
            "python_bridge_end_to_end": "Public Python bridge including input/output JSON conversion",
            "rust_native_gap": "Run a Rust caller against the same envelope to remove the PyO3/string seam; this script does not claim that result.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-count", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--field-mode", default="llm_in")
    parser.add_argument(
        "--write-input",
        type=Path,
        help="write the canonical envelope fixture for the Rust-native benchmark",
    )
    args = parser.parse_args()
    if args.model_count < 1 or args.iterations < 2:
        parser.error("--model-count must be positive and --iterations must be at least 2")
    if args.write_input is not None:
        args.write_input.parent.mkdir(parents=True, exist_ok=True)
        args.write_input.write_text(
            json.dumps(
                [
                    build_graph_model_slice_envelope(model, field_mode=args.field_mode)
                    for model in _models(args.model_count)
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            run(
                model_count=args.model_count,
                iterations=args.iterations,
                field_mode=args.field_mode,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
