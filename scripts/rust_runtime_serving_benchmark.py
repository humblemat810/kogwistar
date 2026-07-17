"""Benchmark ADR-015 indexed runtime serving lookups as history grows.

This is a code-smell regression gate, not a Python-vs-Rust language benchmark.
It verifies the specific projection/index fix: 100x unrelated recorded history
must not cause linear growth in exact retry or warm current-state reads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Callable

from kogwistar.runtime.rust_runtime_adapter import (
    apply_recorded_transition,
    read_recorded_runtime_state,
)


DATASET = "rust-runtime-serving-v1"
SMALL_HISTORY = 100
LARGE_HISTORY = 10_000
HISTORY_MULTIPLIER = LARGE_HISTORY // SMALL_HISTORY
MAX_GROWTH_RATIO = 2.5


def _start() -> dict[str, Any]:
    return {
        "contract_version": 1,
        "transition_id": "target-transition",
        "expected_event_seq": 0,
        "kind": "start",
        "run_id": "target-run",
        "workflow_id": "wf-1",
        "conversation_id": "conv-1",
        "step_seq": 0,
        "node_id": "node-1",
        "token_id": "token-1",
        "parent_token_id": None,
        "user_id": "user-1",
        "user_turn_node_id": "turn-1",
        "initial_state": {"answer": "seed"},
        "frontier": {
            "pending": [["node-1", 0, "token-1", None]],
            "suspended": [],
            "join_node_ids": [],
            "join_outstanding": [],
            "join_waiters": {},
        },
    }


def _seed(path: Path, history_size: int) -> dict[str, Any]:
    transition = _start()
    accepted = apply_recorded_transition(path=path, transition=transition)
    with sqlite3.connect(path) as connection:
        payload_json = connection.execute(
            "SELECT payload_json FROM server_run_events "
            "WHERE run_id='target-run' AND event_type='workflow.recorded_transition.v1'"
        ).fetchone()[0]
        payload = json.loads(payload_json)
        rows = []
        for index in range(history_size):
            noise = dict(payload)
            noise["transition_id"] = f"noise-{index}"
            rows.append(
                (
                    f"noise-run-{index}",
                    "workflow.recorded_transition.v1",
                    json.dumps(noise, separators=(",", ":")),
                )
            )
        connection.executemany(
            "INSERT INTO server_run_events(run_id,event_type,payload_json,created_at_ms) "
            "VALUES (?,?,?,0)",
            rows,
        )
    return {"transition": transition, "accepted": accepted}


def _measure(operation: Callable[[], None], iterations: int) -> float:
    operation()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    return (time.perf_counter_ns() - started) / iterations


def _sample(path: Path, history_size: int, iterations: int) -> dict[str, float | int]:
    seeded = _seed(path, history_size)
    transition = seeded["transition"]

    def retry() -> None:
        result = apply_recorded_transition(path=path, transition=transition)
        if result.get("idempotent") is not True:
            raise RuntimeError("exact retry was not idempotent")

    def read() -> None:
        state = read_recorded_runtime_state(
            path=path,
            run_id="target-run",
            workflow_id="wf-1",
            conversation_id="conv-1",
        )
        if state is None or state.get("state", {}).get("answer") != "seed":
            raise RuntimeError("warm current-state read drifted")

    return {
        "history_size": history_size,
        "retry_mean_ns": _measure(retry, iterations),
        "state_read_mean_ns": _measure(read, iterations),
    }


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def run(output_root: Path, *, samples: int, iterations: int) -> dict[str, Any]:
    if samples < 2 or iterations < 1:
        raise ValueError("samples must be >= 2 and iterations must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    measured: dict[str, list[dict[str, float | int]]] = {}
    for label, history_size in (("small", SMALL_HISTORY), ("large", LARGE_HISTORY)):
        values = []
        for sample in range(samples):
            path = output_root / f"{label}-{sample}.sqlite3"
            path.unlink(missing_ok=True)
            values.append(_sample(path, history_size, iterations))
        measured[label] = values

    summary: dict[str, dict[str, float | int]] = {}
    for label, values in measured.items():
        summary[label] = {
            "history_size": int(values[0]["history_size"]),
            "retry_median_ns": statistics.median(float(row["retry_mean_ns"]) for row in values),
            "retry_p95_ns": _p95([float(row["retry_mean_ns"]) for row in values]),
            "state_read_median_ns": statistics.median(
                float(row["state_read_mean_ns"]) for row in values
            ),
            "state_read_p95_ns": _p95(
                [float(row["state_read_mean_ns"]) for row in values]
            ),
        }
    ratios = {
        "retry_p95_large_over_small": summary["large"]["retry_p95_ns"]
        / max(float(summary["small"]["retry_p95_ns"]), 1.0),
        "state_read_p95_large_over_small": summary["large"]["state_read_p95_ns"]
        / max(float(summary["small"]["state_read_p95_ns"]), 1.0),
    }
    checks = {name: value <= MAX_GROWTH_RATIO for name, value in ratios.items()}
    return {
        "benchmark_version": 1,
        "dataset": DATASET,
        "python": sys.version,
        "history_multiplier": HISTORY_MULTIPLIER,
        "samples": samples,
        "iterations_per_sample": iterations,
        "scope": "indexed exact retry and warm current-state projection lookup",
        "summary": summary,
        "ratios": ratios,
        "gate": {
            "max_growth_ratio": MAX_GROWTH_RATIO,
            "checks": checks,
            "status": "passed" if all(checks.values()) else "failed",
        },
        "raw_samples": measured,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.smoke:
        args.samples = 2
        args.iterations = 5
    report = run(args.output_root.resolve(), samples=args.samples, iterations=args.iterations)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    return 0 if report["gate"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
