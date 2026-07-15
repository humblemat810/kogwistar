from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Callable

from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.runtime.perf_profile import (
    _build_profile_engine,
    _mk_edge,
    _mk_node,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_VERSION = "rust-port-benchmark-v1"


def _memory_info() -> tuple[int | None, int | None]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss), None
    except (ImportError, OSError):
        pass
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak *= 1024
        return None, peak
    except (ImportError, OSError, ValueError):
        return None, None


def _artifact_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _measure(fn: Callable[[], int]) -> dict[str, int | float | None]:
    tracemalloc.start()
    rss_before, peak_rss_before = _memory_info()
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    output_count = int(fn())
    wall_seconds = time.perf_counter() - wall_before
    cpu_seconds = time.process_time() - cpu_before
    _, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after, peak_rss_after = _memory_info()
    return {
        "wall_ms": round(wall_seconds * 1000.0, 6),
        "cpu_ms": round(cpu_seconds * 1000.0, 6),
        "peak_traced_bytes": int(peak_traced),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "peak_rss_bytes": peak_rss_after or peak_rss_before,
        "output_count": output_count,
    }


@dataclass
class _WorkflowNode:
    id: str
    op: str
    terminal: bool
    fanout: bool
    metadata: dict[str, Any]

    def safe_get_id(self) -> str:
        return self.id


@dataclass
class _WorkflowEdge:
    id: str
    label: str
    predicate: str | None
    source_ids: list[str]
    target_ids: list[str]
    multiplicity: str
    is_default: bool
    metadata: dict[str, Any]

    def safe_get_id(self) -> str:
        return self.id


class _WorkflowEngine:
    def __init__(self, nodes: list[_WorkflowNode], edges: list[_WorkflowEdge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.read = self
        self.write = self

    def get_nodes(self, **_kwargs):
        return self.nodes

    def get_edges(self, **_kwargs):
        return self.edges


class _ConversationEngine:
    def __init__(self) -> None:
        self.nodes: list[Any] = []
        self.edges: list[Any] = []
        self.read = self
        self.write = self

    def add_node(self, node):
        self.nodes.append(node)
        return node

    def add_edge(self, edge):
        self.edges.append(edge)
        return edge

    def get_nodes(self, **_kwargs):
        return self.nodes

    def get_edges(self, **_kwargs):
        return self.edges


def _workflow_node(
    node_id: str, *, workflow_id: str, start: bool = False, terminal: bool = False,
    fanout: bool = False, join: bool = False,
) -> _WorkflowNode:
    metadata = {
        "entity_type": "workflow_node",
        "workflow_id": workflow_id,
        "wf_op": node_id,
        "wf_version": "v1",
        "wf_start": start,
        "wf_terminal": terminal,
        "wf_fanout": fanout,
    }
    if join:
        metadata["wf_join"] = True
    return _WorkflowNode(node_id, node_id, terminal, fanout, metadata)


def _workflow_edge(edge_id: str, *, workflow_id: str, source: str, target: str) -> _WorkflowEdge:
    metadata = {
        "entity_type": "workflow_edge",
        "workflow_id": workflow_id,
        "wf_predicate": None,
        "wf_priority": 100,
        "wf_is_default": False,
        "wf_multiplicity": "one",
    }
    return _WorkflowEdge(
        edge_id,
        edge_id,
        None,
        [source],
        [target],
        "one",
        False,
        metadata,
    )


def _fanout_join_workload(iterations: int) -> int:
    workflow_id = "rust-port-benchmark-fanout-join"
    nodes = [
        _workflow_node("start", workflow_id=workflow_id, start=True),
        _workflow_node("fork", workflow_id=workflow_id, fanout=True),
        _workflow_node("branch_a", workflow_id=workflow_id),
        _workflow_node("branch_b", workflow_id=workflow_id),
        _workflow_node("join", workflow_id=workflow_id, join=True),
        _workflow_node("end", workflow_id=workflow_id, terminal=True),
    ]
    edges = [
        _workflow_edge("e1", workflow_id=workflow_id, source="start", target="fork"),
        _workflow_edge("e2", workflow_id=workflow_id, source="fork", target="branch_a"),
        _workflow_edge("e3", workflow_id=workflow_id, source="fork", target="branch_b"),
        _workflow_edge("e4", workflow_id=workflow_id, source="branch_a", target="join"),
        _workflow_edge("e5", workflow_id=workflow_id, source="branch_b", target="join"),
        _workflow_edge("e6", workflow_id=workflow_id, source="join", target="end"),
    ]
    resolver = MappingStepResolver()
    for operation in ("start", "fork", "branch_a", "branch_b", "join", "end"):
        resolver.register(operation)(
            lambda _context: RunSuccess(conversation_node_id=None, state_update=[])
        )
    runtime = WorkflowRuntime(
        workflow_engine=_WorkflowEngine(nodes, edges),
        conversation_engine=_ConversationEngine(),
        step_resolver=resolver,
        predicate_registry={},
        max_workers=2,
        trace=False,
    )
    completed = 0
    trace_logger = logging.getLogger("workflow.trace")
    previous_level = trace_logger.level
    trace_logger.setLevel(logging.WARNING)
    try:
        for index in range(iterations):
            result = runtime.run(
                workflow_id=workflow_id,
                conversation_id=f"benchmark-conversation-{index}",
                turn_node_id=f"benchmark-turn-{index}",
                initial_state={},
                run_id=f"benchmark-run-{index}",
            )
            if result.status == "succeeded":
                completed += 1
    finally:
        trace_logger.setLevel(previous_level)
    return completed


def _run_benchmarks(output_root: Path, *, items: int, workflow_iterations: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    engine = _build_profile_engine(
        output_root / "graph", graph_type="knowledge", backend_kind="fake"
    )
    node_ids = [f"benchmark-node-{index:05d}" for index in range(items)]

    def ingest() -> int:
        for index, node_id in enumerate(node_ids):
            engine.write.add_node(_mk_node(node_id, doc_id=f"benchmark-doc-{index:05d}"))
        for index in range(max(0, items - 1)):
            engine.write.add_edge(
                _mk_edge(
                    f"benchmark-edge-{index:05d}",
                    src=node_ids[index],
                    tgt=node_ids[index + 1],
                    doc_id=f"benchmark-edge-doc-{index:05d}",
                )
            )
        return items + max(0, items - 1)

    metrics = {"graph_ingest": _measure(ingest)}
    metrics["graph_retrieval"] = _measure(
        lambda: len(engine.read.get_nodes(ids=node_ids, limit=items))
        + len(
            engine.read.query_nodes(
                query_embeddings=[[0.1, 0.2, 0.3]], n_results=items
            )[0]
        )
    )
    metrics["event_replay"] = _measure(
        lambda: len(
            list(
                engine.meta_sqlite.iter_entity_events(
                    namespace=engine.namespace, from_seq=1, batch_size=max(1, items)
                )
            )
        )
    )

    def claim_queue() -> int:
        for index in range(items):
            engine.meta_sqlite.enqueue_index_job(
                job_id=f"benchmark-job-{index:05d}",
                entity_kind="node",
                entity_id=f"benchmark-queued-node-{index:05d}",
                index_kind="node",
                op="UPSERT",
                namespace=engine.namespace,
            )
        return len(
            engine.meta_sqlite.claim_index_jobs(
                limit=items, lease_seconds=60, namespace=engine.namespace
            )
        )

    metrics["queue_claim"] = _measure(claim_queue)
    metrics["workflow_fanout_join"] = _measure(
        lambda: _fanout_join_workload(workflow_iterations)
    )
    report = {
        "benchmark_version": 1,
        "dataset": DATASET_VERSION,
        "command": (
            f"{sys.executable} scripts/rust_port_benchmark.py --items {items} "
            f"--workflow-iterations {workflow_iterations} --output-root {output_root}"
        ),
        "python": sys.version,
        "backend": "in-memory",
        "items": items,
        "workflow_iterations": workflow_iterations,
        "database_load": {"kind": "in-memory", "external_queries": 0},
        "artifact_bytes": _artifact_bytes(output_root),
        "metrics": metrics,
    }
    engine.close()
    return report


def run_benchmarks(output_root: Path, *, items: int, workflow_iterations: int) -> dict[str, Any]:
    previous_endpoint = os.environ.get("CDC_PUBLISH_ENDPOINT")
    os.environ["CDC_PUBLISH_ENDPOINT"] = ""
    try:
        return _run_benchmarks(
            output_root, items=items, workflow_iterations=workflow_iterations
        )
    finally:
        if previous_endpoint is None:
            os.environ.pop("CDC_PUBLISH_ENDPOINT", None)
        else:
            os.environ["CDC_PUBLISH_ENDPOINT"] = previous_endpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable ADR-015 baseline workloads.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--workflow-iterations", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.items < 2 or args.workflow_iterations < 1:
        raise SystemExit("--items must be >= 2 and --workflow-iterations must be >= 1")
    report = run_benchmarks(
        args.output_root.resolve(),
        items=args.items,
        workflow_iterations=args.workflow_iterations,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
