"""Run the fixed Python/Rust-native SQLite + Chroma trace comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.native_graph_backend_benchmark import (
    PostgresGraphTraceAdapter,
    build_backend_trace_document,
    run_trace_sample,
    verify_backend_report,
)


def collect_chroma_trace_report(
    trace: Mapping[str, Any],
    run_sample: Callable[[str, str], Mapping[str, Any]],
    *,
    expected_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect warm-up and fresh samples for both implementations."""
    if expected_samples < 2:
        raise ValueError("at least two measured samples per mode are required")
    trace_id = trace.get("trace_id")
    trace_sha = trace.get("trace_sha256")
    if not isinstance(trace_id, str) or not trace_id:
        raise ValueError("trace id is required")
    if not isinstance(trace_sha, str) or not trace_sha:
        raise ValueError("trace digest is required")

    mode_reports: dict[str, Any] = {}
    for mode in ("python", "rust_native"):
        warmup = dict(run_sample(mode, f"{mode}-warmup-{uuid.uuid4().hex[:8]}"))
        samples = [
            dict(run_sample(mode, f"{mode}-sample-{index}-{uuid.uuid4().hex[:8]}"))
            for index in range(1, expected_samples + 1)
        ]
        mode_reports[mode] = {"warmup": warmup, "samples": samples}

    report = {
        "schema": "kogwistar-native-graph-backend-benchmark/v1",
        "backend": "chroma_sqlite_metadata",
        "samples_per_mode": expected_samples,
        "warmup_samples_per_mode": 1,
        "trace": {"trace_id": trace_id, "trace_sha256": trace_sha},
        "modes": mode_reports,
    }
    result = verify_backend_report(
        report,
        expected_backend="chroma_sqlite_metadata",
        expected_samples=expected_samples,
    )
    return report, result


def _fixed_container_limits() -> dict[str, int]:
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        memory_max = Path("/sys/fs/cgroup/memory.max")
        try:
            quota, period = cpu_max.read_text().split()
            memory_text = memory_max.read_text().strip()
            cpu_micros = int(quota)
            period_micros = int(period)
        except OSError:
            cpu_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            cpu_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            memory_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            cpu_micros = int(cpu_quota.read_text().strip())
            period_micros = int(cpu_period.read_text().strip())
            memory_text = memory_limit.read_text().strip()
        memory = int(memory_text)
    except (OSError, ValueError) as error:
        raise RuntimeError("benchmark must run in a cgroup-limited container") from error
    if period_micros <= 0 or cpu_micros != period_micros * 2:
        raise RuntimeError("benchmark cgroup CPU limit must be exactly 2 CPUs")
    if memory != 4 * 1024**3:
        raise RuntimeError("benchmark cgroup memory limit must be exactly 4 GiB")
    return {"cpu_quota_micros": cpu_micros, "cpu_period_micros": period_micros, "memory_bytes": memory}


def _collection_name(kind: str) -> str:
    return f"kgd18_{kind}_{uuid.uuid4().hex[:12]}"


def _make_python_backend(client: Any, node_name: str, edge_name: str) -> Any:
    from kogwistar.engine_core.chroma_backend import ChromaBackend

    def collection(name: str) -> Any:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    return ChromaBackend(
        node_index_collection=collection(_collection_name("node_index")),
        node_collection=collection(node_name),
        edge_collection=collection(edge_name),
        edge_endpoints_collection=collection(_collection_name("endpoints")),
        document_collection=collection(_collection_name("documents")),
        domain_collection=collection(_collection_name("domains")),
        node_docs_collection=collection(_collection_name("node_docs")),
        node_refs_collection=collection(_collection_name("node_refs")),
        edge_refs_collection=collection(_collection_name("edge_refs")),
    )


def run_benchmark(
    *,
    output_dir: Path,
    rust_executable: Path,
    chroma_url: str,
    items: int = 100,
    samples: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    limits = _fixed_container_limits()
    if items < 1:
        raise ValueError("items must be positive")
    if samples < 2:
        raise ValueError("at least two measured samples per mode are required")
    if not rust_executable.is_file():
        raise FileNotFoundError(rust_executable)
    if not chroma_url:
        raise ValueError("Chroma URL is required")
    output_dir.mkdir(parents=True, exist_ok=False)
    trace = build_backend_trace_document(items)
    trace_path = output_dir / "trace.v1.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    def run_one(mode: str, sample_id: str) -> Mapping[str, Any]:
        root = output_dir / mode / sample_id
        root.mkdir(parents=True)
        node_collection = _collection_name("nodes")
        edge_collection = _collection_name("edges")
        namespace = f"dlfc18_{uuid.uuid4().hex[:12]}"
        if mode == "python":
            import chromadb

            from kogwistar.engine_core.engine_sqlite import EngineSQLite

            client = chromadb.HttpClient(
                host=chroma_url.removeprefix("http://").removeprefix("https://").split(":")[0],
                port=int(chroma_url.rsplit(":", 1)[1].rstrip("/")),
            )
            backend = _make_python_backend(client, node_collection, edge_collection)
            metadata = EngineSQLite(root, filename="metadata.sqlite3")
            metadata.ensure_initialized()
            adapter = PostgresGraphTraceAdapter(
                metadata,
                backend,
                namespace=namespace,
                trace_id=trace["trace_id"],
                eventual_projection=True,
            )
            adapter.seed_root(
                trace["root_payload"],
                vector=trace["operations"][0]["vector"],
            )
            return run_trace_sample(
                adapter,
                trace["operations"],
                trace_id=trace["trace_id"],
                trace_sha256=trace["trace_sha256"],
                resource_limits_verified=True,
                ffi_calls=0,
            )

        sqlite_path = root / "metadata.sqlite3"
        env = os.environ.copy()
        env["KOGWISTAR_BENCH_RESOURCE_LIMITS_VERIFIED"] = "1"
        env["KOGWISTAR_TEST_CHROMA_URL"] = chroma_url
        env["KOGWISTAR_BENCH_CHROMA_NODE_COLLECTION"] = node_collection
        env["KOGWISTAR_BENCH_CHROMA_EDGE_COLLECTION"] = edge_collection
        completed = subprocess.run(
            [str(rust_executable), str(trace_path), str(sqlite_path), namespace],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(completed.stdout)

    report, verification = collect_chroma_trace_report(
        trace,
        run_one,
        expected_samples=samples,
    )
    report["resource_limits"] = limits
    report["trace"]["items"] = items
    report["verification"] = verification
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report, verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rust-executable", type=Path, required=True)
    parser.add_argument("--chroma-url", default=os.environ.get("KOGWISTAR_TEST_CHROMA_URL", ""))
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    _, verification = run_benchmark(
        output_dir=args.output_dir,
        rust_executable=args.rust_executable,
        chroma_url=args.chroma_url,
        items=args.items,
        samples=args.samples,
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
