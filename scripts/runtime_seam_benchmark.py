"""Compare Rust-scheduled Python-worker steps with Rust-inline steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = (
    ROOT
    / "rust"
    / "target"
    / "debug"
    / ("kogwistar-server.exe" if os.name == "nt" else "kogwistar-server")
)


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cgroup_value(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    if value:
        return value if value != "max" else None
    if name == "cpu.max":
        try:
            quota = (
                Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
                .read_text(encoding="ascii")
                .strip()
            )
            period = (
                Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
                .read_text(encoding="ascii")
                .strip()
            )
        except OSError:
            return None
        return f"{quota} {period}" if quota and period else None
    if name == "memory.max":
        try:
            return (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                .read_text(encoding="ascii")
                .strip()
            )
        except OSError:
            return None
    return None


def _verify_limits(
    cpu_max: str | None, memory_max: str | None, cpus: float, memory: int
) -> bool:
    if not cpu_max or not memory_max:
        return False
    quota, separator, period = cpu_max.partition(" ")
    try:
        cpu_actual = int(quota) / int(period) if separator else None
        memory_actual = int(memory_max)
    except ValueError:
        return False
    return cpu_actual == cpus and memory_actual == memory


def _wait_ready(process: subprocess.Popen[str], base_url: str) -> float:
    started = time.perf_counter()
    with httpx.Client(timeout=0.5) as client:
        for _ in range(200):
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"Rust server exited before ready: {stderr}")
            try:
                if client.get(f"{base_url}/health").status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                time.sleep(0.025)
    process.terminate()
    try:
        _stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        _stdout, stderr = process.communicate(timeout=5)
    raise TimeoutError(f"Rust server readiness timeout: {stderr[-4000:]}")


def _workflow_payload(*, mode: str, run_id: str, nodes: int) -> dict[str, Any]:
    ids = [f"step-{index:05d}" for index in range(nodes)]
    native = mode == "rust_inline"
    return {
        "run_id": run_id,
        "workflow_id": "runtime-seam-benchmark",
        "conversation_id": f"conversation-{run_id}",
        "initial_state": {"counter": 0},
        "runtime_kind": "sync",
        "start_node_id": ids[0],
        "node_ops": {
            node_id: "increment-counter" if native else "benchmark.increment"
            for node_id in ids
        },
        "node_executor_targets": {
            node_id: "rust-native" if native else "python-worker" for node_id in ids
        },
        "runtime_routes": [
            {
                "source_node_id": source,
                "target_node_id": target,
                "join_mask": 0,
            }
            for source, target in zip(ids, ids[1:])
        ],
    }


def _run_sample(server: Path, root: Path, *, mode: str, nodes: int) -> dict[str, Any]:
    from kogwistar.runtime.models import RunSuccess
    from kogwistar.runtime.resolvers import MappingStepResolver
    from kogwistar.runtime.rust_worker import RustRuntimeWorker

    port = _available_port()
    with tempfile.TemporaryDirectory(prefix="kogwistar-runtime-seam-") as temporary:
        work = Path(temporary)
        env = os.environ.copy()
        env.update(
            {
                "KOGWISTAR_BACKEND": "sqlite",
                "KOGWISTAR_META_SQLITE_PATH": str(work / "runtime.sqlite3"),
                "KOGWISTAR_SERVER_HOST": "127.0.0.1",
                "KOGWISTAR_SERVER_PORT": str(port),
                "KOGWISTAR_SERVER_REQUIRED_ROLES": "rw",
                "KOGWISTAR_RUNTIME_NATIVE_INLINE": "true"
                if mode == "rust_inline"
                else "false",
            }
        )
        for key in ("JWT_SECRET", "JWT_ALG", "JWT_ISS", "JWT_AUD"):
            env.pop(key, None)
        process = subprocess.Popen(
            [str(server)],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            startup_seconds = _wait_ready(process, base_url)
            headers = {"x-kogwistar-roles": "rw"}
            run_id = f"seam-{mode}-{uuid.uuid4().hex}"
            with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
                if mode == "python_worker":
                    resolver = MappingStepResolver()

                    @resolver.register("benchmark.increment")
                    def increment(ctx: Any) -> RunSuccess:
                        return RunSuccess(
                            state_update=[
                                ("u", {"counter": int(ctx.state_view["counter"]) + 1})
                            ]
                        )

                    worker = RustRuntimeWorker.from_step_resolver(
                        base_url=base_url,
                        worker_id=f"worker-{uuid.uuid4().hex}",
                        journal_path=work / "worker.sqlite3",
                        step_resolver=resolver,
                        headers=headers,
                    )
                else:
                    worker = None
                submitted = client.post(
                    "/api/workflow/runs",
                    json=_workflow_payload(mode=mode, run_id=run_id, nodes=nodes),
                )
                if submitted.status_code != 202:
                    raise RuntimeError(f"run admission failed: {submitted.text}")
                started = time.perf_counter()
                try:
                    deadline = started + 300
                    while time.perf_counter() < deadline:
                        status = client.get(f"/api/workflow/runs/{run_id}")
                        status.raise_for_status()
                        state = status.json().get("status")
                        if state in {"succeeded", "failed", "cancelled", "suspended"}:
                            if state != "succeeded":
                                raise RuntimeError(f"run ended {state}: {status.text}")
                            break
                        if worker is not None:
                            worker.poll_once(limit=1, lease_seconds=60, run_id=run_id)
                        else:
                            time.sleep(0.005)
                    else:
                        raise TimeoutError(f"run {run_id} exceeded 300 seconds")
                finally:
                    if worker is not None:
                        worker.close()
                elapsed = time.perf_counter() - started
                checkpoints = client.get(f"/api/workflow/runs/{run_id}/checkpoints")
                checkpoints.raise_for_status()
                rows = checkpoints.json().get("checkpoints", [])
                latest = max(
                    rows,
                    key=lambda row: (row.get("step_seq", -1), row.get("event_seq", -1)),
                )
                final_state = latest.get("state")
                if (
                    not isinstance(final_state, dict)
                    or final_state.get("counter") != nodes
                ):
                    raise RuntimeError(f"final state mismatch: {final_state!r}")
                return {
                    "mode": mode,
                    "nodes": nodes,
                    "startup_seconds": startup_seconds,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": nodes / elapsed,
                    "final_state_digest": _digest({"counter": final_state["counter"]}),
                }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def run_benchmark(
    *,
    server: Path,
    nodes: int,
    samples: int,
    expected_cpus: float | None = None,
    expected_memory_bytes: int | None = None,
) -> dict[str, Any]:
    if not server.is_file():
        raise FileNotFoundError(f"Rust server not found: {server}")
    if nodes < 1 or samples < 2:
        raise ValueError("nodes must be positive and samples must be at least 2")
    if (expected_cpus is None) != (expected_memory_bytes is None):
        raise ValueError("CPU and memory expectations must be provided together")
    actual_cpu_max = _cgroup_value("cpu.max") if expected_cpus is not None else None
    actual_memory_max = (
        _cgroup_value("memory.max") if expected_memory_bytes is not None else None
    )
    limits_verified = (
        _verify_limits(
            actual_cpu_max, actual_memory_max, expected_cpus, expected_memory_bytes
        )
        if expected_cpus is not None and expected_memory_bytes is not None
        else None
    )
    if limits_verified is False:
        raise RuntimeError(
            f"resource limits mismatch: cpu.max={actual_cpu_max!r}, memory.max={actual_memory_max!r}"
        )
    measurements: dict[str, list[dict[str, Any]]] = {
        "python_worker": [],
        "rust_inline": [],
    }
    warmups: dict[str, dict[str, Any]] = {}
    for mode in measurements:
        warmups[mode] = _run_sample(server, ROOT, mode=mode, nodes=nodes)
        for _ in range(samples):
            measurements[mode].append(_run_sample(server, ROOT, mode=mode, nodes=nodes))
    digests = {
        row["final_state_digest"] for rows in measurements.values() for row in rows
    }
    if len(digests) != 1:
        raise RuntimeError("modes produced different final-state digests")
    return {
        "schema": "kogwistar-runtime-python-worker-seam-benchmark/v1",
        "status": "complete",
        "measurement_scope": "Rust SQLite scheduler; Python worker HTTP claim/result seam vs Rust inline steps",
        "nodes_per_run": nodes,
        "runtime_environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "rust_server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        },
        "warmups": warmups,
        "samples": measurements,
        "summary": {
            mode: {
                "elapsed_seconds": _summary([row["elapsed_seconds"] for row in rows]),
                "steps_per_second": _summary([row["steps_per_second"] for row in rows]),
                "startup_seconds": _summary([row["startup_seconds"] for row in rows]),
            }
            for mode, rows in measurements.items()
        },
        "final_state_digest": next(iter(digests)),
        "resource_limits": (
            "verified by cgroup"
            if limits_verified
            else "not constrained; local exploratory benchmark"
        ),
        "resource_limit_verification": {
            "verified": limits_verified,
            "expected_cpus": expected_cpus,
            "expected_memory_bytes": expected_memory_bytes,
            "observed_cpu_max": actual_cpu_max,
            "observed_memory_max": actual_memory_max,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cpus", type=float)
    parser.add_argument("--expected-memory-bytes", type=int)
    args = parser.parse_args()
    report = run_benchmark(
        server=args.server.resolve(),
        nodes=args.nodes,
        samples=args.samples,
        expected_cpus=args.expected_cpus,
        expected_memory_bytes=args.expected_memory_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
