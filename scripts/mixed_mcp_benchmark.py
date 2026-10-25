"""Compare the same deterministic graph-storage tools in MCP registry and HTTP."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Coroutine
import hashlib
import inspect
import json
import math
import os
import subprocess
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_NATIVE_BINARY_READY = False
DEFAULT_MODES = ("in_process", "rust_authority", "http", "rust_native")
MCP_OPERATION_TIMEOUT_SECONDS = 30
MCP_SESSION_TIMEOUT_SECONDS = 600
NATIVE_HTTP_TIMEOUT_SECONDS = 30
_AwaitedT = TypeVar("_AwaitedT")

import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite
from kogwistar.server.mcp_registry import McpRegistry
try:
    from .mixed_workload import (
        ConcurrentPhase,
        ROOT_NODE_ID,
        WorkloadOperation,
        build_graph_payloads,
        build_root_graph_payload,
        graph_edge_id,
        graph_node_id,
        previous_graph_node_id,
        write_concurrent_trace,
        write_trace,
    )
except ImportError:
    from mixed_workload import (
        ConcurrentPhase,
        ROOT_NODE_ID,
        WorkloadOperation,
        build_graph_payloads,
        build_root_graph_payload,
        graph_edge_id,
        graph_node_id,
        previous_graph_node_id,
        write_concurrent_trace,
        write_trace,
    )


@dataclass
class _ConcurrentMcpBarriers:
    start_step: threading.Barrier
    transaction_open: threading.Barrier
    precommit_reads_done: threading.Barrier
    commit_done: threading.Barrier
    step_done: threading.Barrier

    @classmethod
    def create(cls) -> _ConcurrentMcpBarriers:
        return cls(*(threading.Barrier(5, timeout=30) for _ in range(5)))

    @staticmethod
    def wait(barrier: threading.Barrier, label: str) -> None:
        try:
            barrier.wait()
        except threading.BrokenBarrierError as exc:
            raise RuntimeError(f"concurrent MCP {label} barrier broke") from exc

    def abort(self) -> None:
        for barrier in (
            self.start_step,
            self.transaction_open,
            self.precommit_reads_done,
            self.commit_done,
            self.step_done,
        ):
            barrier.abort()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


async def _await_mcp_operation(awaitable: Awaitable[_AwaitedT], operation: str) -> _AwaitedT:
    """Turn a stalled local MCP transport into a bounded benchmark failure."""

    try:
        return await asyncio.wait_for(awaitable, timeout=MCP_OPERATION_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise TimeoutError(
            f"MCP operation {operation!r} exceeded "
            f"{MCP_OPERATION_TIMEOUT_SECONDS}s"
        ) from exc


@asynccontextmanager
async def _bounded_streamable_http_client(url: str) -> AsyncIterator[tuple[Any, Any]]:
    """Bound MCP session setup/teardown as well as individual operations."""

    client = streamable_http_client(url, terminate_on_close=True)
    try:
        streams = await asyncio.wait_for(
            client.__aenter__(), timeout=MCP_SESSION_TIMEOUT_SECONDS
        )
    except BaseException:
        await asyncio.wait_for(
            client.__aexit__(*sys.exc_info()), timeout=MCP_SESSION_TIMEOUT_SECONDS
        )
        raise

    try:
        yield streams
    except BaseException:
        await asyncio.wait_for(
            client.__aexit__(*sys.exc_info()), timeout=MCP_SESSION_TIMEOUT_SECONDS
        )
        raise
    else:
        await asyncio.wait_for(
            client.__aexit__(None, None, None), timeout=MCP_SESSION_TIMEOUT_SECONDS
        )


def _sqlite_footprint_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.is_file()
    )


def _trace_tool_arguments(step: WorkloadOperation) -> dict[str, object]:
    arguments: dict[str, object] = {
        "index": step["index"],
        "expected_entity_id": step["expected_entity_id"],
        "expected_edge_id": step["expected_edge_id"],
        "expected_source_id": step["expected_source_id"],
    }
    if step["operation"] == "write_record":
        node_payload, edge_payload = build_graph_payloads(
            step["index"], vector_sha256="0" * 64
        )
        arguments["node_payload"] = node_payload
        arguments["edge_payload"] = edge_payload
    return arguments


def _serve_uvicorn(server: uvicorn.Server, sock: socket.socket) -> None:
    """Run the local benchmark server without Windows Proactor reset noise."""
    if os.name == "nt":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve(sockets=[sock]))
        return
    server.run(sockets=[sock])


def _run_async(awaitable: Coroutine[Any, Any, _AwaitedT]) -> _AwaitedT:
    """Run benchmark coroutines without Windows Proactor self-pipe stalls."""

    if os.name == "nt":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(awaitable)
    return asyncio.run(awaitable)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * percentile + 0.999999) - 1)]


def _operation_counts(outcomes: list[int | BaseException]) -> list[int]:
    counts: list[int] = []
    failures: list[BaseException] = []
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            failures.append(outcome)
        else:
            counts.append(outcome)
    if failures:
        raise BaseExceptionGroup("concurrent MCP actors failed", failures)
    return counts


def _sample_variation(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "stdev": deviation,
        "coefficient_of_variation_pct": deviation * 100 / mean if mean else 0.0,
        "normal_approx_95pct_ci_low": mean - 1.96 * deviation / math.sqrt(len(values)),
        "normal_approx_95pct_ci_high": mean + 1.96 * deviation / math.sqrt(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _make_registry(
    path: Path,
    *,
    authority: str = "python",
    concurrent: _ConcurrentMcpBarriers | None = None,
    sync_concurrent: bool = False,
) -> tuple[McpRegistry, EngineSQLite | RustEngineSQLite]:
    store_type = RustEngineSQLite if authority == "rust" else EngineSQLite
    store = store_type(path.parent, filename=path.name)
    store.ensure_initialized()

    def _write_graph_projection(
        target: EngineSQLite | RustEngineSQLite,
        kind: str,
        entity_id: str,
        payload: dict[str, object],
        seq: int,
    ) -> None:
        target.replace_named_projection(
            "mcp_benchmark_projection",
            f"{kind}:{entity_id}",
            {"entity_kind": kind, "entity_id": entity_id, "entity": payload},
            last_authoritative_seq=seq,
            last_materialized_seq=seq,
            projection_schema_version=1,
            materialization_status="ready",
        )

    root_payload = build_root_graph_payload()
    with store.transaction():
        root_seq = store.append_entity_event(
            namespace="mcp_benchmark",
            event_id="mcp-root-node",
            entity_kind="node",
            entity_id=ROOT_NODE_ID,
            op="UPSERT",
            payload_json=_canonical(root_payload),
        )
        _write_graph_projection(
            store, "node", ROOT_NODE_ID, root_payload, root_seq
        )
        if concurrent is not None:
            store.replace_named_projection(
                "mcp_benchmark_projection",
                "mcp-head",
                {
                    "revision": 0,
                    "entity_id": ROOT_NODE_ID,
                    "edge_id": None,
                    "previous_entity_id": None,
                    "node_seq": root_seq,
                    "edge_seq": 0,
                },
                last_authoritative_seq=root_seq,
                last_materialized_seq=root_seq,
                projection_schema_version=1,
                materialization_status="ready",
            )
    registry = McpRegistry("kogwistar-mixed-workload-benchmark")

    def write_record_sync(
        index: int,
        expected_entity_id: str,
        expected_edge_id: str,
        expected_source_id: str,
        node_payload: dict[str, object],
        edge_payload: dict[str, object],
    ) -> dict[str, object]:
        if (
            node_payload.get("id") != expected_entity_id
            or edge_payload.get("id") != expected_edge_id
            or edge_payload.get("source_ids") != [expected_source_id]
            or edge_payload.get("target_ids") != [expected_entity_id]
        ):
            raise RuntimeError("MCP write bundle does not match the expected graph operation")
        with store.transaction():
            node_seq = store.append_entity_event(
                namespace="mcp_benchmark",
                event_id=f"mcp-node-event-{index:05d}",
                entity_kind="node",
                entity_id=expected_entity_id,
                op="UPSERT",
                payload_json=_canonical(node_payload),
            )
            edge_seq = store.append_entity_event(
                namespace="mcp_benchmark",
                event_id=f"mcp-edge-event-{index:05d}",
                entity_kind="edge",
                entity_id=expected_edge_id,
                op="UPSERT",
                payload_json=_canonical(edge_payload),
            )
            _write_graph_projection(store, "node", expected_entity_id, node_payload, node_seq)
            _write_graph_projection(store, "edge", expected_edge_id, edge_payload, edge_seq)
            if concurrent is not None:
                store.replace_named_projection(
                    "mcp_benchmark_projection",
                    "mcp-head",
                    {
                        "revision": index + 1,
                        "entity_id": expected_entity_id,
                        "edge_id": expected_edge_id,
                        "previous_entity_id": previous_graph_node_id(index),
                        "node_seq": node_seq,
                        "edge_seq": edge_seq,
                    },
                    last_authoritative_seq=edge_seq,
                    last_materialized_seq=edge_seq,
                    projection_schema_version=1,
                    materialization_status="ready",
                )
                # Keep the write uncommitted while readers verify the prior head.
                concurrent.wait(concurrent.transaction_open, "transaction-open")
                concurrent.wait(concurrent.precommit_reads_done, "precommit-read")
        return {"seq": edge_seq, "entity_id": expected_entity_id, "edge_id": expected_edge_id}

    if concurrent is None or sync_concurrent:

        @registry.tool(name="write_record")
        def write_record_tool(
            index: int,
            expected_entity_id: str,
            expected_edge_id: str,
            expected_source_id: str,
            node_payload: dict[str, object],
            edge_payload: dict[str, object],
        ) -> dict[str, object]:
            return write_record_sync(
                index, expected_entity_id, expected_edge_id, expected_source_id,
                node_payload, edge_payload,
            )

    else:

        @registry.tool(name="write_record")
        async def write_record_concurrent_tool(
            index: int,
            expected_entity_id: str,
            expected_edge_id: str,
            expected_source_id: str,
            node_payload: dict[str, object],
            edge_payload: dict[str, object],
        ) -> dict[str, object]:
            # MCP's in-process registry invokes sync tools inline, and HTTP
            # dispatch must not block its event loop while the writer waits on
            # readers. Run this transactional handler in a worker in both modes.
            return await asyncio.to_thread(
                write_record_sync, index, expected_entity_id, expected_edge_id,
                expected_source_id, node_payload, edge_payload,
            )

    @registry.tool(name="read_event")
    def read_event(
        index: int,
        expected_entity_id: str,
        expected_edge_id: str,
        expected_source_id: str,
    ) -> dict[str, object]:
        node_seq = 2 + 2 * index
        rows = list(
            store.iter_entity_events(
                namespace="mcp_benchmark",
                from_seq=node_seq,
                to_seq=node_seq + 1,
                batch_size=2,
            )
        )
        if (
            len(rows) != 2
            or (rows[0][1], rows[0][2]) != ("node", expected_entity_id)
            or (rows[1][1], rows[1][2]) != ("edge", expected_edge_id)
        ):
            raise RuntimeError("MCP graph event read did not observe the expected write")
        edge_payload = json.loads(rows[1][4])
        if (
            not isinstance(edge_payload, dict)
            or edge_payload.get("source_ids") != [expected_source_id]
            or edge_payload.get("target_ids") != [expected_entity_id]
        ):
            raise RuntimeError("MCP edge read returned incorrect graph endpoints")
        return {"seq": rows[1][0], "entity_id": expected_entity_id, "edge_id": expected_edge_id}

    @registry.tool(name="read_projection")
    def read_projection(
        index: int,
        expected_entity_id: str,
        expected_edge_id: str,
        expected_source_id: str,
    ) -> dict[str, object]:
        node_projection = store.get_named_projection(
            "mcp_benchmark_projection", f"node:{expected_entity_id}"
        )
        edge_projection = store.get_named_projection(
            "mcp_benchmark_projection", f"edge:{expected_edge_id}"
        )
        node_value = node_projection.get("payload") if node_projection else None
        edge_value = edge_projection.get("payload") if edge_projection else None
        if (
            not isinstance(node_value, dict)
            or not isinstance(node_value.get("entity"), dict)
            or node_value["entity"].get("id") != expected_entity_id
            or not isinstance(edge_value, dict)
            or not isinstance(edge_value.get("entity"), dict)
            or edge_value["entity"].get("source_ids")
            != [expected_source_id]
            or edge_value["entity"].get("target_ids") != [expected_entity_id]
        ):
            raise RuntimeError("MCP graph projections do not match the trace")
        return {"index": index, "entity_id": expected_entity_id, "edge_id": expected_edge_id}

    if concurrent is not None:

        def read_head_sync(expected_revision: int) -> dict[str, object]:
            projection = store.get_named_projection(
                "mcp_benchmark_projection", "mcp-head"
            )
            if projection is None:
                raise RuntimeError("MCP head projection is missing")
            payload = projection.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("MCP head projection payload is not an object")
            revision = payload.get("revision")
            entity_id = payload.get("entity_id")
            edge_id = payload.get("edge_id")
            previous_id = payload.get("previous_entity_id")
            node_seq = payload.get("node_seq")
            edge_seq = payload.get("edge_seq")
            if (
                not isinstance(revision, int)
                or not isinstance(entity_id, str)
                or (edge_id is not None and not isinstance(edge_id, str))
                or (previous_id is not None and not isinstance(previous_id, str))
                or not isinstance(node_seq, int)
                or not isinstance(edge_seq, int)
            ):
                raise RuntimeError("MCP head projection graph fields are malformed")
            if revision != expected_revision:
                raise RuntimeError(
                    f"MCP reader saw revision {revision}; expected {expected_revision}"
                )
            expected_entity_id = ROOT_NODE_ID if revision == 0 else graph_node_id(revision - 1)
            expected_edge_id = None if revision == 0 else graph_edge_id(revision - 1)
            expected_previous = None if revision == 0 else previous_graph_node_id(revision - 1)
            if (
                entity_id != expected_entity_id
                or edge_id != expected_edge_id
                or previous_id != expected_previous
                or node_seq != (1 if revision == 0 else 2 * revision)
                or edge_seq != (0 if revision == 0 else node_seq + 1)
            ):
                raise RuntimeError("MCP head projection does not match graph revision")
            if revision == 0:
                rows = list(
                    store.iter_entity_events(
                        namespace="mcp_benchmark",
                        from_seq=node_seq,
                        to_seq=node_seq,
                        batch_size=1,
                    )
                )
                if len(rows) != 1 or (rows[0][1], rows[0][2]) != ("node", entity_id):
                    raise RuntimeError("MCP graph root head points to a mismatched node")
            else:
                rows = list(
                    store.iter_entity_events(
                        namespace="mcp_benchmark",
                        from_seq=node_seq,
                        to_seq=edge_seq,
                        batch_size=2,
                    )
                )
                if (
                    len(rows) != 2
                    or (rows[0][1], rows[0][2]) != ("node", entity_id)
                    or (rows[1][1], rows[1][2]) != ("edge", edge_id)
                ):
                    raise RuntimeError("MCP graph head points to mismatched events")
                edge_payload = json.loads(rows[1][4])
                if (
                    not isinstance(edge_payload, dict)
                    or edge_payload.get("source_ids") != [previous_id]
                    or edge_payload.get("target_ids") != [entity_id]
                ):
                    raise RuntimeError("MCP graph head edge has mismatched endpoints")
            return {"revision": revision, "entity_id": entity_id, "edge_id": edge_id}

        if sync_concurrent:

            @registry.tool(name="read_head")
            def read_head(expected_revision: int) -> dict[str, object]:
                return read_head_sync(expected_revision)

        else:

            @registry.tool(name="read_head")
            async def read_head(expected_revision: int) -> dict[str, object]:
                return await asyncio.to_thread(read_head_sync, expected_revision)

    return registry, store


@contextmanager
def _reuse_rust_store_session(store: EngineSQLite | RustEngineSQLite):
    """Keep one Rust SQLite handle alive for a sequential authority trace."""

    if isinstance(store, RustEngineSQLite):
        with store.session():
            yield
        return
    yield


def _result_payload(result: object) -> dict[str, object]:
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, dict):
        raise RuntimeError("MCP tool returned no structured result")
    return structured


def _call_registry_sync(
    registry: McpRegistry, name: str, arguments: dict[str, object]
) -> dict[str, object]:
    """Dispatch synchronous benchmark tools without creating an event loop."""

    record = registry._record_for(name)
    if record is None:
        raise RuntimeError(f"sync MCP tool {name!r} is missing")
    values = record.input_model.model_validate(arguments)
    kwargs = {
        field: getattr(values, field)
        for field in record.input_model.model_fields
    }
    result = record.function(**kwargs)
    if inspect.isawaitable(result):
        raise RuntimeError(f"MCP benchmark tool {name!r} unexpectedly became async")
    if not isinstance(result, dict):
        raise RuntimeError("sync MCP tool returned a non-object result")
    payload = json.loads(json.dumps(result, default=str))
    if not isinstance(payload, dict):
        raise RuntimeError("sync MCP response serialization changed its shape")
    return payload


def _final_state(path: Path) -> dict[str, Any]:
    store = EngineSQLite(path.parent, filename=path.name)
    events = [
        list(row)
        for row in store.iter_entity_events(
            namespace="mcp_benchmark", from_seq=1, batch_size=100_001
        )
    ]
    projections = [
        {key: value for key, value in row.items() if key != "updated_at_ms"}
        for row in store.list_named_projections("mcp_benchmark_projection")
    ]
    return {"events": events, "projections": projections}


async def _serve_http(
    registry: McpRegistry,
    trace: list[WorkloadOperation],
    trace_arguments: list[dict[str, object]],
) -> dict[str, Any]:
    app = registry.http_app(path="/mcp")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="error",
        lifespan="on",
        timeout_graceful_shutdown=15,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_serve_uvicorn, args=(server, sock), daemon=True)
    startup_started = time.perf_counter_ns()
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=3)
        raise TimeoutError("local MCP HTTP server failed to become ready")
    startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000

    timings: dict[str, list[float]] = {"write_record": [], "read_event": [], "read_projection": []}
    elapsed_ms = 0.0
    try:
        async with _bounded_streamable_http_client(f"http://{host}:{port}/mcp") as (read, write):
            async with ClientSession(read, write) as session:
                init_started = time.perf_counter_ns()
                await _await_mcp_operation(session.initialize(), "initialize")
                startup_ms += (time.perf_counter_ns() - init_started) / 1_000_000
                names = {
                    tool.name
                    for tool in (
                        await _await_mcp_operation(session.list_tools(), "list_tools")
                    ).tools
                }
                if not {"write_record", "read_event", "read_projection"} <= names:
                    raise RuntimeError("MCP HTTP surface omitted required benchmark tools")
                wall_started = time.perf_counter_ns()
                for step, arguments in zip(trace, trace_arguments, strict=True):
                    operation = step["operation"]
                    started = time.perf_counter_ns()
                    result = await _await_mcp_operation(
                        session.call_tool(operation, arguments=arguments),
                        operation,
                    )
                    payload = _result_payload(result)
                    if payload.get("entity_id") != step["expected_entity_id"]:
                        raise RuntimeError("MCP HTTP returned an unexpected entity id")
                    if operation == "read_projection" and payload.get("index") != step["index"]:
                        raise RuntimeError("MCP HTTP returned an unexpected projection index")
                    timings[operation].append((time.perf_counter_ns() - started) / 1_000_000)
                elapsed_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 20)
        if thread.is_alive():
            server.force_exit = True
            await asyncio.to_thread(thread.join, 5)
        sock.close()
        if thread.is_alive():
            raise TimeoutError("local MCP HTTP server did not stop cleanly")
    return {
        "startup_ms": startup_ms,
        "elapsed_ms": elapsed_ms,
        "per_operation": timings,
    }


async def _serve_http_concurrent(
    registry: McpRegistry,
    phases: list[ConcurrentPhase],
    barriers: _ConcurrentMcpBarriers,
    graph_payloads: dict[int, tuple[dict[str, object], dict[str, object]]],
) -> dict[str, Any]:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    endpoint = "/mcp"
    managers: dict[str, StreamableHTTPSessionManager] = {}

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        manager = StreamableHTTPSessionManager(
            registry.server,
            json_response=True,
            stateless=True,
        )
        managers["active"] = manager
        async with manager.run():
            try:
                yield
            finally:
                managers.pop("active", None)

    async def handle_request(scope: Scope, receive: Receive, send: Send) -> None:
        request_path = str(scope.get("path") or "")
        if request_path not in {endpoint, endpoint + "/"}:
            await PlainTextResponse("Not Found", status_code=404)(
                scope, receive, send
            )
            return
        manager = managers.get("active")
        if manager is None:
            await PlainTextResponse("MCP server is not started", status_code=503)(
                scope, receive, send
            )
            return
        await manager.handle_request(scope, receive, send)

    app = Starlette(
        routes=[Mount("/", app=handle_request)],
        lifespan=lifespan,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="error",
            lifespan="on",
            timeout_graceful_shutdown=3,
        )
    )
    thread = threading.Thread(target=_serve_uvicorn, args=(server, sock), daemon=True)
    startup_started = time.perf_counter_ns()
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=3)
        sock.close()
        raise TimeoutError("concurrent MCP HTTP server failed to become ready")
    startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000
    timings: dict[str, list[float]] = {
        "write_record": [],
        "read_head": [],
    }
    timings_lock = threading.Lock()

    async def call(session: ClientSession, name: str, arguments: dict[str, object]) -> dict[str, object]:
        try:
            call_started = time.perf_counter_ns()
            result = await _await_mcp_operation(
                session.call_tool(name, arguments=arguments), name
            )
            payload = _result_payload(result)
            with timings_lock:
                timings[name].append((time.perf_counter_ns() - call_started) / 1_000_000)
            return payload
        except Exception:
            barriers.abort()
            raise

    async def actor(actor_id: int | None) -> int:
        operations = 0
        async with _bounded_streamable_http_client(f"http://{host}:{port}/mcp") as (read, write):
            async with ClientSession(read, write) as session:
                await _await_mcp_operation(session.initialize(), "initialize")
                tools = {
                    tool.name
                    for tool in (
                        await _await_mcp_operation(session.list_tools(), "list_tools")
                    ).tools
                }
                if not {"write_record", "read_head"} <= tools:
                    raise RuntimeError("concurrent MCP HTTP tools are missing")
                for phase in phases:
                    for step in phase["steps"]:
                        await asyncio.to_thread(
                            barriers.wait, barriers.start_step, "step-start"
                        )
                        if actor_id is None:
                            index = step["write_index"]
                            entity_id = graph_node_id(index)
                            edge_id = graph_edge_id(index)
                            source_id = previous_graph_node_id(index)
                            node_payload, edge_payload = graph_payloads[index]
                            result = await call(
                                session,
                                "write_record",
                                {
                                    "index": index,
                                    "expected_entity_id": entity_id,
                                    "expected_edge_id": edge_id,
                                    "expected_source_id": source_id,
                                    "node_payload": node_payload,
                                    "edge_payload": edge_payload,
                                },
                            )
                            if result.get("entity_id") != entity_id:
                                raise RuntimeError("concurrent MCP write returned wrong entity")
                            operations += 1
                        else:
                            await asyncio.to_thread(
                                barriers.wait,
                                barriers.transaction_open,
                                "transaction-open",
                            )
                            reads = {
                                item["visibility"]: item
                                for item in step["reads"]
                                if item["reader_id"] == actor_id
                            }
                            before = reads.get("before_commit")
                            if before is not None:
                                result = await call(
                                    session,
                                    "read_head",
                                    {"expected_revision": before["minimum_revision"]},
                                )
                                operations += 1
                                if result.get("revision") != before["minimum_revision"]:
                                    raise RuntimeError("MCP precommit visibility mismatch")
                        if actor_id is not None:
                            await asyncio.to_thread(
                                barriers.wait,
                                barriers.precommit_reads_done,
                                "precommit-read",
                            )
                        else:
                            # The synchronous tool handler waits at this point while
                            # the writer transaction is still open.
                            pass
                        await asyncio.to_thread(
                            barriers.wait, barriers.commit_done, "commit"
                        )
                        if actor_id is not None:
                            reads = {
                                item["visibility"]: item
                                for item in step["reads"]
                                if item["reader_id"] == actor_id
                            }
                            after = reads.get("after_commit")
                            if after is not None:
                                result = await call(
                                    session,
                                    "read_head",
                                    {"expected_revision": after["minimum_revision"]},
                                )
                                operations += 1
                                if result.get("revision") != after["minimum_revision"]:
                                    raise RuntimeError("MCP postcommit visibility mismatch")
                        await asyncio.to_thread(
                            barriers.wait, barriers.step_done, "step-done"
                        )
        return operations

    async def guarded_actor(actor_id: int | None) -> int:
        try:
            return await actor(actor_id)
        except BaseException:
            # Setup failures must release peers waiting on a shared phase.
            barriers.abort()
            raise

    elapsed_started = time.perf_counter_ns()
    try:
        outcomes = await asyncio.gather(
            guarded_actor(None),
            *(guarded_actor(reader_id) for reader_id in range(4)),
            return_exceptions=True,
        )
        operation_counts = _operation_counts(outcomes)
        elapsed_ms = (time.perf_counter_ns() - elapsed_started) / 1_000_000
    finally:
        server.should_exit = True
        await asyncio.to_thread(thread.join, 5)
        if thread.is_alive():
            server.force_exit = True
            await asyncio.to_thread(thread.join, 3)
        sock.close()
        if thread.is_alive():
            raise TimeoutError("concurrent MCP HTTP server did not stop cleanly")
    return {
        "startup_ms": startup_ms,
        "elapsed_ms": elapsed_ms,
        "operations": sum(operation_counts),
        "per_operation": timings,
    }


async def _run_in_process_concurrent(
    registry: McpRegistry,
    phases: list[ConcurrentPhase],
    barriers: _ConcurrentMcpBarriers,
    graph_payloads: dict[int, tuple[dict[str, object], dict[str, object]]],
) -> dict[str, Any]:
    timings: dict[str, list[float]] = {"write_record": [], "read_head": []}
    timings_lock = threading.Lock()

    async def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
        try:
            call_started = time.perf_counter_ns()
            payload = _result_payload(await registry.call_tool(name, arguments))
            with timings_lock:
                timings[name].append((time.perf_counter_ns() - call_started) / 1_000_000)
            return payload
        except Exception:
            barriers.abort()
            raise

    async def actor(actor_id: int | None) -> int:
        operations = 0
        for phase in phases:
            for step in phase["steps"]:
                await asyncio.to_thread(barriers.wait, barriers.start_step, "step-start")
                if actor_id is None:
                    index = step["write_index"]
                    entity_id = graph_node_id(index)
                    edge_id = graph_edge_id(index)
                    source_id = previous_graph_node_id(index)
                    node_payload, edge_payload = graph_payloads[index]
                    result = await call(
                        "write_record",
                        {
                            "index": index,
                            "expected_entity_id": entity_id,
                            "expected_edge_id": edge_id,
                            "expected_source_id": source_id,
                            "node_payload": node_payload,
                            "edge_payload": edge_payload,
                        },
                    )
                    if result.get("entity_id") != entity_id:
                        raise RuntimeError("concurrent MCP write returned wrong entity")
                    operations += 1
                else:
                    await asyncio.to_thread(
                        barriers.wait, barriers.transaction_open, "transaction-open"
                    )
                    reads = {
                        item["visibility"]: item
                        for item in step["reads"]
                        if item["reader_id"] == actor_id
                    }
                    before = reads.get("before_commit")
                    if before is not None:
                        result = await call(
                            "read_head",
                            {"expected_revision": before["minimum_revision"]},
                        )
                        if result.get("revision") != before["minimum_revision"]:
                            raise RuntimeError("MCP precommit visibility mismatch")
                        operations += 1
                    await asyncio.to_thread(
                        barriers.wait,
                        barriers.precommit_reads_done,
                        "precommit-read",
                    )
                await asyncio.to_thread(barriers.wait, barriers.commit_done, "commit")
                if actor_id is not None:
                    reads = {
                        item["visibility"]: item
                        for item in step["reads"]
                        if item["reader_id"] == actor_id
                    }
                    after = reads.get("after_commit")
                    if after is not None:
                        result = await call(
                            "read_head",
                            {"expected_revision": after["minimum_revision"]},
                        )
                        if result.get("revision") != after["minimum_revision"]:
                            raise RuntimeError("MCP postcommit visibility mismatch")
                        operations += 1
                await asyncio.to_thread(barriers.wait, barriers.step_done, "step-done")
        return operations

    started = time.perf_counter_ns()
    outcomes = await asyncio.gather(
        actor(None), *(actor(reader_id) for reader_id in range(4)),
        return_exceptions=True,
    )
    operation_counts = _operation_counts(outcomes)
    return {
        "startup_ms": 0.0,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000,
        "operations": sum(operation_counts),
        "per_operation": timings,
    }


def _run_sync_threadpool_concurrent(
    registry: McpRegistry,
    phases: list[ConcurrentPhase],
    barriers: _ConcurrentMcpBarriers,
    graph_payloads: dict[int, tuple[dict[str, object], dict[str, object]]],
) -> dict[str, Any]:
    """Run the same MCP tool handlers from five synchronous actor threads."""
    from concurrent.futures import ThreadPoolExecutor

    timings: dict[str, list[float]] = {"write_record": [], "read_head": []}
    timings_lock = threading.Lock()

    def call(name: str, arguments: dict[str, object]) -> dict[str, object]:
        call_started = time.perf_counter_ns()
        try:
            record = registry._record_for(name)
            if record is None:
                raise RuntimeError(f"sync MCP benchmark tool {name!r} is missing")
            values = record.input_model.model_validate(arguments)
            kwargs = {
                field: getattr(values, field)
                for field in record.input_model.model_fields
            }
            result = record.function(**kwargs)
            if not isinstance(result, dict):
                raise RuntimeError("sync MCP tool returned a non-object result")
            # Include response serialization in the same way as the async registry path.
            payload = json.loads(json.dumps(result, default=str))
            if not isinstance(payload, dict):
                raise RuntimeError("sync MCP response serialization changed its shape")
            return payload
        except Exception:
            barriers.abort()
            raise
        finally:
            with timings_lock:
                timings[name].append((time.perf_counter_ns() - call_started) / 1_000_000)

    def actor(actor_id: int | None) -> int:
        operations = 0
        for phase in phases:
            for step in phase["steps"]:
                barriers.wait(barriers.start_step, "step-start")
                if actor_id is None:
                    index = step["write_index"]
                    entity_id = graph_node_id(index)
                    edge_id = graph_edge_id(index)
                    source_id = previous_graph_node_id(index)
                    node_payload, edge_payload = graph_payloads[index]
                    result = call(
                        "write_record",
                        {
                            "index": index,
                            "expected_entity_id": entity_id,
                            "expected_edge_id": edge_id,
                            "expected_source_id": source_id,
                            "node_payload": node_payload,
                            "edge_payload": edge_payload,
                        },
                    )
                    if result.get("entity_id") != entity_id:
                        raise RuntimeError("sync MCP write returned wrong entity")
                    operations += 1
                else:
                    barriers.wait(barriers.transaction_open, "transaction-open")
                    reads = {
                        item["visibility"]: item
                        for item in step["reads"]
                        if item["reader_id"] == actor_id
                    }
                    before = reads.get("before_commit")
                    if before is not None:
                        result = call(
                            "read_head",
                            {"expected_revision": before["minimum_revision"]},
                        )
                        if result.get("revision") != before["minimum_revision"]:
                            raise RuntimeError("sync MCP precommit visibility mismatch")
                        operations += 1
                    barriers.wait(barriers.precommit_reads_done, "precommit-read")
                barriers.wait(barriers.commit_done, "commit")
                if actor_id is not None:
                    reads = {
                        item["visibility"]: item
                        for item in step["reads"]
                        if item["reader_id"] == actor_id
                    }
                    after = reads.get("after_commit")
                    if after is not None:
                        result = call(
                            "read_head",
                            {"expected_revision": after["minimum_revision"]},
                        )
                        if result.get("revision") != after["minimum_revision"]:
                            raise RuntimeError("sync MCP postcommit visibility mismatch")
                        operations += 1
                barriers.wait(barriers.step_done, "step-done")
        return operations

    started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(actor, actor_id) for actor_id in (None, 0, 1, 2, 3)]
        outcomes = [future.result() for future in futures]
    return {
        "startup_ms": 0.0,
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000,
        "operations": sum(outcomes),
        "per_operation": timings,
    }


def _run_concurrent_one(
    path: Path,
    phases: list[ConcurrentPhase],
    *,
    authority: str,
    transport: str,
) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to collect concurrent MCP process metrics") from exc
    process = psutil.Process()
    graph_payloads = {
        step["write_index"]: build_graph_payloads(
            step["write_index"], vector_sha256="0" * 64
        )
        for phase in phases
        for step in phase["steps"]
    }
    rss_start = int(process.memory_info().rss)
    cpu_started = time.process_time_ns()
    barriers = _ConcurrentMcpBarriers.create()
    registry, _store = _make_registry(path, authority=authority, concurrent=barriers)
    if transport == "http":
        transport_result = _run_async(
            _serve_http_concurrent(registry, phases, barriers, graph_payloads)
        )
    else:
        transport_result = _run_async(
            _run_in_process_concurrent(registry, phases, barriers, graph_payloads)
        )
    state = _final_state(path)
    head = next(
        (
            item["payload"]
            for item in state["projections"]
            if item["key"] == "mcp-head"
        ),
        None,
    )
    expected_writes = sum(len(phase["steps"]) for phase in phases)
    if (
        not isinstance(head, dict)
        or head.get("revision") != expected_writes
        or len(state["events"]) != 1 + 2 * expected_writes
    ):
        raise RuntimeError("concurrent MCP final graph state is incomplete")
    elapsed_ms = float(transport_result["elapsed_ms"])
    operation_count = int(transport_result["operations"])
    if operation_count < 1 or elapsed_ms <= 0:
        raise RuntimeError("concurrent MCP trace reported no work or invalid elapsed time")
    rss_end = int(process.memory_info().rss)
    operation_latencies = transport_result["per_operation"]
    expected_reads = sum(
        len(step["reads"]) for phase in phases for step in phase["steps"]
    )
    if (
        len(operation_latencies["write_record"]) != expected_writes
        or len(operation_latencies["read_head"]) != expected_reads
        or operation_count != expected_writes + expected_reads
    ):
        raise RuntimeError("concurrent MCP trace skipped or duplicated an operation")
    return {
        "authority": authority,
        "transport": transport,
        "state_digest": _digest(state),
        "elapsed_ms": elapsed_ms,
        "startup_ms": transport_result["startup_ms"],
        "operations": operation_count,
        "throughput_ops_per_sec": operation_count * 1000 / elapsed_ms,
        "host_process_cpu_elapsed_ms": (time.process_time_ns() - cpu_started) / 1_000_000,
        "host_process_rss_start_bytes": rss_start,
        "host_process_rss_end_bytes": rss_end,
        "host_process_rss_delta_bytes": rss_end - rss_start,
        "sqlite_database_footprint_bytes": _sqlite_footprint_bytes(path),
        "per_operation": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
            for name, values in operation_latencies.items()
        },
    }


def _run_sync_threadpool_one(
    path: Path,
    phases: list[ConcurrentPhase],
) -> dict[str, Any]:
    """Measure synchronous concurrent MCP calls with the canonical trace."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to collect concurrent MCP process metrics") from exc
    process = psutil.Process()
    graph_payloads = {
        step["write_index"]: build_graph_payloads(
            step["write_index"], vector_sha256="0" * 64
        )
        for phase in phases
        for step in phase["steps"]
    }
    rss_start = int(process.memory_info().rss)
    cpu_started = time.process_time_ns()
    barriers = _ConcurrentMcpBarriers.create()
    registry, _store = _make_registry(
        path, authority="python", concurrent=barriers, sync_concurrent=True
    )
    transport_result = _run_sync_threadpool_concurrent(
        registry, phases, barriers, graph_payloads
    )
    state = _final_state(path)
    head = next(
        (item["payload"] for item in state["projections"] if item["key"] == "mcp-head"),
        None,
    )
    expected_writes = sum(len(phase["steps"]) for phase in phases)
    if (
        not isinstance(head, dict)
        or head.get("revision") != expected_writes
        or len(state["events"]) != 1 + 2 * expected_writes
    ):
        raise RuntimeError("sync concurrent MCP final graph state is incomplete")
    elapsed_ms = float(transport_result["elapsed_ms"])
    operation_count = int(transport_result["operations"])
    expected_reads = sum(len(step["reads"]) for phase in phases for step in phase["steps"])
    operation_latencies = transport_result["per_operation"]
    if (
        elapsed_ms <= 0
        or operation_count != expected_writes + expected_reads
        or len(operation_latencies["write_record"]) != expected_writes
        or len(operation_latencies["read_head"]) != expected_reads
    ):
        raise RuntimeError("sync concurrent MCP trace skipped or duplicated work")
    rss_end = int(process.memory_info().rss)
    return {
        "authority": "python",
        "transport": "sync_threadpool",
        "state_digest": _digest(state),
        "elapsed_ms": elapsed_ms,
        "startup_ms": 0.0,
        "operations": operation_count,
        "throughput_ops_per_sec": operation_count * 1000 / elapsed_ms,
        "host_process_cpu_elapsed_ms": (time.process_time_ns() - cpu_started) / 1_000_000,
        "host_process_rss_start_bytes": rss_start,
        "host_process_rss_end_bytes": rss_end,
        "host_process_rss_delta_bytes": rss_end - rss_start,
        "sqlite_database_footprint_bytes": _sqlite_footprint_bytes(path),
        "per_operation": {
            name: {
                "count": len(values),
                "p50_ms": statistics.median(values),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
            for name, values in operation_latencies.items()
        },
    }


async def _run_one(
    path: Path,
    trace: list[WorkloadOperation],
    mode: str,
    *,
    trace_arguments: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to collect MCP benchmark RSS metrics") from exc
    if trace_arguments is None:
        trace_arguments = [_trace_tool_arguments(step) for step in trace]
    elif len(trace_arguments) != len(trace):
        raise ValueError("trace arguments must have one entry per workload operation")
    host_process = psutil.Process()
    host_rss_start = int(host_process.memory_info().rss)
    host_cpu_started = time.process_time_ns()
    started = time.perf_counter_ns()
    timings: dict[str, list[float]] = {"write_record": [], "read_event": [], "read_projection": []}
    registry = None
    startup_ms = 0.0
    native_result: dict[str, Any] = {}
    if mode in {"in_process", "rust_authority"}:
        registry, _store = _make_registry(
            path,
            authority="rust" if mode == "rust_authority" else "python",
        )
        startup_ms = (time.perf_counter_ns() - started) / 1_000_000
        wall_started = time.perf_counter_ns()
        for step, arguments in zip(trace, trace_arguments, strict=True):
            operation = step["operation"]
            started = time.perf_counter_ns()
            payload = _call_registry_sync(registry, operation, arguments)
            if payload.get("entity_id") != step["expected_entity_id"]:
                raise RuntimeError("in-process MCP returned an unexpected entity id")
            if operation == "read_projection" and payload.get("index") != step["index"]:
                raise RuntimeError("in-process MCP returned an unexpected projection index")
            timings[operation].append((time.perf_counter_ns() - started) / 1_000_000)
        elapsed_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
    elif mode == "http":
        registry, _store = _make_registry(path)
        startup_ms = (time.perf_counter_ns() - started) / 1_000_000
        http_result = await _serve_http(registry, trace, trace_arguments)
        startup_ms += http_result["startup_ms"]
        timings = http_result["per_operation"]
        elapsed_ms = http_result["elapsed_ms"]
    else:
        native_result = await asyncio.to_thread(
            _run_native_server, path, trace, trace_arguments
        )
        startup_ms = native_result["startup_ms"]
        timings = native_result["per_operation"]
        elapsed_ms = native_result["elapsed_ms"]

    state = _final_state(path)
    host_cpu_elapsed_ms = (time.process_time_ns() - host_cpu_started) / 1_000_000
    host_rss_end = int(host_process.memory_info().rss)
    result = {
        "state_digest": _digest(state),
        "elapsed_ms": elapsed_ms,
        "startup_ms": startup_ms,
        "operations": len(trace),
        "throughput_ops_per_sec": len(trace) * 1000 / elapsed_ms,
        "host_process_cpu_elapsed_ms": host_cpu_elapsed_ms,
        "host_process_rss_start_bytes": host_rss_start,
        "host_process_rss_end_bytes": host_rss_end,
        "host_process_rss_delta_bytes": host_rss_end - host_rss_start,
        "sqlite_database_footprint_bytes": _sqlite_footprint_bytes(path),
        "per_operation": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
            for name, values in timings.items()
        },
    }
    if mode == "rust_native":
        result["native_cpu_elapsed_ms"] = native_result["cpu_elapsed_ms"]
        result["native_peak_rss_bytes"] = native_result["peak_rss_bytes"]
    return result


def _run_one_sync(
    path: Path,
    trace: list[WorkloadOperation],
    mode: str,
    *,
    trace_arguments: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Run synchronous registry modes without an asyncio dependency."""

    if mode not in {"in_process", "rust_authority"}:
        raise ValueError(f"sync MCP benchmark does not support mode {mode!r}")
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to collect MCP benchmark RSS metrics") from exc
    if trace_arguments is None:
        trace_arguments = [_trace_tool_arguments(step) for step in trace]
    elif len(trace_arguments) != len(trace):
        raise ValueError("trace arguments must have one entry per workload operation")
    process = psutil.Process()
    rss_start = int(process.memory_info().rss)
    cpu_started = time.process_time_ns()
    started = time.perf_counter_ns()
    registry, store = _make_registry(
        path,
        authority="rust" if mode == "rust_authority" else "python",
    )
    startup_ms = (time.perf_counter_ns() - started) / 1_000_000
    wall_started = time.perf_counter_ns()
    timings: dict[str, list[float]] = {
        "write_record": [],
        "read_event": [],
        "read_projection": [],
    }
    with _reuse_rust_store_session(store):
        for step, arguments in zip(trace, trace_arguments, strict=True):
            operation = step["operation"]
            operation_started = time.perf_counter_ns()
            payload = _call_registry_sync(registry, operation, arguments)
            if payload.get("entity_id") != step["expected_entity_id"]:
                raise RuntimeError("in-process MCP returned an unexpected entity id")
            if operation == "read_projection" and payload.get("index") != step["index"]:
                raise RuntimeError("in-process MCP returned an unexpected projection index")
            timings[operation].append(
                (time.perf_counter_ns() - operation_started) / 1_000_000
            )
    elapsed_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
    state = _final_state(path)
    return {
        "state_digest": _digest(state),
        "elapsed_ms": elapsed_ms,
        "startup_ms": startup_ms,
        "operations": len(trace),
        "throughput_ops_per_sec": len(trace) * 1000 / elapsed_ms,
        "host_process_cpu_elapsed_ms": (time.process_time_ns() - cpu_started) / 1_000_000,
        "host_process_rss_start_bytes": rss_start,
        "host_process_rss_end_bytes": int(process.memory_info().rss),
        "host_process_rss_delta_bytes": int(process.memory_info().rss) - rss_start,
        "sqlite_database_footprint_bytes": _sqlite_footprint_bytes(path),
        "per_operation": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
            for name, values in timings.items()
        },
    }


def _run_native_server(
    path: Path,
    trace: list[WorkloadOperation],
    trace_arguments: list[dict[str, object]],
) -> dict[str, Any]:
    binary = _native_binary_path()
    if not binary.is_file():
        raise FileNotFoundError(
            f"Rust benchmark server is not built: {binary}; run cargo build --release --manifest-path rust/Cargo.toml -p kogwistar-api --example mixed_mcp_authority_benchmark"
        )
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to measure native child CPU and RSS") from exc

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    database = path.resolve()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [str(binary), str(database), f"127.0.0.1:{port}"],
        env={
            **os.environ,
            "KOGWISTAR_MIXED_ROOT_PAYLOAD": json.dumps(
                build_root_graph_payload(), sort_keys=True, separators=(",", ":")
            ),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=path.resolve().parent,
    )
    child = psutil.Process(process.pid)
    startup_started = time.perf_counter_ns()
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Rust MCP benchmark server exited with {process.returncode}")
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=0.5) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.01)
        else:
            raise TimeoutError("Rust MCP benchmark server failed to become ready")
        startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000

        cpu_before = child.cpu_times()
        peak_rss = child.memory_info().rss
        stop_sampling = threading.Event()

        def sample_rss() -> None:
            nonlocal peak_rss
            while not stop_sampling.wait(0.01):
                try:
                    peak_rss = max(peak_rss, child.memory_info().rss)
                except psutil.Error:
                    return

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        timings: dict[str, list[float]] = {
            "write_record": [], "read_event": [], "read_projection": []
        }
        wall_started = time.perf_counter_ns()
        for step, arguments in zip(trace, trace_arguments, strict=True):
            operation = step["operation"]
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": step["index"] + 1,
                "method": "tools/call",
                "params": {
                    "name": operation,
                    "arguments": arguments,
                },
            }).encode("utf-8")
            call_started = time.perf_counter_ns()
            http_request = urllib.request.Request(
                f"{url}/mcp", data=request, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(
                    http_request, timeout=NATIVE_HTTP_TIMEOUT_SECONDS
                ) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(
                    "Rust MCP request failed: "
                    f"operation={operation!r} index={step['index']} "
                    f"status={exc.code} body={detail!r}"
                ) from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                raise TimeoutError(
                    "Rust MCP request did not complete: "
                    f"operation={operation!r} index={step['index']} "
                    f"timeout_seconds={NATIVE_HTTP_TIMEOUT_SECONDS} "
                    f"error={exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Rust MCP returned invalid JSON: "
                    f"operation={operation!r} index={step['index']}"
                ) from exc
            result = payload.get("result", {}).get("structuredContent", {})
            if payload.get("id") != step["index"] + 1 or result.get("entity_id") != step["expected_entity_id"]:
                raise RuntimeError("Rust MCP returned an unexpected result or request id")
            if operation == "read_projection" and result.get("index") != step["index"]:
                raise RuntimeError("Rust MCP returned an unexpected projection index")
            timings[operation].append((time.perf_counter_ns() - call_started) / 1_000_000)
        elapsed_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
        cpu_after = child.cpu_times()
        stop_sampling.set()
        sampler.join(timeout=2)
        cpu_ms = (cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system) * 1000
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    return {
        "startup_ms": startup_ms,
        "elapsed_ms": elapsed_ms,
        "cpu_elapsed_ms": cpu_ms,
        "peak_rss_bytes": peak_rss,
        "per_operation": timings,
    }


def _ensure_native_benchmark_binary() -> Path:
    global _NATIVE_BINARY_READY
    configured = os.environ.get("KOGWISTAR_MIXED_NATIVE_BINARY")
    if configured:
        binary = Path(configured)
        if not binary.is_file():
            raise FileNotFoundError(
                f"configured Rust MCP benchmark server is not a file: {binary}"
            )
        _NATIVE_BINARY_READY = True
        return binary
    binary_name = (
        "mixed_mcp_authority_benchmark.exe"
        if os.name == "nt"
        else "mixed_mcp_authority_benchmark"
    )
    repository = Path(__file__).resolve().parents[1]
    binary = repository / "rust" / "target" / "release" / "examples" / binary_name
    if not _NATIVE_BINARY_READY or not binary.is_file():
        subprocess.run(
            [
                "cargo", "build", "--release", "--manifest-path",
                str(repository / "rust" / "Cargo.toml"),
                "-p", "kogwistar-api", "--example", "mixed_mcp_authority_benchmark",
            ],
            cwd=repository / "rust",
            check=True,
        )
        _NATIVE_BINARY_READY = True
    if not binary.is_file():
        raise FileNotFoundError(f"Cargo succeeded but native benchmark binary is absent: {binary}")
    return binary


def _native_binary_path() -> Path:
    configured = os.environ.get("KOGWISTAR_MIXED_NATIVE_BINARY")
    if configured:
        binary = Path(configured)
        if not binary.is_file():
            raise FileNotFoundError(
                f"configured Rust MCP benchmark server is not a file: {binary}"
            )
        return binary
    return _ensure_native_benchmark_binary()


async def _run_native_concurrent_one(
    path: Path, phases: list[ConcurrentPhase]
) -> dict[str, Any]:
    """Run the deterministic concurrent trace through the Rust HTTP MCP server."""
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required to measure native child CPU and RSS") from exc

    graph_payloads = {
        step["write_index"]: build_graph_payloads(
            step["write_index"], vector_sha256="0" * 64
        )
        for phase in phases
        for step in phase["steps"]
    }
    binary_name = (
        "mixed_mcp_authority_benchmark.exe"
        if os.name == "nt"
        else "mixed_mcp_authority_benchmark"
    )
    binary = (
        Path(__file__).resolve().parents[1]
        / "rust"
        / "target"
        / "release"
        / "examples"
        / binary_name
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    database = path.resolve()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [str(binary), str(database), f"127.0.0.1:{port}", "concurrent"],
        env={
            **os.environ,
            "KOGWISTAR_MIXED_ROOT_PAYLOAD": json.dumps(
                build_root_graph_payload(), sort_keys=True, separators=(",", ":")
            ),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=database.parent,
    )
    child = psutil.Process(process.pid)
    startup_started = time.perf_counter_ns()
    deadline = time.monotonic() + 30
    sampler: threading.Thread | None = None
    stop_sampling = threading.Event()
    peak_rss = 0
    cpu_before = None
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
                raise RuntimeError(
                    f"Rust concurrent MCP benchmark server exited with {process.returncode}: {stderr}"
                )
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=0.5) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.01)
        else:
            raise TimeoutError("Rust concurrent MCP benchmark server failed to become ready")
        startup_ms = (time.perf_counter_ns() - startup_started) / 1_000_000
        cpu_before = child.cpu_times()
        peak_rss = child.memory_info().rss

        def sample_rss() -> None:
            nonlocal peak_rss
            while not stop_sampling.wait(0.01):
                try:
                    peak_rss = max(peak_rss, child.memory_info().rss)
                except psutil.Error:
                    return

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        timings: dict[str, list[float]] = {"write_record": [], "read_head": []}
        timing_lock = threading.Lock()

        def call(name: str, arguments: dict[str, object], request_id: int) -> dict[str, Any]:
            request_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{url}/mcp",
                data=request_body,
                headers={"Content-Type": "application/json"},
            )
            started = time.perf_counter_ns()
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            if payload.get("id") != request_id:
                raise RuntimeError("Rust concurrent MCP response id did not match request")
            if "error" in payload:
                raise RuntimeError(f"Rust concurrent MCP request failed: {payload['error']}")
            result = payload.get("result", {}).get("structuredContent")
            if not isinstance(result, dict):
                raise RuntimeError("Rust concurrent MCP returned no structured result")
            with timing_lock:
                timings[name].append(elapsed)
            return result

        async def actor(reader_id: int | None) -> int:
            operations = 0
            for phase in phases:
                for step in phase["steps"]:
                    index = step["write_index"]
                    before_readers = sum(
                        item["visibility"] == "before_commit" for item in step["reads"]
                    )
                    after_readers = sum(
                        item["visibility"] == "after_commit" for item in step["reads"]
                    )
                    if reader_id is None:
                        entity_id = graph_node_id(index)
                        edge_id = graph_edge_id(index)
                        source_id = previous_graph_node_id(index)
                        node_payload, edge_payload = graph_payloads[index]
                        result = await asyncio.to_thread(
                            call,
                            "write_record",
                            {
                                "index": index,
                                "expected_entity_id": entity_id,
                                "expected_edge_id": edge_id,
                                "expected_source_id": source_id,
                                "node_payload": node_payload,
                                "edge_payload": edge_payload,
                                "before_readers": before_readers,
                                "after_readers": after_readers,
                            },
                            (index + 1) * 100,
                        )
                        if result.get("entity_id") != entity_id:
                            raise RuntimeError("Rust concurrent MCP write returned wrong entity")
                        operations += 1
                    else:
                        reads = [
                            item
                            for item in step["reads"]
                            if item["reader_id"] == reader_id
                        ]
                        if not reads:
                            raise RuntimeError("concurrent trace omitted a reader operation")
                        for read_number, read in enumerate(reads):
                            request_id = (index + 1) * 100 + (reader_id + 1) * 10 + read_number
                            result = await asyncio.to_thread(
                                call,
                                "read_head",
                                {
                                    "index": index,
                                    "visibility": read["visibility"],
                                    "expected_revision": read["minimum_revision"],
                                    "before_readers": before_readers,
                                    "after_readers": after_readers,
                                },
                                request_id,
                            )
                            if result.get("revision") != read["minimum_revision"]:
                                raise RuntimeError("Rust concurrent MCP visibility mismatch")
                            operations += 1
            return operations

        wall_started = time.perf_counter_ns()
        outcomes = await asyncio.gather(
            actor(None), *(actor(reader_id) for reader_id in range(4)),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise BaseExceptionGroup("Rust concurrent MCP actors failed", failures)
        operation_count = sum(outcome for outcome in outcomes if isinstance(outcome, int))
        elapsed_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
        cpu_after = child.cpu_times()
        cpu_ms = (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        ) * 1000
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join(timeout=2)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    return {
        "authority": "rust_native",
        "transport": "http",
        "startup_ms": startup_ms,
        "elapsed_ms": elapsed_ms,
        "operations": operation_count,
        "throughput_ops_per_sec": operation_count * 1000 / elapsed_ms,
        "native_cpu_elapsed_ms": cpu_ms,
        "native_peak_rss_bytes": peak_rss,
        "per_operation": {
            name: {
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
            for name, values in timings.items()
        },
    }


def _run_concurrent_mcp_profiles(
    output: Path, *, items: int, samples: int
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    run_root = output / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Any] = {}
    all_digests: set[str] = set()
    progress_path = run_root / "progress.v3.json"
    progress: dict[str, Any] = {
        "schema": "kogwistar-mixed-mcp-concurrent-progress/v3",
        "status": "running",
        "items": items,
        "samples_per_mode": samples,
        "completed_mixes": {},
        "partial_samples": {},
        "active": None,
    }
    _write_json_atomic(progress_path, progress)
    _ensure_native_benchmark_binary()
    for mix in ("balanced", "read_heavy"):
        trace_path = run_root / f"concurrent_{mix}.trace.v3.json"
        phases = write_concurrent_trace(trace_path, items, mix=mix)
        mode_reports: dict[str, Any] = {}
        for authority in ("python", "rust"):
            for transport in ("in_process", "http"):
                mode = f"{authority}_{transport}"
                warmup_path = run_root / f"concurrent_{mix}-{mode}-warmup.sqlite3"
                progress["active"] = {
                    "mix": mix,
                    "mode": mode,
                    "phase": "warmup",
                }
                _write_json_atomic(progress_path, progress)
                print(
                    f"[mixed-mcp] concurrent warmup: mix={mix} mode={mode} items={items}",
                    file=sys.stderr,
                    flush=True,
                )
                warmup = _run_concurrent_one(
                    warmup_path,
                    phases,
                    authority=authority,
                    transport=transport,
                )
                runs = []
                sample_key = f"{mix}/{mode}"
                progress["partial_samples"][sample_key] = {"warmup": warmup, "samples": runs}
                for sample in range(samples):
                    db_path = run_root / f"concurrent_{mix}-{mode}-{sample}.sqlite3"
                    progress["active"] = {
                        "mix": mix,
                        "mode": mode,
                        "phase": "sample",
                        "sample": sample + 1,
                        "samples": samples,
                    }
                    _write_json_atomic(progress_path, progress)
                    print(
                        f"[mixed-mcp] concurrent sample {sample + 1}/{samples}: "
                        f"mix={mix} mode={mode} items={items}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = _run_concurrent_one(
                        db_path,
                        phases,
                        authority=authority,
                        transport=transport,
                    )
                    runs.append(result)
                    _write_json_atomic(progress_path, progress)
                    all_digests.add(result["state_digest"])
                if warmup["state_digest"] != runs[0]["state_digest"]:
                    raise RuntimeError(
                        f"concurrent MCP warm-up mismatch for {mix}/{mode}"
                    )
                mode_reports[mode] = {
                    "warmup": warmup,
                    "state_digest": runs[0]["state_digest"],
                    "median_elapsed_ms": statistics.median(
                        run["elapsed_ms"] for run in runs
                    ),
                    "median_startup_ms": statistics.median(
                        run["startup_ms"] for run in runs
                    ),
                    "median_throughput_ops_per_sec": statistics.median(
                        run["throughput_ops_per_sec"] for run in runs
                    ),
                    "elapsed_sample_variation": _sample_variation(
                        [run["elapsed_ms"] for run in runs]
                    ),
                    "throughput_sample_variation": _sample_variation(
                        [run["throughput_ops_per_sec"] for run in runs]
                    ),
                    "median_host_process_cpu_elapsed_ms": statistics.median(
                        run["host_process_cpu_elapsed_ms"] for run in runs
                    ),
                    "median_host_process_rss_delta_bytes": int(
                        statistics.median(
                            run["host_process_rss_delta_bytes"] for run in runs
                        )
                    ),
                    "samples": runs,
                }
        native_mode = "rust_native_http"
        native_warmup_path = run_root / f"concurrent_{mix}-{native_mode}-warmup.sqlite3"
        progress["active"] = {"mix": mix, "mode": native_mode, "phase": "warmup"}
        _write_json_atomic(progress_path, progress)
        print(
            f"[mixed-mcp] concurrent warmup: mix={mix} mode={native_mode} items={items}",
            file=sys.stderr,
            flush=True,
        )
        native_warmup = _run_async(
            _run_native_concurrent_one(native_warmup_path, phases)
        )
        native_warmup["sqlite_database_footprint_bytes"] = _sqlite_footprint_bytes(
            native_warmup_path
        )
        native_runs = []
        native_sample_key = f"{mix}/{native_mode}"
        progress["partial_samples"][native_sample_key] = {
            "warmup": native_warmup,
            "samples": native_runs,
        }
        for sample in range(samples):
            db_path = run_root / f"concurrent_{mix}-{native_mode}-{sample}.sqlite3"
            progress["active"] = {
                "mix": mix,
                "mode": native_mode,
                "phase": "sample",
                "sample": sample + 1,
                "samples": samples,
            }
            _write_json_atomic(progress_path, progress)
            print(
                f"[mixed-mcp] concurrent sample {sample + 1}/{samples}: "
                f"mix={mix} mode={native_mode} items={items}",
                file=sys.stderr,
                flush=True,
            )
            result = _run_async(_run_native_concurrent_one(db_path, phases))
            state = _final_state(db_path)
            expected_writes = sum(len(phase["steps"]) for phase in phases)
            expected_reads = sum(
                len(step["reads"]) for phase in phases for step in phase["steps"]
            )
            if (
                len(state["events"]) != 1 + 2 * expected_writes
                or result["operations"] != expected_writes + expected_reads
                or result["per_operation"]["write_record"]["count"] != expected_writes
                or result["per_operation"]["read_head"]["count"] != expected_reads
            ):
                raise RuntimeError(
                    "Rust-native concurrent trace is incomplete: "
                    f"events={len(state['events'])}/{1 + 2 * expected_writes}, "
                    f"operations={result['operations']}/{expected_writes + expected_reads}, "
                    f"writes={result['per_operation']['write_record']['count']}/{expected_writes}, "
                    f"reads={result['per_operation']['read_head']['count']}/{expected_reads}"
                )
            result["state_digest"] = _digest(state)
            result["sqlite_database_footprint_bytes"] = _sqlite_footprint_bytes(db_path)
            native_runs.append(result)
            _write_json_atomic(progress_path, progress)
            all_digests.add(result["state_digest"])
        native_state = _final_state(native_warmup_path)
        native_warmup["state_digest"] = _digest(native_state)
        if native_warmup["state_digest"] != native_runs[0]["state_digest"]:
            raise RuntimeError(f"native concurrent warm-up mismatch for {mix}")
        mode_reports[native_mode] = {
            "warmup": native_warmup,
            "state_digest": native_runs[0]["state_digest"],
            "median_elapsed_ms": statistics.median(
                run["elapsed_ms"] for run in native_runs
            ),
            "median_startup_ms": statistics.median(
                run["startup_ms"] for run in native_runs
            ),
            "median_throughput_ops_per_sec": statistics.median(
                run["throughput_ops_per_sec"] for run in native_runs
            ),
            "elapsed_sample_variation": _sample_variation(
                [run["elapsed_ms"] for run in native_runs]
            ),
            "throughput_sample_variation": _sample_variation(
                [run["throughput_ops_per_sec"] for run in native_runs]
            ),
            "median_native_cpu_elapsed_ms": statistics.median(
                run["native_cpu_elapsed_ms"] for run in native_runs
            ),
            "median_native_peak_rss_bytes": int(
                statistics.median(run["native_peak_rss_bytes"] for run in native_runs)
            ),
            "samples": native_runs,
        }
        if len({row["state_digest"] for row in mode_reports.values()}) != 1:
            raise RuntimeError(f"concurrent MCP transport/authority mismatch for {mix}")
        reports[mix] = {
            "trace_path": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "state_digest_equal": True,
            "modes": mode_reports,
        }
        progress["completed_mixes"][mix] = reports[mix]
        progress["active"] = None
        _write_json_atomic(progress_path, progress)
    if len(all_digests) != 1:
        raise RuntimeError("concurrent MCP modes disagree on one or more trace states")
    progress["status"] = "complete"
    _write_json_atomic(progress_path, progress)
    return {
        "schema": "kogwistar-mixed-mcp-concurrent-report/v3",
        "run_id": run_id,
        "items": items,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "workers": {"writers": 1, "readers": 4},
        "mixes": reports,
    }


def run_benchmark(
    output: Path,
    *,
    items: int,
    samples: int,
    concurrent_items: int = 8,
    modes: tuple[str, ...] = DEFAULT_MODES,
    include_concurrent: bool = True,
) -> dict[str, Any]:
    supported_modes = {"in_process", "rust_authority", "http", "rust_native"}
    if not modes or any(mode not in supported_modes for mode in modes):
        raise ValueError(f"unsupported MCP benchmark mode selection: {modes!r}")
    output.mkdir(parents=True, exist_ok=True)
    if "rust_native" in modes:
        _ensure_native_benchmark_binary()
    run_id = uuid.uuid4().hex[:12]
    run_root = output / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    trace = write_trace(run_root / "mcp_trace.v1.json", items)
    results: dict[str, Any] = {}
    digests: set[str] = set()
    progress_path = run_root / "progress.v3.json"
    progress: dict[str, Any] = {
        "schema": "kogwistar-mixed-mcp-progress/v3",
        "status": "running",
        "run_id": run_id,
        "items": items,
        "samples_per_mode": samples,
        "completed_modes": [],
        "modes": results,
        "partial_samples": {},
        "active": None,
    }
    _write_json_atomic(progress_path, progress)
    progress["active"] = {"phase": "prepare_trace_arguments", "operations": len(trace)}
    _write_json_atomic(progress_path, progress)
    print(
        f"[mixed-mcp] prepare tool arguments once: operations={len(trace)}",
        file=sys.stderr,
        flush=True,
    )
    argument_preparation_started = time.perf_counter_ns()
    trace_arguments = [_trace_tool_arguments(step) for step in trace]
    argument_preparation_ms = (time.perf_counter_ns() - argument_preparation_started) / 1_000_000
    progress["tool_argument_preparation_ms"] = argument_preparation_ms
    progress["active"] = None
    _write_json_atomic(progress_path, progress)
    for mode in modes:
        warmup_path = run_root / f"{mode}-warmup.sqlite3"
        progress["active"] = {"mode": mode, "phase": "warmup"}
        _write_json_atomic(progress_path, progress)
        print(
            f"[mixed-mcp] warmup: mode={mode} items={items}",
            file=sys.stderr,
            flush=True,
        )
        warmup = (
            _run_one_sync(warmup_path, trace, mode, trace_arguments=trace_arguments)
            if mode in {"in_process", "rust_authority"}
            else _run_async(_run_one(warmup_path, trace, mode, trace_arguments=trace_arguments))
        )
        mode_samples = []
        progress["partial_samples"][mode] = {"warmup": warmup, "samples": mode_samples}
        for sample in range(samples):
            path = run_root / f"{mode}-{sample}.sqlite3"
            progress["active"] = {
                "mode": mode,
                "phase": "sample",
                "sample": sample + 1,
                "samples": samples,
            }
            _write_json_atomic(progress_path, progress)
            print(
                f"[mixed-mcp] sample {sample + 1}/{samples}: mode={mode} items={items}",
                file=sys.stderr,
                flush=True,
            )
            result = (
                _run_one_sync(path, trace, mode, trace_arguments=trace_arguments)
                if mode in {"in_process", "rust_authority"}
                else _run_async(_run_one(path, trace, mode, trace_arguments=trace_arguments))
            )
            digests.add(result["state_digest"])
            mode_samples.append(result)
            _write_json_atomic(progress_path, progress)
        results[mode] = {
            "warmup": warmup,
            "state_digest": mode_samples[0]["state_digest"],
            "median_elapsed_ms": statistics.median(row["elapsed_ms"] for row in mode_samples),
            "elapsed_sample_variation": _sample_variation(
                [row["elapsed_ms"] for row in mode_samples]
            ),
            "median_startup_ms": statistics.median(row["startup_ms"] for row in mode_samples),
            "median_throughput_ops_per_sec": statistics.median(
                row["throughput_ops_per_sec"] for row in mode_samples
            ),
            "median_native_cpu_elapsed_ms": (
                statistics.median(row["native_cpu_elapsed_ms"] for row in mode_samples)
                if mode == "rust_native" else None
            ),
            "median_native_peak_rss_bytes": (
                int(statistics.median(row["native_peak_rss_bytes"] for row in mode_samples))
                if mode == "rust_native" else None
            ),
            "median_host_process_cpu_elapsed_ms": statistics.median(
                row["host_process_cpu_elapsed_ms"] for row in mode_samples
            ),
            "median_host_process_rss_delta_bytes": int(statistics.median(
                row["host_process_rss_delta_bytes"] for row in mode_samples
            )),
            "throughput_sample_variation": _sample_variation(
                [row["throughput_ops_per_sec"] for row in mode_samples]
            ),
            "samples": mode_samples,
        }
        progress["completed_modes"] = list(results)
        progress["active"] = None
        _write_json_atomic(progress_path, progress)
    if len(digests) != 1 or any(
        mode_result["warmup"]["state_digest"] != next(iter(digests))
        for mode_result in results.values()
    ):
        raise RuntimeError("MCP registry and HTTP transports persisted different states")
    concurrent_report = None
    if include_concurrent:
        progress["active"] = {"phase": "concurrent_profiles", "items": concurrent_items}
        _write_json_atomic(progress_path, progress)
        concurrent_report = _run_concurrent_mcp_profiles(
            run_root / "concurrent", items=concurrent_items, samples=samples
        )
    report = {
        "schema": "kogwistar-mixed-mcp-report/v3",
        "run_id": run_id,
        "items": items,
        "samples_per_mode": samples,
        "warmup_samples_per_mode": 1,
        "tool_argument_preparation_ms": argument_preparation_ms,
        "state_digest_equal": True,
        "modes": results,
        "concurrent": concurrent_report,
    }
    progress["status"] = "complete"
    progress["active"] = None
    progress["concurrent"] = concurrent_report
    _write_json_atomic(progress_path, progress)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrent-items", type=int, default=8)
    parser.add_argument(
        "--mode",
        action="append",
        choices=DEFAULT_MODES,
        help="run only the selected sequential mode; repeat for a comparison subset",
    )
    args = parser.parse_args()
    if args.items < 1 or args.concurrent_items < 1 or args.samples < 2:
        parser.error("item counts must be positive and --samples must be at least 2")
    report = run_benchmark(
        args.output,
        items=args.items,
        samples=args.samples,
        concurrent_items=args.concurrent_items,
        modes=tuple(args.mode) if args.mode else DEFAULT_MODES,
    )
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _write_json_atomic(args.output / "report.v3.json", report)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
