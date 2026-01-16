"""
workflow/executor.py

LangGraph-like execution semantics over a static workflow graph stored in your engine.

Core guarantees:
- Topology is static: workflow nodes/edges are pre-registered and never mutated during a run.
- Routing is dynamic: predicates evaluate (state, last_result) to choose next steps.
- Parallelism: multiple nodes can execute concurrently; a node may fan out to multiple next nodes.
- Shared mutable message queue: steps can publish events; other steps can observe during retries.
- Single-writer: worker threads compute results; scheduler thread applies results and routes tokens.
- Caching: per-node step caching via joblib, isolated by run_id.

This module does NOT require LangGraph.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from graph_knowledge_engine.engine import GraphKnowledgeEngine

from joblib import Memory

from .contract import (
    WorkflowSpec,
    WorkflowNodeInfo,
    WorkflowEdgeInfo,
    load_workflow_graph,
    validate_workflow,
    Predicate,
    State,
    Result,
)

Json = Any
StepFn = Callable[["StepContext"], Result]


@dataclass(frozen=True)
class WorkflowEvent:
    run_id: str
    seq: int
    type: str
    ts_ms: int
    payload: Dict[str, Json]

    def to_dict(self) -> Dict[str, Json]:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "type": self.type,
            "ts_ms": self.ts_ms,
            "payload": self.payload,
        }


class StepContext:
    """Context passed to each step. Provides shared state + message queue helpers."""

    def __init__(
        self,
        *,
        run_id: str,
        node_id: str,
        op: str,
        state: State,
        message_queue: "queue.Queue[Dict[str, Json]]",
    ) -> None:
        self.run_id = run_id
        self.node_id = node_id
        self.op = op
        self.state = state
        self._mq = message_queue

    def publish(self, message: Dict[str, Json]) -> None:
        """Publish a message for other steps to observe."""
        self._mq.put(message)

    def drain_messages(self, *, max_items: int = 100) -> List[Dict[str, Json]]:
        """Non-blocking drain."""
        out: List[Dict[str, Json]] = []
        for _ in range(max_items):
            try:
                out.append(self._mq.get_nowait())
            except queue.Empty:
                break
        return out


class WorkflowExecutor:
    """
    Executes a workflow spec loaded from your existing engine.

    step_resolver(op_name) -> StepFn
      - This is the key requirement: step functions are delegated via resolver,
        not hard-coded inside the executor.
    """

    def __init__(
        self,
        *,
        engine: GraphKnowledgeEngine,
        workflow: WorkflowSpec,
        step_resolver: Callable[[str], StepFn],
        predicate_registry: Optional[Dict[str, Predicate]] = None,
        cache_root: str | Path = ".workflow_cache",
        max_workers: int = 4,
    ) -> None:
        self.engine = engine
        self.workflow = workflow
        self.step_resolver = step_resolver
        self.predicate_registry = predicate_registry or {}
        self.max_workers = max_workers

        self._nodes, self._adj = load_workflow_graph(engine=engine, spec=workflow)
        validate_workflow(engine=engine, spec=workflow, predicate_registry=self.predicate_registry)

        self.state: State = {}
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._mq: queue.Queue[Dict[str, Json]] = queue.Queue()
        self._cache_root = Path(cache_root)

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _emit(self, run_id: str, typ: str, payload: Dict[str, Json]) -> WorkflowEvent:
        return WorkflowEvent(
            run_id=run_id,
            seq=self._next_seq(),
            type=typ,
            ts_ms=int(time.time() * 1000),
            payload=payload,
        )

    def _memory_for_run(self, run_id: str) -> Memory:
        d = self._cache_root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return Memory(location=str(d), verbose=0)

    def _cache_key(self, *, node: WorkflowNodeInfo, step_input: Dict[str, Json]) -> str:
        obj = {
            "workflow_id": self.workflow.workflow_id,
            "node_id": node.node_id,
            "op": node.op,
            "version": node.version,
            "input": step_input,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _apply_result(self, *, node: WorkflowNodeInfo, result: Result) -> None:
        """
        Default applier:
        - stores raw result under result.<op>
        - also places canonical keys for common ops used by orchestrator integration
        """
        self.state[f"result.{node.op}"] = result
        if node.op == "memory_retrieve":
            self.state["memory"] = result
        elif node.op == "kg_retrieve":
            self.state["kg"] = result
        elif node.op == "answer_only":
            self.state["answer"] = result

    def _choose_next(self, *, node: WorkflowNodeInfo, last_result: Result) -> List[str]:
        edges = self._adj.get(node.node_id, [])
        matched: List[str] = []

        for e in edges:
            if e.predicate is None:
                continue
            pred = self.predicate_registry.get(e.predicate)
            if pred is None:
                continue
            ok = False
            try:
                ok = bool(pred(self.state, last_result))
            except Exception:
                ok = False
            if ok:
                matched.append(e.dst)
                # no fanout => take first unless edge explicitly says many
                if not node.fanout and e.multiplicity != "many":
                    return matched

        # fanout => take all matches
        if matched and (node.fanout or any(ed.multiplicity == "many" for ed in edges)):
            return matched

        # default edge if none matched
        if not matched:
            for e in edges:
                if e.is_default:
                    return [e.dst]
        return matched

    def run(self, *, run_id: str, initial_state: State) -> Iterable[Dict[str, Json]]:
        """
        End-user interface: for ev in executor.run(...)

        The yielded events are suitable to be mirrored into your conversation graph as tool events.
        """
        self.state = dict(initial_state)
        mem = self._memory_for_run(run_id)

        token_q: queue.Queue[str] = queue.Queue()
        token_q.put(self.workflow.start_node_id)

        done_q: queue.Queue[Tuple[str, Result, bool]] = queue.Queue()

        yield self._emit(run_id, "workflow_start", {"workflow_id": self.workflow.workflow_id, "start": self.workflow.start_node_id}).to_dict()

        def worker(node_id: str) -> None:
            node = self._nodes[node_id]
            step_fn = self.step_resolver(node.op)

            # Minimal deterministic cache input; you can extend this per-op if you want stronger caching
            step_input = {"state_keys": sorted(list(self.state.keys()))}
            key = self._cache_key(node=node, step_input=step_input)

            cached_flag = False
            if node.cacheable:
                @mem.cache
                def _cached_call(k: str) -> Result:
                    _ = k
                    ctx = StepContext(run_id=run_id, node_id=node.node_id, op=node.op, state=self.state, message_queue=self._mq)
                    return step_fn(ctx)

                # best-effort cache hit detection
                try:
                    call_id = _cached_call._get_args_id(k=key)  # type: ignore[attr-defined]
                    cached_flag = mem.store_backend.contains_item(_cached_call.func_id, call_id)  # type: ignore[attr-defined]
                except Exception:
                    cached_flag = False

                result = _cached_call(k=key)
            else:
                ctx = StepContext(run_id=run_id, node_id=node.node_id, op=node.op, state=self.state, message_queue=self._mq)
                result = step_fn(ctx)

            done_q.put((node_id, result, cached_flag))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            inflight: Dict[str, Any] = {}

            while True:
                # schedule while capacity
                while len(inflight) < self.max_workers:
                    try:
                        nid = token_q.get_nowait()
                    except queue.Empty:
                        break
                    inflight[nid] = pool.submit(worker, nid)
                    yield self._emit(run_id, "step_scheduled", {"node_id": nid, "op": self._nodes[nid].op}).to_dict()

                if not inflight and token_q.empty():
                    yield self._emit(run_id, "workflow_done", {"state": self.state}).to_dict()
                    return

                node_id, result, cached_flag = done_q.get()
                inflight.pop(node_id, None)

                node = self._nodes[node_id]
                yield self._emit(run_id, "step_completed", {"node_id": node_id, "op": node.op, "cached": cached_flag}).to_dict()

                # single-writer apply
                self._apply_result(node=node, result=result)

                # terminal check
                if node.terminal or len(self._adj.get(node_id, [])) == 0:
                    continue

                next_nodes = self._choose_next(node=node, last_result=result)
                for nxt in next_nodes:
                    token_q.put(nxt)
                    yield self._emit(run_id, "token_spawned", {"from": node_id, "to": nxt}).to_dict()
