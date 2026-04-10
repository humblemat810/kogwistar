from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Edge, Grounding, MentionVerification, Node, Span
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.runtime import WorkflowRuntime


class _ProfileEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "profile-fake-3d"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            base = float((len(str(text)) % 7) + 1)
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors


def _mk_span(doc_id: str) -> Span:
    return Span(
        collection_page_url=f"document_collection/{doc_id}",
        document_page_url=f"document/{doc_id}",
        doc_id=doc_id,
        insertion_method="profile_seed",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="system", is_verified=True, score=1.0, notes=""),
    )


def _mk_node(node_id: str, *, doc_id: str) -> Node:
    return Node(
        id=node_id,
        label=f"Node {node_id}",
        type="entity",
        summary=f"Summary {node_id}",
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        embedding=[0.1, 0.2, 0.3],
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _mk_edge(edge_id: str, *, src: str, tgt: str, doc_id: str) -> Edge:
    return Edge(
        id=edge_id,
        label=f"Edge {edge_id}",
        type="relationship",
        summary=f"Summary {edge_id}",
        relation="related_to",
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=None,
        target_edge_ids=None,
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        metadata={"level_from_root": 0, "entity_type": "kg_relation"},
        embedding=[0.1, 0.2, 0.3],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


@dataclass
class _TimingStat:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += float(elapsed_s)
        self.max_s = max(self.max_s, float(elapsed_s))

    def to_dict(self) -> dict[str, float | int]:
        avg_s = (self.total_s / self.count) if self.count else 0.0
        return {
            "count": self.count,
            "total_ms": round(self.total_s * 1000.0, 3),
            "avg_ms": round(avg_s * 1000.0, 3),
            "max_ms": round(self.max_s * 1000.0, 3),
        }


class TimingRecorder:
    def __init__(self) -> None:
        self._stats: dict[str, _TimingStat] = defaultdict(_TimingStat)

    def add(self, label: str, elapsed_s: float) -> None:
        self._stats[str(label)].add(float(elapsed_s))

    @contextmanager
    def wrap_method(self, obj: Any, attr: str, *, label: str | None = None):
        if obj is None or not hasattr(obj, attr):
            yield False
            return
        original = getattr(obj, attr)
        if not callable(original):
            yield False
            return

        recorder = self
        timing_label = str(label or attr)

        def _wrapped(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                recorder.add(timing_label, time.perf_counter() - started)

        setattr(obj, attr, _wrapped)
        try:
            yield True
        finally:
            setattr(obj, attr, original)

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        items = sorted(
            self._stats.items(),
            key=lambda item: item[1].total_s,
            reverse=True,
        )
        return {name: stat.to_dict() for name, stat in items}


class SysMonitoringWallProfiler:
    def __init__(
        self,
        *,
        include_files: tuple[str, ...] = (),
        include_names: tuple[str, ...] = (),
    ) -> None:
        self.include_files = tuple(x.replace("\\", "/") for x in include_files)
        self.include_names = tuple(include_names)
        self._stats: dict[str, _TimingStat] = defaultdict(_TimingStat)
        self._tls = threading.local()
        self._enabled = False

    def _tracked(self, code: Any) -> bool:
        filename = str(getattr(code, "co_filename", "") or "").replace("\\", "/")
        qualname = str(getattr(code, "co_qualname", "") or getattr(code, "co_name", ""))
        if self.include_files and not any(part in filename for part in self.include_files):
            return False
        if self.include_names and not any(part in qualname for part in self.include_names):
            return False
        return True

    def _stack(self) -> list[tuple[str, float, bool]]:
        stack = getattr(self._tls, "stack", None)
        if stack is None:
            stack = []
            self._tls.stack = stack
        return stack

    def _on_start(self, code: Any, *args: Any) -> None:
        qualname = str(getattr(code, "co_qualname", "") or getattr(code, "co_name", ""))
        self._stack().append((qualname, time.perf_counter(), self._tracked(code)))

    def _on_stop(self, code: Any, *args: Any) -> None:
        stack = self._stack()
        if not stack:
            return
        qualname, started, tracked = stack.pop()
        if tracked:
            self._stats[qualname].add(time.perf_counter() - started)

    def start(self) -> bool:
        if not hasattr(sys, "monitoring"):
            return False
        if self._enabled:
            return True
        monitoring = sys.monitoring
        tool_id = monitoring.PROFILER_ID
        monitoring.use_tool_id(tool_id, "kogwistar.perf_profile")
        monitoring.register_callback(tool_id, monitoring.events.PY_START, self._on_start)
        monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, self._on_stop)
        monitoring.register_callback(tool_id, monitoring.events.PY_UNWIND, self._on_stop)
        monitoring.set_events(
            tool_id,
            monitoring.events.PY_START
            | monitoring.events.PY_RETURN
            | monitoring.events.PY_UNWIND,
        )
        self._enabled = True
        return True

    def stop(self) -> None:
        if not hasattr(sys, "monitoring") or not self._enabled:
            return
        monitoring = sys.monitoring
        tool_id = monitoring.PROFILER_ID
        monitoring.set_events(tool_id, 0)
        monitoring.register_callback(tool_id, monitoring.events.PY_START, None)
        monitoring.register_callback(tool_id, monitoring.events.PY_RETURN, None)
        monitoring.register_callback(tool_id, monitoring.events.PY_UNWIND, None)
        monitoring.free_tool_id(tool_id)
        self._enabled = False

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        items = sorted(
            self._stats.items(),
            key=lambda item: item[1].total_s,
            reverse=True,
        )
        return {name: stat.to_dict() for name, stat in items}


def _build_in_memory_engine(root: Path, *, graph_type: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(root),
        kg_graph_type=graph_type,
        embedding_function=_ProfileEmbeddingFunction(),
        backend_factory=build_in_memory_backend,
    )


def _profile_one_scenario(
    root: Path,
    *,
    iterations: int,
    fast_trace_persistence: bool,
    include_monitoring: bool,
    use_validation_cache: bool,
) -> dict[str, Any]:
    workflow_engine = _build_in_memory_engine(root / "wf", graph_type="workflow")
    conversation_engine = _build_in_memory_engine(root / "conv", graph_type="conversation")
    workflow_engine._phase1_enable_validation_cache = bool(use_validation_cache)  # type: ignore[attr-defined]
    conversation_engine._phase1_enable_validation_cache = bool(use_validation_cache)  # type: ignore[attr-defined]
    runtime = WorkflowRuntime(
        workflow_engine=workflow_engine,
        conversation_engine=conversation_engine,
        step_resolver=lambda _op: (lambda _ctx: RunSuccess(conversation_node_id=None, state_update=[])),
        predicate_registry={},
        checkpoint_every_n_steps=1,
        fast_trace_persistence=fast_trace_persistence,
    )
    recorder = TimingRecorder()
    monitor = SysMonitoringWallProfiler(
        include_files=(
            "kogwistar/engine_core/indexing.py",
            "kogwistar/engine_core/subsystems/write.py",
            "kogwistar/engine_core/engine_sqlite.py",
        ),
        include_names=(
            "reconcile_indexes",
            "apply_index_job",
            "fanout_endpoints_rows",
            "enqueue_index_job",
            "claim_index_jobs",
            "mark_index_job_done",
            "append_entity_event",
        ),
    )

    method_specs: list[tuple[Any, str, str]] = [
        (runtime, "_persist_workflow_run", "runtime.persist_workflow_run"),
        (runtime, "_persist_step_exec", "runtime.persist_step_exec"),
        (runtime, "_persist_checkpoint", "runtime.persist_checkpoint"),
        (conversation_engine.write, "add_node", "write.add_node"),
        (conversation_engine.write, "add_edge", "write.add_edge"),
        (conversation_engine, "_append_event_for_entity", "engine.append_entity_event"),
        (conversation_engine, "_emit_change", "engine.emit_change"),
        (conversation_engine, "enqueue_index_jobs_for_node", "index.enqueue_jobs_for_node"),
        (conversation_engine, "enqueue_index_jobs_for_edge", "index.enqueue_jobs_for_edge"),
        (conversation_engine, "reconcile_indexes", "index.reconcile_indexes"),
        (conversation_engine.indexing, "apply_index_job", "index.apply_index_job"),
        (conversation_engine.backend, "node_add", "backend.node_add"),
        (conversation_engine.backend, "edge_add", "backend.edge_add"),
        (conversation_engine.backend, "node_docs_add", "backend.node_docs_add"),
        (conversation_engine.backend, "node_docs_get", "backend.node_docs_get"),
        (conversation_engine.backend, "node_docs_upsert", "backend.node_docs_upsert"),
        (conversation_engine.backend, "node_refs_add", "backend.node_refs_add"),
        (conversation_engine.backend, "node_refs_get", "backend.node_refs_get"),
        (conversation_engine.backend, "node_refs_upsert", "backend.node_refs_upsert"),
        (conversation_engine.backend, "edge_refs_add", "backend.edge_refs_add"),
        (conversation_engine.backend, "edge_refs_get", "backend.edge_refs_get"),
        (conversation_engine.backend, "edge_refs_upsert", "backend.edge_refs_upsert"),
        (conversation_engine.backend, "edge_endpoints_add", "backend.edge_endpoints_add"),
        (conversation_engine.backend, "edge_endpoints_get", "backend.edge_endpoints_get"),
        (conversation_engine.backend, "edge_endpoints_upsert", "backend.edge_endpoints_upsert"),
        (conversation_engine.backend, "edge_endpoints_delete", "backend.edge_endpoints_delete"),
        (conversation_engine.meta_sqlite, "append_entity_event", "meta.append_entity_event"),
        (conversation_engine.meta_sqlite, "enqueue_index_job", "meta.enqueue_index_job"),
        (conversation_engine.meta_sqlite, "claim_index_jobs", "meta.claim_index_jobs"),
        (conversation_engine.meta_sqlite, "mark_index_job_done", "meta.mark_index_job_done"),
        (
            conversation_engine.meta_sqlite,
            "get_index_applied_fingerprint",
            "meta.get_index_applied_fingerprint",
        ),
        (
            conversation_engine.meta_sqlite,
            "set_index_applied_fingerprint",
            "meta.set_index_applied_fingerprint",
        ),
    ]

    scenario_started = time.perf_counter()
    with ExitStack() as stack:
        for obj, attr, label in method_specs:
            stack.enter_context(recorder.wrap_method(obj, attr, label=label))
        if include_monitoring:
            monitor.start()
            stack.callback(monitor.stop)

        run_id = f"perf-run-{uuid.uuid4().hex}"
        conversation_id = f"perf-conv-{uuid.uuid4().hex}"

        workflow_run = runtime._persist_workflow_run(
            conversation_id=conversation_id,
            workflow_id="perf-workflow",
            run_id=run_id,
            turn_node_id="turn-0",
            status="running",
        )

        last_exec_node = workflow_run
        for step_seq in range(1, int(iterations) + 1):
            step_exec = runtime._persist_step_exec(
                conversation_id=conversation_id,
                workflow_id="perf-workflow",
                run_id=run_id,
                step_seq=step_seq,
                workflow_node_id="perf-step",
                op="perf-op",
                status="succeeded",
                duration_ms=1,
                result=RunSuccess(conversation_node_id=None, state_update=[]),
                state={"conversation_id": conversation_id, "user_id": "perf-user"},  # type: ignore[arg-type]
                last_exec_node=last_exec_node,
            )
            runtime._persist_checkpoint(
                conversation_id=conversation_id,
                workflow_id="perf-workflow",
                run_id=run_id,
                step_seq=step_seq,
                state={"conversation_id": conversation_id, "user_id": "perf-user"},  # type: ignore[arg-type]
                last_exec_node=step_exec,
            )
            last_exec_node = step_exec

    total_ms = round((time.perf_counter() - scenario_started) * 1000.0, 3)
    return {
        "fast_trace_persistence": bool(fast_trace_persistence),
        "iterations": int(iterations),
        "scenario_total_ms": total_ms,
        "method_timings": recorder.to_dict(),
        "monitoring_timings": monitor.to_dict() if include_monitoring else {},
    }


def _seed_index_job_batch(
    engine: GraphKnowledgeEngine,
    *,
    namespace: str,
    batch_id: str,
) -> tuple[str, str]:
    node_id = f"{batch_id}-node"
    edge_id = f"{batch_id}-edge"
    doc_id = f"{namespace}-doc"
    node = _mk_node(node_id, doc_id=doc_id)
    edge = _mk_edge(edge_id, src=node_id, tgt=node_id, doc_id=doc_id)

    node_meta = {
        "doc_id": node.doc_id,
        "label": node.label,
        "type": node.type,
        "summary": node.summary,
        "entity_type": node.metadata.get("entity_type"),
        "level_from_root": node.metadata.get("level_from_root", 0),
    }
    edge_meta = {
        "doc_id": edge.doc_id,
        "label": edge.label,
        "type": edge.type,
        "summary": edge.summary,
        "relation": edge.relation,
        "entity_type": edge.metadata.get("entity_type"),
        "level_from_root": edge.metadata.get("level_from_root", 0),
    }

    engine.backend.node_add(
        ids=[node.id],
        documents=[node.model_dump_json(field_mode="backend")],
        metadatas=[node_meta],
        embeddings=[node.embedding or [0.1, 0.2, 0.3]],
    )
    engine.backend.edge_add(
        ids=[edge.id],
        documents=[edge.model_dump_json(field_mode="backend")],
        metadatas=[edge_meta],
        embeddings=[edge.embedding or [0.1, 0.2, 0.3]],
    )

    for entity_kind, entity_id, index_kind, op in (
        ("node", node.id, "node_docs", "UPSERT"),
        ("node", node.id, "node_refs", "UPSERT"),
        ("edge", edge.id, "edge_refs", "UPSERT"),
        ("edge", edge.id, "edge_endpoints", "UPSERT"),
    ):
        engine.meta_sqlite.enqueue_index_job(
            job_id=f"{batch_id}-{entity_kind}-{index_kind}",
            namespace=namespace,
            entity_kind=entity_kind,
            entity_id=entity_id,
            index_kind=index_kind,
            op=op,
        )

    return node_id, edge_id


def _profile_job_loop_scenario(
    root: Path,
    *,
    mode: str,
    iterations: int,
    include_monitoring: bool,
    use_validation_cache: bool,
) -> dict[str, Any]:
    conversation_engine = _build_in_memory_engine(root / "conv", graph_type="conversation")
    outer = TimingRecorder()
    inner = TimingRecorder()
    if mode in {"eager_reconcile", "apply_only"}:
        conversation_engine.indexing._profile_hook = inner.add  # type: ignore[attr-defined]

    monitor = SysMonitoringWallProfiler(
        include_files=(
            "kogwistar/engine_core/indexing.py",
            "kogwistar/engine_core/in_memory_meta.py",
            "kogwistar/engine_core/engine_sqlite.py",
            "kogwistar/engine_core/in_memory_backend.py",
        ),
        include_names=(
            "reconcile_indexes",
            "apply_index_job",
            "claim_index_jobs",
            "mark_index_job_done",
            "bump_retry_and_requeue",
            "get_index_applied_fingerprint",
            "set_index_applied_fingerprint",
            "node_get",
            "edge_get",
            "node_docs_get",
            "node_refs_get",
            "edge_refs_get",
            "edge_endpoints_get",
            "edge_endpoints_upsert",
            "edge_endpoints_delete",
            "fanout_endpoints_rows",
            "model_validate_json",
        ),
    )

    method_specs: list[tuple[Any, str, str]] = [
        (conversation_engine.meta_sqlite, "claim_index_jobs", "meta.claim_index_jobs"),
        (conversation_engine.meta_sqlite, "mark_index_job_done", "meta.mark_index_job_done"),
        (conversation_engine.meta_sqlite, "bump_retry_and_requeue", "meta.bump_retry_and_requeue"),
        (conversation_engine.meta_sqlite, "get_index_applied_fingerprint", "meta.get_index_applied_fingerprint"),
        (conversation_engine.meta_sqlite, "set_index_applied_fingerprint", "meta.set_index_applied_fingerprint"),
        (conversation_engine.backend, "node_get", "backend.node_get"),
        (conversation_engine.backend, "edge_get", "backend.edge_get"),
        (conversation_engine.backend, "node_docs_get", "backend.node_docs_get"),
        (conversation_engine.backend, "node_refs_get", "backend.node_refs_get"),
        (conversation_engine.backend, "edge_refs_get", "backend.edge_refs_get"),
        (conversation_engine.backend, "edge_endpoints_get", "backend.edge_endpoints_get"),
        (conversation_engine.backend, "edge_endpoints_upsert", "backend.edge_endpoints_upsert"),
        (conversation_engine.backend, "edge_endpoints_delete", "backend.edge_endpoints_delete"),
        (conversation_engine.write, "fanout_endpoints_rows", "write.fanout_endpoints_rows"),
        (conversation_engine.indexing, "apply_index_job", "index.apply_index_job"),
        (conversation_engine.indexing, "reconcile_indexes", "index.reconcile_indexes"),
    ]

    setup_total_s = 0.0
    loop_total_s = 0.0
    with ExitStack() as stack:
        for obj, attr, label in method_specs:
            stack.enter_context(outer.wrap_method(obj, attr, label=label))
        if include_monitoring:
            monitor.start()
            stack.callback(monitor.stop)

        for iteration in range(1, int(iterations) + 1):
            namespace = f"{mode}-{iteration}-{uuid.uuid4().hex}"
            batch_id = f"{mode}-{iteration}-{uuid.uuid4().hex}"
            seed_started = time.perf_counter()
            _seed_index_job_batch(
                conversation_engine, namespace=namespace, batch_id=batch_id
            )
            setup_total_s += time.perf_counter() - seed_started

            loop_started = time.perf_counter()
            if mode == "claim_only":
                conversation_engine.meta_sqlite.claim_index_jobs(
                    limit=10, lease_seconds=60, namespace=namespace
                )
                loop_total_s += time.perf_counter() - loop_started
                continue

            if mode == "apply_only":
                claimed = conversation_engine.meta_sqlite.claim_index_jobs(
                    limit=10, lease_seconds=60, namespace=namespace
                )
                if not claimed:
                    raise AssertionError(
                        "expected claimed jobs for apply-only scenario"
                    )
                cache: dict[tuple[str, str, str], object] | None = (
                    {} if use_validation_cache else None
                )
                for job in claimed:
                    conversation_engine.indexing.apply_index_job(
                        job_id=str(job.job_id),
                        entity_kind=str(job.entity_kind),
                        entity_id=str(job.entity_id),
                        index_kind=str(job.index_kind),
                        op=str(job.op),
                        namespace=namespace,
                        validated_entity_cache=cache,
                    )
                    conversation_engine.meta_sqlite.mark_index_job_done(str(job.job_id))
                loop_total_s += time.perf_counter() - loop_started
                continue

            if mode == "eager_reconcile":
                conversation_engine.reconcile_indexes(max_jobs=20, namespace=namespace)
                loop_total_s += time.perf_counter() - loop_started
                continue

            raise ValueError(f"unknown profile mode: {mode!r}")

    total_ms = round(loop_total_s * 1000.0, 3)
    return {
        "mode": str(mode),
        "iterations": int(iterations),
        "use_validation_cache": bool(use_validation_cache),
        "scenario_total_ms": total_ms,
        "seed_total_ms": round(setup_total_s * 1000.0, 3),
        "method_timings": outer.to_dict(),
        "internal_timings": inner.to_dict(),
        "monitoring_timings": monitor.to_dict() if include_monitoring else {},
    }


def profile_in_memory_index_job_breakdown(
    output_root: str | Path | None = None,
    *,
    iterations: int = 1,
    include_monitoring: bool = False,
    use_validation_cache: bool = True,
) -> dict[str, Any]:
    root = (
        Path(output_root)
        if output_root is not None
        else (Path.cwd() / ".tmp_perf_profiles" / f"kogwistar_job_breakdown_{uuid.uuid4().hex}")
    )
    root.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "claim_only": _profile_job_loop_scenario(
            root / "claim_only",
            mode="claim_only",
            iterations=iterations,
            include_monitoring=include_monitoring,
            use_validation_cache=use_validation_cache,
        ),
        "apply_only": _profile_job_loop_scenario(
            root / "apply_only",
            mode="apply_only",
            iterations=iterations,
            include_monitoring=include_monitoring,
            use_validation_cache=use_validation_cache,
        ),
        "eager_reconcile": _profile_job_loop_scenario(
            root / "eager_reconcile",
            mode="eager_reconcile",
            iterations=iterations,
            include_monitoring=include_monitoring,
            use_validation_cache=use_validation_cache,
        ),
    }
    return {
        "python": sys.version,
        "sys_monitoring_available": bool(hasattr(sys, "monitoring")),
        "output_root": str(root),
        "iterations": int(iterations),
        "scenarios": scenarios,
    }


def profile_in_memory_checkpoint_write_mode(
    output_root: str | Path | None = None,
    *,
    iterations: int = 5,
    fast_trace_persistence: bool,
    include_monitoring: bool = False,
    use_validation_cache: bool = True,
) -> dict[str, Any]:
    root = (
        Path(output_root)
        if output_root is not None
        else (
            Path.cwd() / ".tmp_perf_profiles" / f"kogwistar_perf_{uuid.uuid4().hex}"
        )
    )
    root.mkdir(parents=True, exist_ok=True)

    return {
        "python": sys.version,
        "sys_monitoring_available": bool(hasattr(sys, "monitoring")),
        "output_root": str(root),
        "iterations": int(iterations),
        "scenarios": {
            "fast_inline" if fast_trace_persistence else "eager_reconcile": _profile_one_scenario(
                root / ("fast_inline" if fast_trace_persistence else "eager_reconcile"),
                iterations=iterations,
                fast_trace_persistence=fast_trace_persistence,
                include_monitoring=include_monitoring,
                use_validation_cache=use_validation_cache,
            )
        },
    }


def profile_in_memory_checkpoint_write(
    output_root: str | Path | None = None,
    *,
    iterations: int = 5,
    compare_fast_path: bool = True,
    include_monitoring: bool = False,
    use_validation_cache: bool = True,
) -> dict[str, Any]:
    root = (
        Path(output_root)
        if output_root is not None
        else (Path.cwd() / ".tmp_perf_profiles" / f"kogwistar_perf_{uuid.uuid4().hex}")
    )
    root.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "eager_reconcile": _profile_one_scenario(
            root / "eager_reconcile",
            iterations=iterations,
            fast_trace_persistence=False,
            include_monitoring=include_monitoring,
            use_validation_cache=use_validation_cache,
        )
    }
    if compare_fast_path:
        scenarios["fast_inline"] = _profile_one_scenario(
            root / "fast_inline",
            iterations=iterations,
            fast_trace_persistence=True,
            include_monitoring=include_monitoring,
            use_validation_cache=use_validation_cache,
        )
    return {
        "python": sys.version,
        "sys_monitoring_available": bool(hasattr(sys, "monitoring")),
        "output_root": str(root),
        "iterations": int(iterations),
        "scenarios": scenarios,
    }


def format_profile_report(report: dict[str, Any]) -> str:
    lines = [
        f"Python: {report.get('python')}",
        f"Output root: {report.get('output_root')}",
        f"Iterations: {report.get('iterations')}",
    ]
    for scenario_name, scenario in (report.get("scenarios") or {}).items():
        lines.append("")
        lines.append(f"[{scenario_name}] total={scenario.get('scenario_total_ms')} ms")
        if scenario.get("seed_total_ms") is not None:
            lines.append(f"Seed/setup={scenario.get('seed_total_ms')} ms")
        lines.append("Top method timings:")
        method_timings = scenario.get("method_timings") or {}
        for name, stats in list(method_timings.items())[:12]:
            lines.append(
                "  "
                + f"{name}: total={stats.get('total_ms')} ms "
                + f"avg={stats.get('avg_ms')} ms count={stats.get('count')}"
            )
        monitoring_timings = scenario.get("monitoring_timings") or {}
        if monitoring_timings:
            lines.append("Top sys.monitoring timings:")
            for name, stats in list(monitoring_timings.items())[:12]:
                lines.append(
                    "  "
                    + f"{name}: total={stats.get('total_ms')} ms "
                    + f"avg={stats.get('avg_ms')} ms count={stats.get('count')}"
                )
        internal_timings = scenario.get("internal_timings") or {}
        if internal_timings:
            lines.append("Top internal timings:")
            for name, stats in list(internal_timings.items())[:12]:
                lines.append(
                    "  "
                    + f"{name}: total={stats.get('total_ms')} ms "
                    + f"avg={stats.get('avg_ms')} ms count={stats.get('count')}"
                )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile in-memory runtime checkpoint/trace write path."
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--compare-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile both eager reconcile and fast inline modes.",
    )
    parser.add_argument(
        "--include-monitoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect Python function timings with sys.monitoring when available.",
    )
    parser.add_argument(
        "--validation-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the transient entity validation cache for index reconciliation.",
    )
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the full JSON report instead of the text summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = profile_in_memory_checkpoint_write(
        args.output_root,
        iterations=int(args.iterations),
        compare_fast_path=bool(args.compare_fast_path),
        include_monitoring=bool(args.include_monitoring),
        use_validation_cache=bool(args.validation_cache),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_profile_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
