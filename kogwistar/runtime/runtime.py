from __future__ import annotations

import logging
import pathlib
import queue
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from os import PathLike
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from kogwistar.engine_core.sqlite_context import sqlite_execution_bound
from kogwistar.implementation_selection import (
    ImplementationSelectionSnapshot,
    capture_implementation_selection,
    implementation_selection_bound,
)
from kogwistar.cdc.sqlite_sink import _get_shared_sqlite_sink
from kogwistar.id_provider import stable_id
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime.budget import (
    BudgetAttribution,
    budget_ledger_from_state,
)
from kogwistar.runtime.budget_adapters import adapt_budget_events
from kogwistar.runtime.models import (
    RunFailure,
    RunSuspended,
    StateUpdate,
    WorkflowDesignArtifact,
    WorkflowEdge as WorkflowEdge,
    WorkflowInvocationRequest,
    WorkflowNode as WorkflowNode,
    WorkflowRunNode,
    StepRunResult as StepRunResult,
    WorkflowState,
    WorkflowStepExecNode,
)
from kogwistar.utils.log import bind_log_context

from .base_runtime import (
    BaseRuntime,
    _ChildRunResult,
    apply_state_update_inplace,
    validate_initial_state,
)
from .contract import CancellationChecker, Predicate
from .design import validate_workflow_design
from .join_topology import (
    _compute_may_reach_join_bitsets as _compute_may_reach_join_bitsets,
    _iter_bits,
)
from .native_contracts import join_arrival_result, successor_plan
from .join_state import (
    JoinToken,
    RuntimeJoinSnapshot,
)
from .runtime_join_state import RuntimeJoinState
from .runtime_routing import WorkflowRouteNextMixin
from .runtime_work_state import RuntimeWorkState
from .step_worker import CompletedStep, SyncStepWorker
from .run_ownership import (
    DEFAULT_RUN_OWNERSHIP_LEASE_MS,
    RunOwnership,
)
from .runtime_artifacts import (
    TraceSpanPayload as TraceSpanPayload,
)
from .runtime_checkpoints import (
    CheckpointRuntimeMixin,
    RuntimeMetadataStore as RuntimeMetadataStore,
    WorkflowRunOwnershipError as WorkflowRunOwnershipError,
)
from .runtime_terminals import (
    CancelInfo as CancelInfo,
    TerminalPersistenceMixin,
)
from .runtime_trace_persistence import RuntimeTracePersistenceMixin
from .runtime_invocations import (
    WorkflowInvocationMixin,
    derive_child_authority_context as derive_child_authority_context,
)
from .runtime_resume import SuspendedRunResumeMixin
from .runtime_conversation import (
    ConversationBackendLike as ConversationBackendLike,
    ConversationRuntimeMixin,
    CurrentUserSequenceStore as CurrentUserSequenceStore,
    _json_int as _json_int,
    _json_string_list as _json_string_list,
)
from .step_context import (
    LaneMessageEventSinkLike,
    LaneMessageSenderLike,
    RouteDecision as RouteDecision,
    RunResult as RunResult,
    StepResolver,
    StepContext as StepContext,
)
from .telemetry import (
    EventEmitter,
    EventSink,
    TraceContext,
)

if TYPE_CHECKING:
    from ..engine_core.engine import GraphKnowledgeEngine


RunID = uuid.UUID | str
Json = JsonValue
State = dict[str, Json]
# Result = Json

def sink_observes_otel(sink: object) -> bool:
    """Detect an explicitly supplied OTel sink without importing OTel."""

    if sink is None:
        return False
    if type(sink).__module__ == "kogwistar.runtime.telemetry_otel":
        return True
    return any(sink_observes_otel(item) for item in getattr(sink, "_sinks", ()))
def _now_ms() -> int:
    return int(time.time() * 1000)


class WorkflowRuntime(
    CheckpointRuntimeMixin,
    TerminalPersistenceMixin,
    RuntimeTracePersistenceMixin,
    ConversationRuntimeMixin,
    WorkflowInvocationMixin,
    SuspendedRunResumeMixin,
    WorkflowRouteNextMixin,
    BaseRuntime[StepResolver],
):
    CHECKPOINT_SCHEMA_VERSION = 1
    """
    Core engine for executing graph-based **workflow designs**.

    This runtime:
    1.  Reads the workflow design (nodes/edges) from the `workflow_engine`.
    2.  Executes steps using the provided `step_resolver`.
    3.  Manages state transitions, branching, and join/barrier logic.
    4.  Persists execution traces (`WorkflowRunNode`, `WorkflowStepExecNode`) and checkpoints
        to the `conversation_engine` (or whichever engine is designated for traces).

    It is the backend execution engine used by high-level orchestrators like `ConversationOrchestrator`.

    ``otel_enabled`` is an explicit, optional observer switch. It is disabled
    by default and never inferred from environment variables. When enabled, the
    optional OTel sink is composed with the normal sink without changing
    workflow authority.
    """

    def __init__(
        self,
        *,
        workflow_engine: GraphKnowledgeEngine,
        conversation_engine: GraphKnowledgeEngine,
        step_resolver: StepResolver,
        predicate_registry: dict[str, Predicate],
        checkpoint_every_n_steps: int = 1,
        max_workers: int = 4,
        transaction_mode: str | None = None,  # "step" | "run" | "none" (default auto)
        # Quick-fix knobs for nested runs / resource sharing
        trace: bool = True,
        events: EventEmitter | None = None,
        sink: EventSink | None = None,
        otel_enabled: bool = False,
        cancel_requested: CancellationChecker | None = None,
        lane_message_sender: LaneMessageSenderLike | None = None,
        lane_message_event_sink: LaneMessageEventSinkLike | None = None,
        fast_trace_persistence: bool | None = None,
        max_nested_workflow_depth: int = 8,
        run_ownership_lease_ms: int = DEFAULT_RUN_OWNERSHIP_LEASE_MS,
    ) -> None:
        workflow_selection = getattr(
            workflow_engine, "_implementation_selection_snapshot", None
        )
        conversation_selection = getattr(
            conversation_engine, "_implementation_selection_snapshot", None
        )
        if (
            workflow_selection is not None
            and conversation_selection is not None
            and workflow_selection != conversation_selection
        ):
            raise ValueError(
                "workflow and conversation engines use different implementation selections"
            )
        self._implementation_selection_snapshot: ImplementationSelectionSnapshot = (
            workflow_selection
            or conversation_selection
            or capture_implementation_selection()
        )
        self.workflow_engine: GraphKnowledgeEngine = workflow_engine
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.step_resolver: StepResolver = step_resolver
        self.predicate_registry = predicate_registry
        self.checkpoint_every_n_steps = max(1, int(checkpoint_every_n_steps))
        self.max_workers = max_workers
        self.max_nested_workflow_depth = max(0, int(max_nested_workflow_depth))
        if int(run_ownership_lease_ms) <= 0:
            raise ValueError("run_ownership_lease_ms must be positive")
        self.run_ownership_lease_ms = int(run_ownership_lease_ms)
        self.cancel_requested = cancel_requested
        self.lane_message_sender = lane_message_sender
        if self.lane_message_sender is None:
            default_sender = getattr(conversation_engine, "send_lane_message", None)
            if callable(default_sender):
                self.lane_message_sender = default_sender
        self.lane_message_event_sink = lane_message_event_sink
        if fast_trace_persistence is None:
            self.fast_trace_persistence = (
                str(getattr(conversation_engine, "backend_kind", "")).lower() == "memory"
            )
        else:
            self.fast_trace_persistence = bool(fast_trace_persistence)
        # Transaction policy: default to per-step transactions when conversation_engine is Postgres-backed.
        if transaction_mode is None:
            try:
                from kogwistar.engine_core.postgres_backend import (
                    PgVectorBackend,
                )

                self.transaction_mode = (
                    "step"
                    if isinstance(
                        getattr(conversation_engine, "backend", None), PgVectorBackend
                    )
                    else "none"
                )
            except Exception:
                self.transaction_mode = "none"
        else:
            self.transaction_mode = str(transaction_mode)
        self.state_lock: dict[RunID, Lock] = {}  # look up of run specific state lock
        self._otel_sink: EventSink | None = None

        # --------------------------------------------------------------
        # Trace plumbing (quick-fix for nested runtimes)
        # --------------------------------------------------------------
        # If an EventEmitter is provided, reuse it (prevents duplicate writers).
        # OTel is deliberately explicit and cannot silently replace a caller's
        # emitter; callers that own an emitter must compose its sink themselves.
        if events is not None:
            if otel_enabled:
                raise ValueError(
                    "otel_enabled cannot be combined with an existing events emitter; "
                    "compose the OpenTelemetry sink before creating EventEmitter"
                )
            self.emitter = events
            self.sink = getattr(events, "sink", None)
        else:
            if sink is not None:
                self.sink = sink
            elif not trace:
                self.sink = None
            else:
                self.sink = None
                if getattr(workflow_engine, "persist_directory", None) is not None:
                    persist_directory = workflow_engine.persist_directory
                    if persist_directory is None:
                        raise RuntimeError("trace persistence directory disappeared")
                    db_path = str(pathlib.Path(persist_directory) / "wf_trace.sqlite")
                    # Share a single sink per db_path in this process to reduce SQLite contention.
                    self.sink = _get_shared_sqlite_sink(db_path, drop_when_full=True)
            if otel_enabled:
                # Keep the optional dependency out of TraceContext and the
                # module import path. Missing OTel disables only this observer.
                from .telemetry import FanoutEventSink
                from .telemetry_otel import try_create_opentelemetry_sink

                otel_sink = try_create_opentelemetry_sink()
                if otel_sink is not None:
                    self._otel_sink = otel_sink
                    self.sink = (
                        FanoutEventSink([self.sink, otel_sink])
                        if self.sink is not None
                        else otel_sink
                    )
                else:
                    logging.getLogger("workflow.trace").info(
                        "OpenTelemetry disabled: optional dependency is unavailable"
                    )
            self.emitter = EventEmitter(
                sink=self.sink, logger=logging.getLogger("workflow.trace")
            )

    def close(self, timeout: float = 1.0) -> None:
        """Boundedly stop an OTel observer created by this runtime.

        Caller-owned sinks and shared SQLite sinks remain caller-owned. This
        prevents closing a sink shared by sibling runtimes while still giving
        explicit OTel opt-in a deterministic shutdown seam.
        """

        otel_sink = self._otel_sink
        if otel_sink is None:
            return
        self._otel_sink = None
        try:
            otel_sink.close(max(float(timeout), 0.0))
        except Exception:
            logging.getLogger("workflow.trace").exception(
                "OpenTelemetry sink close failed"
            )

    @contextmanager
    def _trace_write_mode(self) -> Iterator[None]:
        eng = self.conversation_engine
        if not self.fast_trace_persistence:
            yield
            return
        prev_idx = getattr(eng, "_phase1_enable_index_jobs", False)
        eng._phase1_enable_index_jobs = False
        try:
            yield
        finally:
            eng._phase1_enable_index_jobs = prev_idx

    def _maybe_step_uow(self) -> AbstractContextManager[object]:
        """Open a UoW transaction only when transaction_mode=='step'.

        This keeps callsites clean:
            with self._maybe_step_uow():
                ...
        """
        if self.transaction_mode == "step":
            return self.conversation_engine.uow()
        return nullcontext()

    def _close_sandbox_run(self, run_id: str) -> None:
        super()._close_sandbox_run(run_id)

    def _should_step_uow(self, op: str, state: WorkflowState) -> bool:
        """Best-effort guard to avoid holding a step UoW across nested workflow execution.

        Quick fix: if the current op is marked as a "nested op" by the resolver
        (MappingStepResolver.nested_ops), we skip the outer step UoW so the nested
        workflow can perform its own per-step UoWs without deadlocking/contending.

        Fallback: if resolver doesn't expose nested_ops, keep the old heuristic for
        `answer` calling `agent.answer_workflow_v2`.
        """
        if self.transaction_mode != "step":
            return False
        try:
            nested_ops = getattr(self.step_resolver, "nested_ops", None)
            if isinstance(nested_ops, set) and str(op) in nested_ops:
                return False

            if str(op) == "answer":
                deps = state.get("_deps") if isinstance(state, dict) else None
                if isinstance(deps, dict):
                    agent = deps.get("agent")
                    if agent is not None and hasattr(agent, "answer_workflow_v2"):
                        return False
        except Exception:
            return True
        return True

    def _persist_workflow_design_artifact(
        self, design: WorkflowDesignArtifact
    ) -> None:
        """Persist a workflow design artifact into the workflow engine."""
        super()._persist_workflow_design_artifact(design)

    def _child_workflow_initial_state(
        self,
        *,
        parent_state: WorkflowState,
        invocation: WorkflowInvocationRequest,
    ) -> WorkflowState:
        return super()._child_workflow_initial_state(
            parent_state=parent_state,
            invocation=invocation,
        )

    def _apply_workflow_invocation_result(
        self,
        *,
        state: WorkflowState,
        invocation: WorkflowInvocationRequest,
        child_result: _ChildRunResult,
    ) -> None:
        super()._apply_workflow_invocation_result(
            state=state,
            invocation=invocation,
            child_result=child_result,
        )

    def apply_state_update(
        self,
        mute_state: WorkflowState,
        state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate],
        update: dict | None = None,
    ) -> None:
        apply_state_update_inplace(
            mute_state,
            state_update,
            update,
            state_schema=getattr(self.step_resolver, "_state_schema", None),
        )


    @implementation_selection_bound
    @sqlite_execution_bound("workflow_engine", "conversation_engine")
    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None = None,  # parent run may trigger another run in a node
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
        _run_metadata: Mapping[str, JsonValue] | None = None,
        _trace_context: TraceContext | None = None,
        _authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult:
        """
        Returns (final_state, run_id).

        Design lives in workflow_engine.
        Traces/checkpoints live in conversation_engine.
        """
        with bind_log_context(
            conversation_id=conversation_id, workflow_run_id=f"{workflow_id}--{run_id}"
        ):
            self.workflow_id = str(workflow_id)
            if _authority_context is not None and "effective_capabilities" in _authority_context:
                initial_state["effective_capabilities"] = list(
                    _authority_context.get("effective_capabilities", ())
                )
            initial_state.setdefault("_wf_invocation_path", [str(workflow_id)])
            initial_state.setdefault("_wf_current_run_id", str(run_id or ""))
            self.ensure_budget_ledger(
                initial_state,
                ceilings=(
                    _authority_context.get("budget_limits")
                    if _authority_context is not None
                    else None
                ),
            )
            # repair fields auto set by default schema
            state_schema = getattr(self.step_resolver, "_state_schema", None)
            if state_schema and isinstance(state_schema, dict):
                for k, v in state_schema.items():
                    if v == "u" and k not in initial_state:
                        initial_state[k] = None
                    if v == "a" and k not in initial_state:
                        initial_state[k] = []

            validate_initial_state(initial_state)

            requested_run_id = run_id
            run_id = run_id or f"run|{uuid.uuid4()}"
            ownership: RunOwnership | None = None
            if requested_run_id is not None:
                existing_terminal = self._terminal_run_result(
                    conversation_id=str(conversation_id), run_id=str(run_id)
                )
                if existing_terminal is not None and not (
                    _resume_step_seq is not None and existing_terminal.status == "failure"
                ):
                    return existing_terminal
                ownership = self._claim_workflow_run(
                    conversation_id=str(conversation_id),
                    workflow_id=str(workflow_id),
                    run_id=str(run_id),
                    allow_resume_takeover=_resume_step_seq is not None,
                )
            from .rust_runtime_authority import (
                run_with_rust_authority,
                rust_runtime_authority_selected,
            )

            if rust_runtime_authority_selected():
                if _resume_step_seq is not None or _resume_last_exec_node is not None:
                    raise NotImplementedError(
                        "Rust runtime authority uses the versioned resume endpoint; "
                        "Python resume markers are unsupported"
                    )
                return run_with_rust_authority(
                    self,
                    workflow_id=str(workflow_id),
                    conversation_id=str(conversation_id),
                    turn_node_id=(str(turn_node_id) if turn_node_id is not None else None),
                    initial_state=initial_state,
                    run_id=str(run_id),
                    cache_dir=cache_dir,
                )
            self.state_lock[str(run_id)] = Lock()

            # an interworker, orchestrator message channel
            mq: queue.Queue[dict[str, Json]] = queue.Queue(maxsize=10000)

            start, nodes, adj = validate_workflow_design(
                workflow_engine=self.workflow_engine,
                workflow_id=workflow_id,
                predicate_registry=self.predicate_registry,
                resolver=self.step_resolver,
            )
            start_node_id = start.id or "start"

            # Keep graph analysis and mutable join counters outside the scheduler
            # body; the aliases preserve the scheduler's established snapshot path.
            join_runtime = RuntimeJoinState.for_workflow(nodes=nodes, adjacency=adj)
            join_node_ids = join_runtime.join_node_ids
            _join_is_merge = join_runtime.join_is_merge
            _may_reach_join = join_runtime.may_reach_join
            _join_pos = join_runtime.join_pos
            _join_outstanding = join_runtime.join_outstanding
            _join_waiters = join_runtime.join_waiters
            _inc = join_runtime.inc
            _dec = join_runtime.dec
            _rt_join_snapshot = join_runtime.snapshot

            def _rt_join_set(payload: RuntimeJoinSnapshot) -> None:
                # Stored in workflow state so checkpoints can resume joins.
                join_runtime.set_snapshot(state, payload)

            def _rt_join_restore() -> list[JoinToken] | None:
                return join_runtime.restore(state)

            # start, nodes, adj = load_workflow_design(workflow_engine=self.workflow_engine, workflow_id=workflow_id)

            state: WorkflowState = initial_state
            step_seq = int(_resume_step_seq) if _resume_step_seq is not None else 0

            # Establish the execution span before persisting the run record so
            # restart/reconciliation can find trace identity even before the
            # first checkpoint exists.
            run_trace_context = (
                _trace_context
                if _trace_context is not None
                and str(_trace_context.run_id) == str(run_id)
                and (
                    _trace_context.has_valid_w3c_ids
                    or not (self._otel_sink is not None or sink_observes_otel(self.sink))
                )
                else (
                    _trace_context.child_run(
                        run_id=str(run_id),
                        token_id=str(run_id),
                        step_seq=0,
                        node_id=start_node_id,
                    )
                    if _trace_context is not None
                    and (
                        _trace_context.has_valid_w3c_ids
                        or not (self._otel_sink is not None or sink_observes_otel(self.sink))
                    )
                    else (
                        TraceContext.new_root(
                            run_id=str(run_id),
                            token_id=str(run_id),
                            step_seq=0,
                            node_id=start_node_id,
                            conversation_id=str(conversation_id)
                            if conversation_id is not None
                            else None,
                            turn_node_id=str(turn_node_id)
                            if turn_node_id is not None
                            else None,
                        )
                        if self._otel_sink is not None or sink_observes_otel(self.sink)
                        else None
                    )
                )
            )

            self_run_id = str(run_id)
            self_conversation_id = str(conversation_id)
            self_turn_node_id = str(turn_node_id)

            def _lifecycle_trace_context(
                *,
                token_id: str,
                step_seq: int,
                node_id: str,
                attempt: int = 1,
                run_id: str | None = None,
                conversation_id: str | None = None,
                turn_node_id: str | None = None,
            ) -> TraceContext:
                """Keep lifecycle events in this run's trace across all choke points."""
                effective_run_id = str(run_id or self_run_id)
                effective_conversation_id = str(
                    conversation_id or self_conversation_id
                )
                effective_turn_node_id = str(turn_node_id or self_turn_node_id)
                base = run_trace_context or TraceContext(
                    run_id=effective_run_id,
                    token_id=effective_run_id,
                    step_seq=0,
                    node_id=start_node_id,
                    attempt=1,
                    conversation_id=effective_conversation_id,
                    turn_node_id=effective_turn_node_id,
                )
                return base.child_span(
                    token_id=str(token_id),
                    step_seq=int(step_seq),
                    node_id=str(node_id),
                    attempt=int(attempt),
                )

            # Persist workflow_run node in conversation_engine
            wf_run_root_node = self._persist_workflow_run(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                turn_node_id=str(turn_node_id or ""),
                status="running",
                run_metadata=_run_metadata,
                trace_context=run_trace_context,
            )
            # trace: workflow run started
            try:
                # The run context itself is the execution span identity.  Do
                # not create an un-emitted child span here: nested runs use
                # this event's span as their durable parent.
                tc_run = run_trace_context or _lifecycle_trace_context(
                    token_id=str(run_id), step_seq=0, node_id="start"
                )
                self.emitter.emit(
                    type="workflow_run_started",
                    ctx=tc_run,
                    payload={"workflow_id": str(workflow_id)},
                )
            except Exception:
                pass
            # state_mutex_lock: threading.Lock = threading.Lock()
            scheduled_q: queue.Queue[tuple[str, int, str, str | None]] = (
                queue.Queue()
            )  # (node_id, mask, token_id, parent_token_id)
            pending_tokens: set[tuple[str, int, str, str | None]] = set()
            inflight_tokens: set[tuple[str, int, str, str | None]] = set()
            suspended_tokens: dict[str, tuple[str, int, str, str | None]] = {}
            done_q: queue.Queue[CompletedStep] = queue.Queue()
            cancel_pending = False
            cancel_info: CancelInfo | None = None
            run_suspended = False
            run_failed = False
            run_errors: list[str] = []

            graveyard: queue.Queue[CompletedStep] = queue.Queue()

            # Keep the same scheduler-only condition boundary; never hold it
            # across storage I/O, resolver code, or queue waits.
            work_state = RuntimeWorkState()
            work_cond = work_state.condition
            _work_add = work_state.add
            _work_done = work_state.done
            _work_snapshot = work_state.snapshot

            def _persist_rt_join_runtime() -> None:
                # Persist both pending + inflight as pending-on-resume (idempotent).
                # Suspended tokens are stored separately so they are parked, not re-scheduled.
                combined = sorted(pending_tokens.union(inflight_tokens))
                suspended = sorted(suspended_tokens.values())
                _rt_join_set(_rt_join_snapshot(combined, suspended=suspended))

            def _enter_failed() -> bool:
                nonlocal run_failed
                if run_failed:
                    return False
                run_failed = True

                while True:
                    try:
                        nid, mask, tok, parent_tok = scheduled_q.get_nowait()
                    except queue.Empty:
                        break
                    pending_tokens.discard((str(nid), int(mask), str(tok), parent_tok))
                    _dec(int(mask))
                    _work_done(1)

                for jid in join_node_ids:
                    waiters = list(_join_waiters.get(jid, []))
                    if not waiters:
                        continue
                    _join_waiters[jid] = []
                    _work_done(len(waiters))
                    for wm, _tok, _parent in waiters:
                        _dec(int(wm))

                _persist_rt_join_runtime()
                return True

            def _cancel_requested() -> bool:
                nonlocal cancel_info
                if bool(self.cancel_requested and self.cancel_requested(str(run_id))):
                    if cancel_info is None:
                        cancel_info = {
                            "source": "callback",
                            "node_id": None,
                            "seq": None,
                            "watermark": None,
                        }
                    return True
                graph_cancel = self._find_eligible_cancel_request(
                    conversation_id=str(conversation_id),
                    run_id=str(run_id),
                )
                if graph_cancel is not None:
                    graph_cancel["source"] = "graph"
                    cancel_info = graph_cancel
                    return True
                return False

            def _enter_cancelling(
                reason_node_id: str = "run", token_id: str | None = None
            ) -> bool:
                nonlocal cancel_pending
                if cancel_pending or not _cancel_requested():
                    return cancel_pending
                cancel_pending = True
                while True:
                    try:
                        nid, mask, tok, parent_tok = scheduled_q.get_nowait()
                        pending_tokens.discard(
                            (str(nid), int(mask), str(tok), parent_tok)
                        )
                        _work_done(1)
                    except queue.Empty:
                        break
                _persist_rt_join_runtime()
                try:
                    tc_cancel = _lifecycle_trace_context(
                        run_id=str(run_id),
                        token_id=str(token_id or run_id),
                        step_seq=int(step_seq),
                        node_id=str(reason_node_id),
                        attempt=1,
                        conversation_id=str(conversation_id)
                        if conversation_id is not None
                        else None,
                        turn_node_id=str(turn_node_id)
                        if turn_node_id is not None
                        else None,
                    )
                    self.emitter.emit(
                        type="workflow_run_cancelling",
                        ctx=tc_cancel,
                        payload={"workflow_id": str(workflow_id)},
                    )
                except Exception:
                    pass
                return True

            start_id = start.safe_get_id()

            # Resume-safe join runtime: if a checkpoint restored pending tokens + join counters,
            # seed the scheduler from that payload. Otherwise start from the workflow start node.
            restored = _rt_join_restore()
            if restored:
                for nid, mask, token_id, parent_token_id in restored:
                    t = (str(nid), int(mask), str(token_id), parent_token_id)
                    pending_tokens.add(t)
                    scheduled_q.put(t)
                _work_add(len(restored))
                _persist_rt_join_runtime()
            else:
                start_mask = int(_may_reach_join.get(start_id, 0))
                _inc(start_mask)
                t = (
                    str(start_id),
                    int(start_mask),
                    run_id,  # uuid.uuid4().hex, # token_id = run-id
                    None,
                )
                pending_tokens.add(t)
                scheduled_q.put(t)
                _work_add(1)
                _persist_rt_join_runtime()

            def renew_run_ownership() -> None:
                nonlocal ownership
                ownership = self._renew_workflow_run(
                    conversation_id=str(conversation_id),
                    run_id=str(run_id),
                    ownership=ownership,
                )

            def apply_child_result(
                state: WorkflowState,
                invocation: WorkflowInvocationRequest,
                child_result: RunResult,
            ) -> None:
                with self.state_lock[str(run_id)]:
                    self._apply_workflow_invocation_result(
                        state=state,
                        invocation=invocation,
                        child_result=child_result,
                    )

            worker = SyncStepWorker(
                runtime=self,
                workflow_id=str(workflow_id),
                run_id=str(run_id),
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                nodes=nodes,
                trace_context=run_trace_context,
                message_queue=mq,
                done_queue=done_q,
                work_condition=work_cond,
                authority_context=_authority_context,
                cancel_requested=_cancel_requested,
                parent_lease_renew=_parent_lease_renew,
                renew_run_ownership=renew_run_ownership,
                apply_child_result=apply_child_result,
                now_ms=_now_ms,
            )

            inflight: dict[tuple[str, int, str], Future[None]] = {}
            last_exec_node = (
                _resume_last_exec_node
                if _resume_last_exec_node is not None
                else wf_run_root_node
            )
            with ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix=f"rt-wf-{workflow_id}"
            ) as pool:
                while True:
                    _enter_cancelling()
                    if (
                        cancel_pending
                        and not inflight
                        and scheduled_q.empty()
                        and done_q.empty()
                    ):
                        self._persist_cancelled_terminal(
                            conversation_id=str(conversation_id),
                            workflow_id=str(workflow_id),
                            run_id=str(run_id),
                            accepted_step_seq=max(-1, int(step_seq) - 1),
                            cancel_info=cancel_info,
                            last_processed_node_id=(
                                str(last_exec_node.safe_get_id())
                                if last_exec_node is not None
                                else None
                            ),
                        )
                        # self._update_workflow_run_status(conversation_id, run_id, "cancelled")
                        try:
                            tc_done = _lifecycle_trace_context(
                                token_id=str(run_id), step_seq=int(step_seq), node_id="run"
                            )
                            self.emitter.emit(
                                type="workflow_run_cancelled",
                                ctx=tc_done,
                                payload={"workflow_id": str(workflow_id)},
                            )
                        except Exception:
                            pass
                        self._close_sandbox_run(run_id)
                        self.state_lock.pop(run_id, None)
                        return RunResult(
                            final_state=state, run_id=run_id, mq=mq, status="cancelled"
                        )
                    if (
                        not cancel_pending
                        and run_suspended
                        and not inflight
                        and scheduled_q.empty()
                        and done_q.empty()
                    ):
                        # Join waiters can remain parked behind a suspended branch.
                        # They are persisted in _rt_join and should not keep the run alive.
                        _persist_rt_join_runtime()
                        try:
                            tc_done = _lifecycle_trace_context(
                                token_id=str(run_id), step_seq=int(step_seq), node_id="run"
                            )
                            self.emitter.emit(
                                type="workflow_run_suspended",
                                ctx=tc_done,
                                payload={"workflow_id": str(workflow_id)},
                            )
                        except Exception:
                            pass
                        self._close_sandbox_run(run_id)
                        self._release_workflow_run_for_resume(
                            conversation_id=str(conversation_id),
                            run_id=str(run_id),
                            ownership=ownership,
                        )
                        self.state_lock.pop(run_id, None)
                        return RunResult(
                            final_state=state,
                            run_id=run_id,
                            mq=mq,
                            status="suspended",
                        )
                    if (
                        not cancel_pending
                        and _work_snapshot() == 0
                        and not inflight
                        and scheduled_q.empty()
                        and done_q.empty()
                    ):
                        if run_failed:
                            self._persist_failed_terminal(
                                conversation_id=str(conversation_id),
                                workflow_id=str(workflow_id),
                                run_id=str(run_id),
                                accepted_step_seq=max(-1, int(step_seq) - 1),
                                errors=list(run_errors),
                                last_processed_node_id=(
                                    str(last_exec_node.safe_get_id())
                                    if last_exec_node is not None
                                    else None
                                ),
                            )
                            # A failed run may still have a durable checkpoint
                            # that is safe to resume.  Do not leave its lease
                            # live after returning the failure result, or the
                            # next worker cannot recover from that checkpoint.
                            if (
                                ownership is not None
                                and self._latest_checkpoint_for_run(
                                    conversation_id=str(conversation_id),
                                    run_id=str(run_id),
                                )
                                is not None
                            ):
                                self._release_workflow_run_for_resume(
                                    conversation_id=str(conversation_id),
                                    run_id=str(run_id),
                                    ownership=ownership,
                                )
                            try:
                                tc_done = _lifecycle_trace_context(
                                    token_id=str(run_id), step_seq=int(step_seq), node_id="run"
                                )
                                self.emitter.emit(
                                    type="workflow_run_failed",
                                    ctx=tc_done,
                                    payload={
                                        "workflow_id": str(workflow_id),
                                        "errors": [error[-4_000:] for error in run_errors],
                                    },
                                )
                            except Exception:
                                pass
                            self._close_sandbox_run(run_id)
                            self.state_lock.pop(run_id, None)
                            return RunResult(
                                final_state=state,
                                run_id=run_id,
                                mq=mq,
                                status="failure",
                                errors=list(run_errors),
                            )
                        if run_suspended:
                            # suspended
                            # self._update_workflow_run_status(conversation_id, run_id, "suspended")
                            # trace: workflow run suspended
                            try:
                                tc_done = _lifecycle_trace_context(
                                    token_id=str(run_id), step_seq=int(step_seq), node_id="run"
                                )
                                self.emitter.emit(
                                    type="workflow_run_suspended",
                                    ctx=tc_done,
                                    payload={"workflow_id": str(workflow_id)},
                                )
                            except Exception:
                                pass
                            self._close_sandbox_run(run_id)
                            self._release_workflow_run_for_resume(
                                conversation_id=str(conversation_id),
                                run_id=str(run_id),
                                ownership=ownership,
                            )
                            self.state_lock.pop(run_id, None)
                            return RunResult(
                                final_state=state,
                                run_id=run_id,
                                mq=mq,
                                status="suspended",
                            )
                        else:
                            # done
                            # self._update_workflow_run_status(conversation_id, run_id, "succeeded")
                            self._persist_completed_terminal(
                                conversation_id=str(conversation_id),
                                workflow_id=str(workflow_id),
                                run_id=str(run_id),
                                accepted_step_seq=max(-1, int(step_seq) - 1),
                                last_processed_node_id=(
                                    str(last_exec_node.safe_get_id())
                                    if last_exec_node is not None
                                    else None
                                ),
                            )
                            # trace: workflow run completed
                            try:
                                tc_done = _lifecycle_trace_context(
                                    token_id=str(run_id), step_seq=int(step_seq), node_id="run"
                                )
                                self.emitter.emit(
                                    type="workflow_run_completed",
                                    ctx=tc_done,
                                    payload={"workflow_id": str(workflow_id)},
                                )
                            except Exception:
                                pass
                            self._close_sandbox_run(run_id)
                            self.state_lock.pop(run_id, None)
                            return RunResult(
                                final_state=state,
                                run_id=run_id,
                                mq=mq,
                                status="succeeded",
                            )
                    queued: list[tuple[str, int, str, str | None]] = []
                    while True:
                        try:
                            queued.append(scheduled_q.get_nowait())
                        except queue.Empty:
                            break
                    tick = self._runtime_scheduler_tick(
                        pending=queued,
                        inflight=len(inflight),
                        max_workers=self.max_workers,
                        cancelling=cancel_pending,
                    )
                    for item in tick["pending"]:
                        scheduled_q.put(
                            (
                                str(item["node_id"]),
                                int(item["join_mask"]),
                                str(item["token_id"]),
                                item.get("parent_token_id"),
                            )
                        )
                    for item in tick["dispatch"]:
                        nid = str(item["node_id"])
                        mask = int(item["join_mask"])
                        token_id = str(item["token_id"])
                        parent_token_id = item.get("parent_token_id")
                        pending_tokens.discard(
                            (nid, mask, token_id, parent_token_id)
                        )
                        _persist_rt_join_runtime()

                        # If this is a join/barrier node, it doesn't execute immediately.
                        if nid in _join_waiters:
                            join_idx = _join_pos.get(nid)
                            released: tuple[int, str, str | None] | None = None
                            collapsed_work_items = 0
                            from kogwistar._rust_bridge import (
                                RustParityError,
                                runtime_apply_join_arrival,
                                runtime_implementation_mode,
                            )

                            runtime_mode = runtime_implementation_mode()
                            native_join = None
                            if runtime_mode in {"shadow", "rust"} and join_idx is not None:
                                native_join = join_arrival_result(runtime_apply_join_arrival(
                                    payload={
                                        "join_index": int(join_idx),
                                        "join_outstanding": list(_join_outstanding),
                                        "waiters": [
                                            {
                                                "join_mask": int(waiter_mask),
                                                "token_id": str(waiter_token),
                                                "parent_token_id": waiter_parent,
                                            }
                                            for waiter_mask, waiter_token, waiter_parent in _join_waiters.get(nid, [])
                                        ],
                                        "arrival": {
                                            "join_mask": int(mask),
                                            "token_id": str(token_id),
                                            "parent_token_id": parent_token_id,
                                        },
                                        "merge": bool(_join_is_merge[nid]),
                                    }
                                ))
                            if runtime_mode == "rust" and native_join is not None:
                                _join_outstanding[:] = [
                                    int(value) for value in native_join["join_outstanding"]
                                ]
                                _join_waiters[nid] = [
                                    (
                                        int(waiter["join_mask"]),
                                        str(waiter["token_id"]),
                                        waiter.get("parent_token_id"),
                                    )
                                    for waiter in native_join["waiters"]
                                ]
                                native_released = native_join.get("released")
                                if native_released is not None:
                                    released = (
                                        int(native_released["join_mask"]),
                                        str(native_released["token_id"]),
                                        native_released.get("parent_token_id"),
                                    )
                                collapsed_work_items = int(
                                    native_join["collapsed_work_items"]
                                )
                                outstanding = int(native_join["outstanding"])
                            else:
                                arrival = join_runtime.arrive(
                                    nid,
                                    mask,
                                    token_id,
                                    parent_token_id,
                                )
                                mask = arrival.mask
                                released = arrival.released
                                collapsed_work_items = arrival.collapsed_work_items
                                outstanding = arrival.outstanding
                                if runtime_mode == "shadow" and native_join is not None:
                                    python_join = {
                                        "join_outstanding": list(_join_outstanding),
                                        "waiters": [
                                            {
                                                "join_mask": int(waiter_mask),
                                                "token_id": str(waiter_token),
                                                "parent_token_id": waiter_parent,
                                            }
                                            for waiter_mask, waiter_token, waiter_parent in _join_waiters[nid]
                                        ],
                                        "released": (
                                            {
                                                "join_mask": int(released[0]),
                                                "token_id": str(released[1]),
                                                "parent_token_id": released[2],
                                            }
                                            if released is not None
                                            else None
                                        ),
                                        "collapsed_work_items": int(collapsed_work_items),
                                        "outstanding": int(outstanding),
                                    }
                                    if native_join != python_join:
                                        raise RustParityError(
                                            "Rust parity mismatch for runtime join arrival: "
                                            f"python={python_join!r}, rust={native_join!r}"
                                        )
                            # trace: token arrived at join
                            try:
                                tcj = _lifecycle_trace_context(
                                    run_id=str(run_id),
                                    token_id=str(token_id),
                                    step_seq=int(step_seq),
                                    node_id=str(nid),
                                    attempt=1,
                                    conversation_id=str(conversation_id)
                                    if conversation_id is not None
                                    else None,
                                    turn_node_id=str(turn_node_id)
                                    if turn_node_id is not None
                                    else None,
                                )
                                self.emitter.join_event(
                                    tcj,
                                    join_node_id=str(nid),
                                    kind="arrived",
                                    outstanding=outstanding,
                                )
                            except Exception:
                                pass
                            # trace: token waiting at join
                            try:
                                tcj = _lifecycle_trace_context(
                                    run_id=str(run_id),
                                    token_id=str(token_id),
                                    step_seq=int(step_seq),
                                    node_id=str(nid),
                                    attempt=1,
                                    conversation_id=str(conversation_id)
                                    if conversation_id is not None
                                    else None,
                                    turn_node_id=str(turn_node_id)
                                    if turn_node_id is not None
                                    else None,
                                )
                                self.emitter.join_event(
                                    tcj,
                                    join_node_id=str(nid),
                                    kind="waiting",
                                    outstanding=outstanding,
                                )
                            except Exception:
                                pass

                            if _join_is_merge[nid]:
                                if released is None:
                                    _persist_rt_join_runtime()
                                    continue
                                try:
                                    tcj = _lifecycle_trace_context(
                                        run_id=str(run_id),
                                        token_id=str(released[1]),
                                        step_seq=int(step_seq),
                                        node_id=str(nid),
                                        attempt=1,
                                        conversation_id=str(conversation_id)
                                        if conversation_id is not None
                                        else None,
                                        turn_node_id=str(turn_node_id)
                                        if turn_node_id is not None
                                        else None,
                                    )
                                    self.emitter.join_event(
                                        tcj,
                                        join_node_id=str(nid),
                                        kind="released",
                                        outstanding=0,
                                    )
                                except Exception:
                                    pass
                                # N join arrivals collapse into one executable token.
                                # Keep one outstanding work item for the released join step.
                                _work_done(collapsed_work_items)
                            if released is not None:
                                mask, token_id, parent_token_id = released
                            _persist_rt_join_runtime()

                        inflight_tokens.add(
                            (str(nid), int(mask), str(token_id), parent_token_id)
                        )
                        _persist_rt_join_runtime()
                        import contextvars

                        ctx = contextvars.copy_context()
                        inflight[(nid, mask, str(token_id))] = pool.submit(
                            ctx.run,
                            worker,
                            nid,
                            state,
                            token_id,
                            parent_token_id,
                            step_seq,
                            mask,
                            cache_dir=cache_dir,
                        )

                    if done_q.empty():
                        with work_cond:
                            work_cond.wait_for(
                                lambda: cancel_pending
                                or (not done_q.empty())
                                or _work_snapshot() == 0
                                or (
                                    run_suspended
                                    and not inflight
                                    and scheduled_q.empty()
                                ),
                                timeout=0.05,
                            )
                    if done_q.empty():
                        continue
                    done_task = done_q.get_nowait()
                    (
                        node_id,
                        run_result,
                        dur_ms,
                        token_id,
                        parent_token_id,
                        status,
                        mask,
                    ) = done_task
                    if status == "failure" and isinstance(run_result, RunFailure):
                        run_errors.extend(
                            str(error) for error in run_result.errors
                        )
                    graveyard.put_nowait(done_task)
                    run_state_lock: Lock = self.state_lock[str(run_id)]
                    with self._maybe_step_uow():
                        with run_state_lock:
                            self.apply_state_update(
                                mute_state=state,
                                state_update=run_result.state_update,
                                update=run_result.update,
                            )

                    budget_ledger = budget_ledger_from_state(state)
                    if budget_ledger is not None:
                        step_attribution = BudgetAttribution(
                            workspace_id=str(state.get("workspace_id") or "") or None,
                            source_document_id=str(
                                state.get("source_document_id") or state.get("document_id") or ""
                            )
                            or None,
                            operation_id=str(stable_id("kogwistar.runtime.operation", run_id, node_id)),
                            operation_kind=f"workflow_step:{node_id}",
                        )
                        try:
                            budget_ledger.debit_step(
                                run_id=str(run_id),
                                reason=f"step:{node_id}",
                                attribution=step_attribution,
                            )
                        except Exception:
                            pass
                        try:
                            budget_ledger.debit_time(
                                max(0, dur_ms),
                                reason=f"step:{node_id}",
                                run_id=str(run_id),
                                attribution=step_attribution,
                            )
                        except Exception:
                            # A slice may be exceeded by one in-flight step;
                            # retain the exhausted marker so the runtime parks
                            # the continuation instead of silently continuing.
                            if budget_ledger.time_budget_ms:
                                budget_ledger.state["time_used_ms"] = budget_ledger.time_budget_ms
                        try:
                            result_payload = dict(
                                run_result.model_dump(exclude_none=True)
                            )
                        except Exception:
                            result_payload = {}
                        usage_payload = None
                        if isinstance(result_payload.get("usage"), dict):
                            usage_payload = {"usage": result_payload.get("usage")}
                        elif isinstance(result_payload.get("usage_metadata"), dict):
                            usage_payload = {
                                "usage": result_payload.get("usage_metadata")
                            }
                        if usage_payload is not None:
                            try:
                                budget_ledger.debit_call(
                                    run_id=str(run_id),
                                    reason=f"llm_call:{node_id}",
                                    attribution=step_attribution,
                                )
                            except Exception:
                                pass
                            for evt in adapt_budget_events(
                                usage_payload,
                                run_id=str(run_id),
                                scope="step",
                                attribution=step_attribution,
                            ):
                                try:
                                    budget_ledger.ingest(evt)
                                except Exception:
                                    pass
                        try:
                            if budget_ledger.should_suspend_for_budget():
                                budget_state = state.get("budget")
                                if isinstance(budget_state, dict):
                                    ready_ms = budget_ledger.rate_window_ready_ms
                                    if ready_ms is not None:
                                        budget_state["budget_wait_until_ms"] = int(
                                            ready_ms
                                        )
                                        budget_state["budget_wait_reason"] = "rate_window"
                                state["wait_reason"] = "rate_window"
                                status = "suspended"
                                run_suspended = True
                        except Exception:
                            pass

                    inflight.pop((node_id, mask, str(token_id)), None)
                    inflight_tokens.discard(
                        (str(node_id), int(mask), str(token_id), parent_token_id)
                    )
                    _persist_rt_join_runtime()

                    wn = nodes[node_id]

                    # If suspended, mark the run as suspended and persist step
                    if status == "suspended":
                        run_suspended = True
                        suspended_tokens[str(token_id)] = (
                            str(node_id),
                            int(mask),
                            str(token_id),
                            parent_token_id,
                        )
                        # Do not enqueue next tokens for this node
                    else:
                        suspended_tokens.pop(str(token_id), None)

                    # persist step exec trace node (same transaction as state update when possible)
                    with self._maybe_step_uow():
                        last_exec_node = self._persist_step_exec(
                            conversation_id=str(conversation_id or ""),
                            workflow_id=workflow_id,
                            run_id=run_id,
                            step_seq=step_seq,
                            workflow_node_id=node_id,
                            op=wn.op,
                            status=status,
                            duration_ms=dur_ms,
                            result=run_result,
                            state=state,
                            token_id=str(token_id),
                            parent_token_id=(
                                str(parent_token_id)
                                if parent_token_id is not None
                                else None
                            ),
                            join_mask=int(mask),
                            last_exec_node=last_exec_node,
                        )

                    # single-writer state apply:
                    # default behavior: store under result.<op> and allow ops to also write into state
                    # state[f"result.{wn.op}"] = result result only visible

                    # NOTE: checkpoint must be persisted after routing/enqueueing next tokens so that
                    # the state snapshot contains a correct _rt_join frontier (pending/inflight) for resume.
                    step_seq_current = step_seq
                    step_seq += 1

                    def _checkpoint_current_step() -> None:
                        nonlocal ownership
                        ownership = self._renew_workflow_run(
                            conversation_id=str(conversation_id or ""),
                            run_id=str(run_id),
                            ownership=ownership,
                        )
                        if (step_seq_current % self.checkpoint_every_n_steps) == 0:
                            with self._maybe_step_uow():
                                self._persist_checkpoint(
                                    conversation_id=str(conversation_id or ""),
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    step_seq=step_seq_current,
                                    state=state,
                                    last_exec_node=last_exec_node,
                                    trace_context=run_trace_context,
                                )
                            try:
                                tc_ck = _lifecycle_trace_context(
                                    token_id=str(token_id),
                                    step_seq=int(step_seq_current),
                                    node_id=str(node_id),
                                )
                                self.emitter.emit(
                                    type="checkpoint_saved",
                                    ctx=tc_ck,
                                    payload={"step_seq": int(step_seq_current)},
                                )
                            except Exception:
                                pass

                    if status == "suspended":
                        # When suspended, the token is parked here pending external resume.
                        # We DO NOT decrement the join obligations because the token still exists
                        # and will eventually reach downstream joins once resumed.
                        wait_reason = (
                            run_result.wait_reason
                            if isinstance(run_result, RunSuspended)
                            else None
                        )
                        if wait_reason:
                            state["wait_reason"] = str(wait_reason)
                        _persist_rt_join_runtime()
                        _checkpoint_current_step()
                        # Token is parked, not active work anymore.
                        _work_done(1)

                        # Emit specific suspension payload for client ingestion
                        if isinstance(run_result, RunSuspended):
                            try:
                                self.emitter.emit(
                                    type="workflow_step_suspended",
                                    ctx=_lifecycle_trace_context(
                                        token_id=str(token_id),
                                        step_seq=int(step_seq_current),
                                        node_id=str(node_id),
                                    ),
                                    payload=run_result.resume_payload,
                                )
                            except Exception:
                                pass
                        continue

                    if _enter_cancelling(
                        reason_node_id=str(node_id), token_id=str(token_id)
                    ):
                        # The current token completed after a cancel request was accepted.
                        # Stop it here so no downstream work is scheduled during cancellation.
                        _dec(mask)
                        _persist_rt_join_runtime()
                        _checkpoint_current_step()
                        _work_done(1)
                        continue

                    # route
                    if wn.terminal or len(adj.get(node_id, [])) == 0:
                        # token ends here -> it will never reach any remaining joins
                        if status == "failure":
                            _enter_failed()
                        _dec(mask)
                        _persist_rt_join_runtime()
                        _checkpoint_current_step()
                        _enter_cancelling(
                            reason_node_id=str(node_id), token_id=str(token_id)
                        )
                        _work_done(1)
                        continue

                    edges = adj.get(node_id, [])
                    next_nodes, route_decision = self._route_next(
                        edges, state, run_result, wn.fanout, nodes
                    )

                    # Trace routing decision (includes rejections) at orchestration choke point
                    try:
                        tc = _lifecycle_trace_context(
                            token_id=str(token_id),
                            step_seq=int(step_seq_current),
                            node_id=str(node_id),
                        )
                        self.emitter.emit(
                            type="routing_decision",
                            ctx=tc,
                            payload={
                                "evaluated": [
                                    (name, bool(value))
                                    for name, value in route_decision.evaluated
                                ],
                                "selected": list(route_decision.selected),
                                "next_nodes": [str(x) for x in (next_nodes or [])],
                            },
                        )
                        # Also emit per-evaluation/per-selection events for telemetry consumers
                        for pred_name, ok in route_decision.evaluated:
                            try:
                                self.emitter.predicate_evaluated(
                                    tc, predicate=str(pred_name), value=bool(ok)
                                )
                            except Exception:
                                pass
                        for edge_id, to_node_id, reason in route_decision.selected:
                            try:
                                self.emitter.edge_selected(
                                    tc,
                                    edge_id=str(edge_id),
                                    to_node_id=str(to_node_id),
                                    reason=str(reason),
                                )
                            except Exception:
                                pass

                        # Tracing must never break execution
                    except Exception:
                        pass

                    if status == "failure":
                        handled_failure = any(
                            reason in {"predicate", "explicit"}
                            for _edge_id, _to_node_id, reason in route_decision.selected
                        )
                        if not handled_failure:
                            next_nodes = []

                    if run_failed:
                        _dec(mask)
                        _persist_rt_join_runtime()
                        _checkpoint_current_step()
                        _work_done(1)
                        continue

                    if not next_nodes:
                        if status == "failure":
                            _enter_failed()
                        _dec(mask)
                        _persist_rt_join_runtime()
                        ownership = self._renew_workflow_run(
                            conversation_id=str(conversation_id or ""),
                            run_id=str(run_id),
                            ownership=ownership,
                        )

                        if (step_seq_current % self.checkpoint_every_n_steps) == 0:
                            with self._maybe_step_uow():
                                self._persist_checkpoint(
                                    conversation_id=str(conversation_id or ""),
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    step_seq=step_seq_current,
                                    state=state,
                                    last_exec_node=last_exec_node,
                                    trace_context=run_trace_context,
                                )
                            try:
                                tc_ck = _lifecycle_trace_context(
                                    token_id=str(token_id),
                                    step_seq=int(step_seq_current),
                                    node_id=str(node_id),
                                )

                                self.emitter.emit(
                                    type="checkpoint_saved",
                                    ctx=tc_ck,
                                    payload={"step_seq": int(step_seq_current)},
                                )

                            except Exception:
                                pass
                        _enter_cancelling(
                            reason_node_id=str(node_id), token_id=str(token_id)
                        )
                        _work_done(1)
                        continue

                    from kogwistar._rust_bridge import (
                        runtime_implementation_mode,
                        runtime_plan_successors,
                    )

                    runtime_mode = runtime_implementation_mode()
                    successor_payload = {
                                "token_id": str(token_id),
                                "parent_token_id": parent_token_id,
                                "step_seq": int(step_seq_current),
                                "current_join_mask": int(mask),
                                "join_outstanding": list(_join_outstanding),
                                "successors": [
                                    {
                                        "node_id": str(nxt),
                                        "join_mask": int(_may_reach_join.get(str(nxt), 0)),
                                    }
                                    for nxt in next_nodes
                                ],
                            }
                    native_plan = None
                    if runtime_mode == "shadow":
                        oracle_outstanding = list(_join_outstanding)
                        oracle_tokens = []
                        for index, successor in enumerate(successor_payload["successors"]):
                            successor_mask = int(successor["join_mask"])
                            if index == 0:
                                for bit in _iter_bits(int(mask) & ~successor_mask):
                                    oracle_outstanding[bit] = max(
                                        0, oracle_outstanding[bit] - 1
                                    )
                                for bit in _iter_bits(successor_mask & ~int(mask)):
                                    oracle_outstanding[bit] += 1
                                oracle_token = str(token_id)
                                oracle_parent = parent_token_id
                            else:
                                for bit in _iter_bits(successor_mask):
                                    oracle_outstanding[bit] += 1
                                oracle_token = "<spawned>"
                                oracle_parent = str(token_id)
                            oracle_tokens.append(
                                {
                                    "node_id": str(successor["node_id"]),
                                    "join_mask": successor_mask,
                                    "token_id": oracle_token,
                                    "parent_token_id": oracle_parent,
                                    "spawned": index > 0,
                                }
                            )
                        runtime_plan_successors(
                            payload=successor_payload,
                            python_value=cast(
                                JsonObject,
                                {
                                    "tokens": cast(Json, oracle_tokens),
                                    "join_outstanding": cast(
                                        Json, oracle_outstanding
                                    ),
                                },
                            ),
                        )
                    elif runtime_mode == "rust":
                        native_plan = runtime_plan_successors(payload=successor_payload)
                        native_tokens, native_join_outstanding = successor_plan(native_plan)
                        _join_outstanding[:] = native_join_outstanding
                        for planned in native_tokens:
                            nxt = str(planned["node_id"])
                            nxt_mask = int(planned["join_mask"])
                            planned_token_id = str(planned["token_id"])
                            planned_parent_id = planned.get("parent_token_id")
                            t = (nxt, nxt_mask, planned_token_id, planned_parent_id)
                            if planned["spawned"]:
                                try:
                                    mq.put_nowait(
                                        {
                                            "type": "token.spawn",
                                            "parent_token_id": str(token_id),
                                            "child_token_id": planned_token_id,
                                            "from_node_id": node_id,
                                            "to_node_id": nxt,
                                        }
                                    )
                                except queue.Full:
                                    pass
                                try:
                                    self.emitter.emit(
                                        type="token_spawned",
                                        ctx=_lifecycle_trace_context(
                                            run_id=str(run_id),
                                            token_id=str(token_id),
                                            step_seq=int(step_seq_current),
                                            node_id=str(node_id),
                                            attempt=1,
                                            conversation_id=(
                                                str(conversation_id)
                                                if conversation_id is not None
                                                else None
                                            ),
                                            turn_node_id=(
                                                str(turn_node_id)
                                                if turn_node_id is not None
                                                else None
                                            ),
                                        ),
                                        payload={
                                            "parent_token_id": str(token_id),
                                            "child_token_id": planned_token_id,
                                            "to_node_id": nxt,
                                        },
                                    )
                                except Exception:
                                    pass
                            pending_tokens.add(t)
                            _work_add(1)
                            scheduled_q.put(t)
                            _persist_rt_join_runtime()

                    # Python fallback continuation/fan-out planner.
                    first = True
                    for i_fanout, nxt in enumerate(next_nodes if native_plan is None else []):
                        nxt = str(nxt)
                        nxt_mask = int(_may_reach_join.get(nxt, 0))

                        if first:
                            # continuation keeps token_id
                            t = (
                                str(nxt),
                                int(nxt_mask),
                                str(token_id),
                                parent_token_id,
                            )

                            leaving = mask & ~nxt_mask
                            if leaving:
                                _dec(leaving)
                                _persist_rt_join_runtime()

                            gained = nxt_mask & ~mask
                            if gained:
                                _inc(gained)
                                _persist_rt_join_runtime()

                            pending_tokens.add(t)
                            _work_add(1)
                            scheduled_q.put(t)
                            _persist_rt_join_runtime()
                            first = False
                        else:
                            # fanout child gets a NEW token_id
                            child_token_id = stable_id(
                                "token_id",
                                f"{token_id}/{step_seq_current}:{i_fanout}:{nxt}",
                            ).hex  # uuid.uuid4().hex
                            t = (str(nxt), int(nxt_mask), child_token_id, str(token_id))

                            try:
                                mq.put_nowait(
                                    {
                                        "type": "token.spawn",
                                        "parent_token_id": str(token_id),
                                        "child_token_id": child_token_id,
                                        "from_node_id": node_id,
                                        "to_node_id": str(nxt),
                                    }
                                )
                            except queue.Full:
                                pass
                            # trace token spawn
                            try:
                                tc_spawn = _lifecycle_trace_context(
                                    run_id=str(run_id),
                                    token_id=str(token_id),
                                    step_seq=int(step_seq_current),
                                    node_id=str(node_id),
                                    attempt=1,
                                    conversation_id=str(conversation_id)
                                    if conversation_id is not None
                                    else None,
                                    turn_node_id=str(turn_node_id)
                                    if turn_node_id is not None
                                    else None,
                                )
                                self.emitter.emit(
                                    type="token_spawned",
                                    ctx=tc_spawn,
                                    payload={
                                        "parent_token_id": str(token_id),
                                        "child_token_id": str(child_token_id),
                                        "to_node_id": str(nxt),
                                    },
                                )
                            except Exception:
                                pass

                            # fanout creates a new token (new outstanding obligations)
                            _inc(nxt_mask)

                            pending_tokens.add(t)
                            _work_add(1)
                            scheduled_q.put(t)
                            _persist_rt_join_runtime()

                    _checkpoint_current_step()
                    _enter_cancelling(
                        reason_node_id=str(node_id), token_id=str(token_id)
                    )
                    _work_done(1)
            raise Exception("unreacheable")
