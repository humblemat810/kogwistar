from __future__ import annotations
import warnings

import time
import uuid
import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, cast
from concurrent.futures import ThreadPoolExecutor
import pathlib
import logging
from contextlib import nullcontext

from graph_knowledge_engine.id_provider import stable_id
from ..conversation_orchestrator import get_id_for_conversation_turn_edge
from graph_knowledge_engine.models import WorkflowNode, MentionVerification, ConversationEdge, WorkflowEdge, WorkflowRunNode

from .design import validate_workflow_design, Predicate
from .serialize import try_serialize_with_ref
RESERVED_ROOT_KEYS = {
    "_deps",
    "_rt_join",
}

RESERVED_PREFIXES = ("_", "__")

def validate_initial_state(initial_state: dict):
    """Validate user-provided initial workflow state.

    Workflow state is user-land *except* for a small set of underscore-prefixed
    keys that are reserved for runtime/DI plumbing.

    Allowed underscore keys:
      - _deps    : injected dependencies (non-serializable; must not be checkpointed)
      - _rt_join : runtime-owned join/barrier bookkeeping (checkpoint/resume)

    All other keys starting with '_' are reserved and will be rejected.

    Note: when underscore keys are present, we emit a RuntimeWarning to make the
    use of advanced/internal features explicit (helps debugging).
    """
    allowed_underscore = {"_deps", "_rt_join"}

    for key in initial_state:
        if key in allowed_underscore:
            warnings.warn(
                f"Using advanced underscore state key '{key}'. This key is reserved for runtime/DI plumbing.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if key in RESERVED_ROOT_KEYS:
            raise ValueError(
                f"'{key}' is reserved by the runtime and cannot be provided by user code."
            )

        if key.startswith(RESERVED_PREFIXES):
            raise ValueError(
                f"Keys starting with '_' or '__' are reserved. "
                f"Invalid key: '{key}'"
            )
# ------------------------------------------------------------------
# Join/barrier support (capability tracking via MAY-reach bitsets)
# ------------------------------------------------------------------

def _tarjan_scc(n: int, succ: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """
    Tarjan SCC.
    Returns:
      comp_of[v] -> component id
      comps -> list of components (each is list of vertices)
    """
    index = 0
    stack: list[int] = []
    onstack = [False] * n
    idxs = [-1] * n
    low = [0] * n
    comp_of = [-1] * n
    comps: list[list[int]] = []

    def strongconnect(v: int) -> None:
        nonlocal index
        idxs[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack[v] = True

        for w in succ[v]:
            if idxs[w] == -1:
                strongconnect(w)
                low[v] = low[v] if low[v] < low[w] else low[w]
            elif onstack[w]:
                low[v] = low[v] if low[v] < idxs[w] else idxs[w]

        if low[v] == idxs[v]:
            comp_id = len(comps)
            comp: list[int] = []
            while True:
                w = stack.pop()
                onstack[w] = False
                comp_of[w] = comp_id
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in range(n):
        if idxs[v] == -1:
            strongconnect(v)

    return comp_of, comps


def _compute_may_reach_join_bitsets(
    *,
    node_ids: list[str],
    adj: dict[str, list["WorkflowEdge"]],
    join_ids: list[str],
) -> dict[str, int]:
    """
    Over-approximate: for every node, compute the set of join/barrier nodes it MAY reach,
    ignoring predicates (static topology only).

    Output is a dict node_id -> bitset (int) where bit i means MAY reach join_ids[i].
    """
    n = len(node_ids)
    id_to_i = {nid: i for i, nid in enumerate(node_ids)}
    succ: list[list[int]] = [[] for _ in range(n)]
    for src, edges in adj.items():
        si = id_to_i.get(src)
        if si is None:
            continue
        for e in edges:
            for dst in getattr(e, "target_ids", []) or []:
                di = id_to_i.get(str(dst))
                if di is not None:
                    succ[si].append(di)

    comp_of, comps = _tarjan_scc(n, succ)
    c = len(comps)

    # component join mask
    join_pos = {jid: p for p, jid in enumerate(join_ids)}
    comp_join_mask = [0] * c
    for v_nid in node_ids:
        p = join_pos.get(v_nid)
        if p is None:
            continue
        v = id_to_i[v_nid]
        comp_join_mask[comp_of[v]] |= (1 << p)

    # condensation DAG
    comp_succ: list[set[int]] = [set() for _ in range(c)]
    indeg = [0] * c
    for v in range(n):
        cv = comp_of[v]
        for w in succ[v]:
            cw = comp_of[w]
            if cv != cw and cw not in comp_succ[cv]:
                comp_succ[cv].add(cw)
                indeg[cw] += 1

    # topo order (Kahn)
    q = [i for i in range(c) if indeg[i] == 0]
    topo: list[int] = []
    while q:
        x = q.pop()
        topo.append(x)
        for y in comp_succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)

    # dp in reverse topo: may_reach[comp] = comp_join | OR succ
    may = comp_join_mask[:]
    for x in reversed(topo):
        m = may[x]
        for y in comp_succ[x]:
            m |= may[y]
        may[x] = m
    out: dict[str, int] = {}
    for nid in node_ids:
        out[nid] = may[comp_of[id_to_i[nid]]]
    return out


def _iter_bits(mask: int):
    """Yield bit positions (0-based) for an int bitset."""
    # 2's complement tricks to get the first set least significant bit in binary representation
    while mask:
        lsb = mask & -mask
        yield (lsb.bit_length() - 1)
        mask ^= lsb


RunID= uuid.UUID | str
Json = Any
State = Dict[str, Json]
# Result = Json
from typing import TypedDict, Literal, TypeAlias, Any, Type, Literal, Union
from .contract import WorkflowEdgeInfo
from dataclasses import dataclass
from pydantic import BaseModel

StateAppendUpdate = tuple[Literal['u'], Any]
StateOverwriteUpdate = tuple[Literal['a'], Any]
StateUpdate = Union[StateAppendUpdate , StateOverwriteUpdate]
from graph_knowledge_engine.models import WorkflowStepExecNode, Grounding, Span
    

class RunSuccess(BaseModel):
    conversation_node_id: str|None  # node id of the 'entry point' of the node cluster created in a resolver step, 
    #step can create multiple node edges but at least should expose a node to connect to the main node net
    state_update: list[StateUpdate]
    # Optional native update dict (schema-driven). This does NOT replace state_update.
    # When present, WorkflowRuntime.run() applies it using state_schema and then
    # falls back unknown keys into DSL ('u') overwrite semantics.
    update: dict[str, Any] | None = None
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["success"] = "success"

class RunFailure(BaseModel):
    conversation_node_id: Optional[str]
    state_update: list[StateUpdate] # can still update, append an error message
    update: dict[str, Any] | None = None
    errors: list[str]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["failure"] = "failure"
StepRunResult: TypeAlias = RunSuccess | RunFailure
StepFn: TypeAlias = Callable[["StepContext"], StepRunResult]
from ..conversation_state_contracts import WorkflowState

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, TypedDict, Iterator, TypeVar
from types import MappingProxyType
import threading
import queue

@dataclass
class RunResult():
    run_id: str
    final_state: WorkflowState
    mq: queue.Queue[Dict[str, Json]]

class _StateWriteTxn:
    def __init__(self, ctx: "StepContext"):
        self._ctx = ctx

    def __enter__(self) -> WorkflowState:
        self._ctx._state_lock.acquire()
        # optional:
        # self._ctx.publish({"type": "state_write_start", "op": self._ctx.op})
        return self._ctx._state

    def __exit__(self, exc_type, exc, tb) -> None:
        # optional:
        # self._ctx.publish({"type": "state_write_end", "op": self._ctx.op, "ok": exc is None})
        self._ctx._state_lock.release()

from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Mapping
from types import MappingProxyType
import threading, queue

from .telemetry import TraceContext, SQLiteEventSink, EventEmitter, bind_logger, BoundLoggerAdapter

@dataclass
class RouteDecision:
    """Structured routing outcome used for trace emission.

    evaluated: list of (edge_id_with_predicate, ok) entries for every predicate/base evaluation attempted.
    selected:  list of (edge_id, to_node_id, reason) entries for chosen edges.
    """
    evaluated: List[Tuple[str, bool]]
    selected: List[Tuple[str, str, str]]


@dataclass
class StepContext:
    # --- execution identity ---
    run_id: str
    workflow_id: str
    workflow_node_id: str
    op: str
    token_id: str
    attempt: int
    step_seq: int

    # --- optional provenance (may be None for non-conversational workloads) ---
    conversation_id: str | None = None
    turn_node_id: str | None = None

    # --- runtime capabilities ---
    message_queue: "queue.Queue[Dict[str, Json]]" = field(repr=False, default_factory=queue.Queue)
    events: EventEmitter | None = field(repr=False, default=None)

    # Accept `state=` in __init__ but don't store it as a field
    state: InitVar["WorkflowState"] = None  # type: ignore[assignment]

    # real storage
    _state: "WorkflowState" = field(init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    log: BoundLoggerAdapter = field(init=False, repr=False)

    def __post_init__(self, state: "WorkflowState") -> None:
        self._state = state
        # bind a correlated logger for resolver-level details
        self.log = bind_logger(logging.getLogger("workflow.resolver"), self.trace_ctx)

    @property
    def state_view(self) -> Mapping[str, Json]:
        return MappingProxyType(self._state)

    @property
    def state_write(self) -> "_StateWriteTxn":
        return _StateWriteTxn(self)

    def publish(self, msg: Dict[str, Json]) -> None:
        self.message_queue.put(msg)

    def drain(self, max_items: int = 200) -> List[Dict[str, Json]]:
        raise Exception("current design does not allow mq drained in context, but only by orchestrator")

    @property
    def trace_ctx(self) -> TraceContext:
        # generic trace context; conversation_id/turn_node_id may be None
        return TraceContext(
            run_id=self.run_id,
            token_id=self.token_id,
            step_seq=self.step_seq,
            node_id=self.workflow_node_id,
            attempt=self.attempt,
            conversation_id=self.conversation_id,
            turn_node_id=self.turn_node_id,
        )

    # Back-compat alias: older code used conv_trace_ctx
    @property
    def conv_trace_ctx(self) -> TraceContext:
        return self.trace_ctx


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_trace_span(*, conversation_id: str, excerpt: str, doc_id: str) -> Any:
    # We avoid importing models here to keep the runtime module lightweight.
    # The orchestrator will pass in a ready Span via a hook, OR we can create a minimal span-like dict.
    # In your repo, you already have Span model and Grounding/MentionVerification.
    return {
        "collection_page_url": f"conversation/{conversation_id}",
        "document_page_url": f"conversation/{conversation_id}",
        "doc_id": doc_id,
        "insertion_method": "workflow_trace",
        "page_number": 1,
        "start_char": 0,
        "end_char": len(excerpt),
        "excerpt": excerpt,
        "context_before": "",
        "context_after": "",
        "chunk_id": None,
        "source_cluster_id": None,
        "verification": {
            "method": "human",
            "is_verified": True,
            "score": 1.0,
            "notes": "workflow trace",
        },
    }
from threading import Lock
class WorkflowRuntime:
    """
    Core engine for executing graph-based **workflow designs**.

    This runtime:
    1.  Reads the workflow design (nodes/edges) from the `workflow_engine`.
    2.  Executes steps using the provided `step_resolver`.
    3.  Manages state transitions, branching, and join/barrier logic.
    4.  Persists execution traces (`WorkflowRunNode`, `WorkflowStepExecNode`) and checkpoints
        to the `conversation_engine` (or whichever engine is designated for traces).

    It is the backend execution engine used by high-level orchestrators like `ConversationOrchestrator`.
    """

    def __init__(
        self,
        *,
        workflow_engine: Any,
        conversation_engine: Any,
        step_resolver: Callable[[str], StepFn],
        predicate_registry: Dict[str, Predicate],
        checkpoint_every_n_steps: int = 1,
        max_workers: int = 4,
        transaction_mode: str | None = None,  # "step" | "run" | "none" (default auto)
    ) -> None:
        from graph_knowledge_engine.engine import GraphKnowledgeEngine
        self.workflow_engine: GraphKnowledgeEngine = workflow_engine
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.step_resolver: Callable[[str], Callable[..., StepRunResult]] = step_resolver
        self.predicate_registry = predicate_registry
        self.checkpoint_every_n_steps = max(1, int(checkpoint_every_n_steps))
        self.max_workers = max_workers
        # Transaction policy: default to per-step transactions when conversation_engine is Postgres-backed.
        if transaction_mode is None:
            try:
                from graph_knowledge_engine.postgres_backend import PgVectorBackend

                self.transaction_mode = "step" if isinstance(getattr(conversation_engine, "backend", None), PgVectorBackend) else "none"
            except Exception:
                self.transaction_mode = "none"
        else:
            self.transaction_mode = str(transaction_mode)
        self.state_lock : dict[RunID, Lock] = {} # look up of run specific state lock
        # somewhere in your runtime init (or run())
        if getattr(workflow_engine, "persist_directory", None) is not None:
            self.sink = SQLiteEventSink(
                db_path=str(pathlib.Path(workflow_engine.persist_directory) / "wf_trace.sqlite"),
                drop_when_full=True,  # trace can be best-effort; set False if you want backpressure
            )
        else:
            self.sink = None
        self.emitter = EventEmitter(sink=self.sink, logger=logging.getLogger("workflow.trace"))

    def _maybe_step_uow(self):
        """Open a UoW transaction only when transaction_mode=='step'.

        This keeps callsites clean:
            with self._maybe_step_uow():
                ...
        """
        if self.transaction_mode == "step":
            return self.conversation_engine.uow()
        return nullcontext()

    def apply_state_update(self, mute_state: WorkflowState, state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate]):
        # inplace update state
        for update_item in state_update:
            update_item: tuple[str, dict[str, Any]] | StateUpdate
            if update_item[0] == 'a': # append
                append_dict: dict = update_item[1]
                for k, v in append_dict.items():
                    mute_state.setdefault(k, []).append(v)
            elif update_item[0] == 'u': # update by overwrite
                update_dict: dict = update_item[1]
                for k, v in update_dict.items():
                    mute_state[k]=v
            elif update_item[0] == 'e': # update by extending list
                update_dict: dict = update_item[1]
                for k, v in update_dict.items():
                    mute_state.setdefault(k, []).extend(v)
            
    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """
        Returns (final_state, run_id).

        Design lives in workflow_engine.
        Traces/checkpoints live in conversation_engine.
        """
        validate_initial_state(initial_state)
        
        run_id = run_id or f"run|{uuid.uuid4()}"
        self.state_lock[str(run_id)] = Lock()
        
        # an interworker, orchestrator message channel
        mq: queue.Queue[Dict[str, Json]] = queue.Queue(maxsize = 10000)

        start, nodes, adj = validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
        )

        # ----------------------------
        # Precompute MAY-reach join/barrier sets (ignoring predicates)
        # ----------------------------
        node_ids_list = list(nodes.keys())
        join_node_ids: list[str] = []
        _join_is_merge = {}
        for _nid, _wn in nodes.items():
            _md = getattr(_wn, "metadata", None) or {}
            if bool(_md.get("wf_join", False)) or str(getattr(_wn, "op", "")) == "join":
                join_node_ids.append(str(_nid))
                _join_is_merge[str(_nid)] = bool(_md.get('wf_join_is_merge') if _md.get('wf_join_is_merge') is not None else True)
        _may_reach_join = (
            _compute_may_reach_join_bitsets(node_ids=node_ids_list, adj=adj, join_ids=join_node_ids)
            if join_node_ids
            else {nid: 0 for nid in node_ids_list}
        )
        _join_pos = {jid: i for i, jid in enumerate(join_node_ids)}
        _join_outstanding: list[int] = [0 for _ in join_node_ids]  # tokens BEFORE each join
        _join_waiters: dict[str, list[int]] = {jid: [] for jid in join_node_ids}  # store per-token mask at join
        
        def _inc(mask: int) -> None:
            for bi in _iter_bits(mask):
                _join_outstanding[bi] += 1

        def _dec(mask: int) -> None:
            for bi in _iter_bits(mask):
                _join_outstanding[bi] -= 1
                if _join_outstanding[bi] < 0:
                    # defensive: never go negative
                    _join_outstanding[bi] = 0

        def _mask_without_join(mask: int, join_id: str) -> int:
            bi = _join_pos.get(join_id)
            if bi is None:
                return mask
            return mask & ~(1 << bi)

        def _bit_for_join(join_id: str) -> int:
            bi = _join_pos.get(join_id)
            return (1 << bi) if bi is not None else 0
        def _rt_join_get() -> dict:
            return cast(dict, state.get("_rt_join", {}))

        def _rt_join_set(payload: dict) -> None:
            # Stored in workflow state so checkpoints can resume joins.
            state["_rt_join"] = payload

        def _rt_join_snapshot(pending: list[tuple[str, int, str, str | None]]) -> dict:
            # Persist inflight as pending-on-resume by including them in pending list upstream.
            return {
                "join_node_ids": join_node_ids,
                "join_outstanding": list(_join_outstanding),
                "join_waiters": {jid: list(masks) for jid, masks in _join_waiters.items()},
                "pending": [(nid, int(mask), str(token_id), (str(parent_token_id) if parent_token_id is not None else None)) for nid, mask, token_id, parent_token_id in pending],
            }

        def _rt_join_restore() -> list[tuple[str, int, str, str | None]] | None:
            payload = _rt_join_get()
            if not payload:
                return None
            if payload.get("join_node_ids") != join_node_ids:
                # Design changed; discard stale join runtime.
                return None
            jo = payload.get("join_outstanding")
            jw = payload.get("join_waiters")
            pend = payload.get("pending")
            if not isinstance(jo, list) or not isinstance(jw, dict) or not isinstance(pend, list):
                return None
            # restore counters/waiters
            for i in range(min(len(_join_outstanding), len(jo))):
                try:
                    _join_outstanding[i] = int(jo[i])
                except Exception:
                    _join_outstanding[i] = 0
            for jid in join_node_ids:
                masks = jw.get(jid, [])
                if isinstance(masks, list):
                    _join_waiters[jid] = [int(x) for x in masks]
            # pending tokens
            out: list[tuple[str, int, str, str | None]] = []
            for item in pend:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    out.append((str(item[0]), int(item[1]), str(item[2]), (str(item[3]) if item[3] is not None else None)))
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    # backward-compat: snapshots without parent_token_id
                    out.append((str(item[0]), int(item[1]), str(item[2]), None))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    # backward-compat: older snapshots without token_id/parent_token_id
                    out.append((str(item[0]), int(item[1]), uuid.uuid4().hex, None))
            return out
            
        # start, nodes, adj = load_workflow_design(workflow_engine=self.workflow_engine, workflow_id=workflow_id)

        state: WorkflowState = initial_state
        step_seq = 0

        # Persist workflow_run node in conversation_engine
        wf_run_root_node = self._persist_workflow_run(
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            run_id=run_id,
            turn_node_id=turn_node_id,
            status="running",
        )
        # trace: workflow run started
        try:
            tc_run = TraceContext(
                run_id=str(run_id),
                token_id=str(run_id),  # root token id is run_id for now
                step_seq=0,
                node_id=str(start_id) if 'start_id' in locals() else "start",
                attempt=1,
                conversation_id=str(conversation_id) if conversation_id is not None else None,
                turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
            )
            self.emitter.emit(type="workflow_run_started", ctx=tc_run, payload={"workflow_id": str(workflow_id)})
        except Exception:
            pass
        state_mutex_lock: threading.Lock = threading.Lock()
        scheduled_q: queue.Queue[Tuple[str, int, str, str | None]] = queue.Queue()  # (node_id, mask, token_id, parent_token_id)
        pending_tokens: set[tuple[str, int, str, str | None]] = set()
        inflight_tokens: set[tuple[str, int, str, str | None]] = set()
        done_q: queue.Queue[Tuple[str, StepRunResult, int, str, str | None, str, int]] = queue.Queue()  # (node_id, result, duration_ms, token_id, parent_token_id, status, mask)
        
        graveyard: queue.Queue[Tuple[str, StepRunResult, int, str, str | None, str, int]] = queue.Queue()  
        def _persist_rt_join_runtime() -> None:
            # Persist both pending + inflight as pending-on-resume (idempotent).
            combined = sorted(pending_tokens.union(inflight_tokens))
            _rt_join_set(_rt_join_snapshot(combined))
        start_id = start.safe_get_id()

        # Resume-safe join runtime: if a checkpoint restored pending tokens + join counters,
        # seed the scheduler from that payload. Otherwise start from the workflow start node.
        restored = _rt_join_restore()
        if restored:
            for nid, mask, token_id, parent_token_id in restored:
                t = (str(nid), int(mask), str(token_id), parent_token_id)
                pending_tokens.add(t)
                scheduled_q.put(t)
            _persist_rt_join_runtime()
        else:
            start_mask = int(_may_reach_join.get(start_id, 0))
            _inc(start_mask)
            t = (str(start_id), 
                 int(start_mask), 
                 run_id, #uuid.uuid4().hex, # token_id = run-id 
                 None)
            pending_tokens.add(t)
            scheduled_q.put(t)
            _persist_rt_join_runtime()

        def worker(node_id: str, state: WorkflowState, token_id: str, parent_token_id: str | None, step_seq: int, mask: int) -> None:
            wn: WorkflowNode
            wn = nodes[node_id]
            op = nodes[node_id].op
            t0 = _now_ms()
            status = "ok"
            try:
                fn: Callable[..., StepRunResult] = self.step_resolver(op)
                ctx = StepContext(
                    run_id=str(run_id),
                    workflow_id=str(workflow_id),
                    workflow_node_id=str(node_id),
                    op=str(op),
                    token_id=str(token_id),
                    attempt=1,
                    step_seq=int(step_seq),
                    conversation_id=str(conversation_id) if conversation_id is not None else None,
                    turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
                    state=state,
                    message_queue=mq,
                    events=self.emitter,
                )
                # step attempt start
                self.emitter.step_started(ctx.trace_ctx)

                # Transaction boundary (best-effort): resolver is expected to call
                # engine methods; when backed by Postgres this ensures those writes
                # are committed or rolled back atomically for this step.
                with self._maybe_step_uow():
                    res: StepRunResult = fn(ctx)
                status = "ok" if getattr(res, "status", None) != "failure" else "failure"

            except Exception as e:
                status = "error"
                res = RunFailure(conversation_node_id=None, status="failure", errors=[str(e)], state_update=[])
            try:
                mq.put_nowait(res.model_dump())
            except queue.Full as _e:
                raise
            t1 = _now_ms()
            # step attempt complete (best-effort; never break worker)
            try:
                self.emitter.step_completed(ctx.trace_ctx, status=str(status), duration_ms=max(0, t1 - t0))
            except Exception:
                pass
            done_q.put((node_id, res, max(0, t1 - t0), token_id, parent_token_id, status, mask))

        inflight: Dict[tuple[str, int, str], Any] = {}
        last_exec_node = wf_run_root_node
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while True:
                if not inflight and scheduled_q.empty() and done_q.empty():
                    # done
                    self._update_workflow_run_status(conversation_id, run_id, "done")
                    # trace: workflow run completed
                    try:
                        tc_done = TraceContext(run_id=str(run_id), token_id=str(run_id), step_seq=int(step_seq), node_id="run", attempt=1, conversation_id=str(conversation_id) if conversation_id is not None else None, turn_node_id=str(turn_node_id) if turn_node_id is not None else None)
                        self.emitter.emit(type="workflow_run_completed", ctx=tc_done, payload={"workflow_id": str(workflow_id)})
                    except Exception:
                        pass
                    self.state_lock.pop(run_id)
                    return RunResult(final_state=state, run_id=run_id, mq=mq)
                # schedule while capacity
                while len(inflight) < self.max_workers:
                    try:
                        nid, mask, token_id, parent_token_id = scheduled_q.get_nowait()
                        pending_tokens.discard((str(nid), int(mask), str(token_id), parent_token_id))
                        _persist_rt_join_runtime()
                    except queue.Empty:
                        break

                    # If this is a join/barrier node, it doesn't execute immediately.
                    if nid in _join_waiters:
                        join_bit = _bit_for_join(nid)
                        was_before_join = bool(mask & join_bit)
                        if was_before_join:

                            # token reached the join -> no longer "before" this join
                            _dec(join_bit)
                            mask = _mask_without_join(mask, nid)
                        # trace: token arrived at join
                        try:
                            join_idx = _join_pos.get(nid)
                            outstanding = int(_join_outstanding[join_idx]) if join_idx is not None else 0
                            tcj = TraceContext(
                                run_id=str(run_id),
                                token_id=str(token_id),
                                step_seq=int(step_seq),
                                node_id=str(nid),
                                attempt=1,
                                conversation_id=str(conversation_id) if conversation_id is not None else None,
                                turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
                            )
                            self.emitter.join_event(tcj, join_node_id=str(nid), kind="arrived", outstanding=outstanding)
                        except Exception:
                            pass

                        _join_waiters[nid].append(mask)
                        _persist_rt_join_runtime()
                        # trace: token waiting at join
                        try:
                            join_idx = _join_pos.get(nid)
                            outstanding = int(_join_outstanding[join_idx]) if join_idx is not None else 0
                            tcj = TraceContext(
                                run_id=str(run_id),
                                token_id=str(token_id),
                                step_seq=int(step_seq),
                                node_id=str(nid),
                                attempt=1,
                                conversation_id=str(conversation_id) if conversation_id is not None else None,
                                turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
                            )
                            self.emitter.join_event(tcj, join_node_id=str(nid), kind="waiting", outstanding=outstanding)
                        except Exception:
                            pass

                        # Release join when no more tokens are outstanding BEFORE it.
                        # We merge all waiting tokens into ONE continuation token.
                        join_idx = _join_pos.get(nid)
                        if _join_is_merge[nid]:
                            if join_idx is not None and _join_outstanding[join_idx] == 0:
                                # trace: join released (all outstanding tokens before join have arrived)
                                try:
                                    tcj = TraceContext(
                                        run_id=str(run_id),
                                        token_id=str(token_id),
                                        step_seq=int(step_seq),
                                        node_id=str(nid),
                                        attempt=1,
                                        conversation_id=str(conversation_id) if conversation_id is not None else None,
                                        turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
                                    )
                                    self.emitter.join_event(tcj, join_node_id=str(nid), kind="released", outstanding=0)
                                except Exception:
                                    pass

                                wait_masks = _join_waiters[nid]
                                _join_waiters[nid] = []

                                # remove all waiter contributions (they will be merged)
                                for wm in wait_masks:
                                    _dec(wm)

                                merged_mask = 0
                                if wait_masks:
                                    merged_mask = wait_masks[0]
                                    for wm in wait_masks[1:]:
                                        merged_mask &= wm

                                # add merged token contribution
                                _inc(merged_mask)

                                # Push a synthetic completion for the join node (noop op).
                                _persist_rt_join_runtime()
                                # join_token_id = uuid.uuid4().hex
                                # done_q.put((nid, RunSuccess(conversation_node_id=None, state_update=[]), 0, join_token_id, None, "ok", merged_mask))
                                # continue
                            else:
                                # merge into 1, if not dec to 0, not the last to join, just continue
                                continue
                        

                    inflight_tokens.add((str(nid), int(mask), str(token_id), parent_token_id))
                    _persist_rt_join_runtime()
                    inflight[(nid, mask, str(token_id))] = pool.submit(worker, nid, state, token_id, parent_token_id, step_seq, mask)


                done_task = done_q.get()
                node_id, run_result, dur_ms, token_id, parent_token_id, status, mask = done_task
                graveyard.put_nowait(done_task)
                run_state_lock: Lock = self.state_lock[str(run_id)]
                with self._maybe_step_uow():
                    with run_state_lock:
                        self.apply_state_update(mute_state=state, state_update=run_result.state_update)
                    
                inflight.pop((node_id, mask, str(token_id)), None)
                inflight_tokens.discard((str(node_id), int(mask), str(token_id), parent_token_id))
                _persist_rt_join_runtime()

                wn = nodes[node_id]

                # persist step exec trace node (same transaction as state update when possible)
                with self._maybe_step_uow():
                    last_exec_node = self._persist_step_exec(
                            conversation_id=conversation_id,
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
                            parent_token_id=(str(parent_token_id) if parent_token_id is not None else None),
                            join_mask=int(mask),
                            last_exec_node=last_exec_node
                        )

                # single-writer state apply:
                # default behavior: store under result.<op> and allow ops to also write into state
                # state[f"result.{wn.op}"] = result result only visible 

                # NOTE: checkpoint must be persisted after routing/enqueueing next tokens so that
                # the state snapshot contains a correct _rt_join frontier (pending/inflight) for resume.
                step_seq_current = step_seq
                step_seq += 1
                
                # route
                if wn.terminal or len(adj.get(node_id, [])) == 0:
                    # token ends here -> it will never reach any remaining joins
                    _dec(mask)
                    _persist_rt_join_runtime()

                    if (step_seq_current % self.checkpoint_every_n_steps) == 0:
                        with self._maybe_step_uow():
                            self._persist_checkpoint(
                                    conversation_id=conversation_id,
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    step_seq=step_seq_current,
                                    state=state,
                                    last_exec_node=last_exec_node,
                            )
                        try:

                            tc_ck = TraceContext(run_id=str(run_id), token_id=str(token_id), step_seq=int(step_seq_current), node_id=str(node_id), attempt=1, conversation_id=str(conversation_id), turn_node_id=str(turn_node_id))

                            self.emitter.emit(type="checkpoint_saved", ctx=tc_ck, payload={"step_seq": int(step_seq_current)})

                        except Exception:

                            pass
                    continue

                edges = adj.get(node_id, [])
                next_nodes, route_decision = self._route_next(edges, state, run_result, wn.fanout)

                # Trace routing decision (includes rejections) at orchestration choke point
                try:
                    tc = TraceContext(
                        run_id=str(run_id),
                        token_id=str(token_id),
                        step_seq=int(step_seq_current),
                        node_id=str(node_id),
                        attempt=1,
                        conversation_id=str(conversation_id),
                        turn_node_id=str(turn_node_id),
                    )
                    self.emitter.emit(
                        type="routing_decision",
                        ctx=tc,
                        payload={
                            "evaluated": [(a, bool(b)) for a, b in getattr(route_decision, 'evaluated', [])],
                            "selected": [(e, t, r) for e, t, r in getattr(route_decision, 'selected', [])],
                            "next_nodes": [str(x) for x in (next_nodes or [])],
                        },
                    )
                    # Also emit per-evaluation/per-selection events for telemetry consumers
                    for pred_name, ok in getattr(route_decision, "evaluated", []):
                        try:
                            self.emitter.predicate_evaluated(tc, predicate=str(pred_name), value=bool(ok))
                        except Exception:
                            pass
                    for edge_id, to_node_id, reason in getattr(route_decision, "selected", []):
                        try:
                            self.emitter.edge_selected(tc, edge_id=str(edge_id), to_node_id=str(to_node_id), reason=str(reason))
                        except Exception:
                            pass

                    # Tracing must never break execution
                except Exception:
                    pass

                if not next_nodes:
                    _dec(mask)
                    _persist_rt_join_runtime()

                    if (step_seq_current % self.checkpoint_every_n_steps) == 0:
                        with self._maybe_step_uow():
                            self._persist_checkpoint(
                                    conversation_id=conversation_id,
                                    workflow_id=workflow_id,
                                    run_id=run_id,
                                    step_seq=step_seq_current,
                                    state=state,
                                    last_exec_node=last_exec_node,
                            )
                        try:

                            tc_ck = TraceContext(run_id=str(run_id), token_id=str(token_id), step_seq=int(step_seq_current), node_id=str(node_id), attempt=1, conversation_id=str(conversation_id), turn_node_id=str(turn_node_id))

                            self.emitter.emit(type="checkpoint_saved", ctx=tc_ck, payload={"step_seq": int(step_seq_current)})

                        except Exception:

                            pass
                    continue

                # continuation token
                first = True
                for i_fanout, nxt in enumerate(next_nodes):
                    nxt = str(nxt)
                    nxt_mask = int(_may_reach_join.get(nxt, 0))

                    if first:
                        # continuation keeps token_id
                        t = (str(nxt), int(nxt_mask), str(token_id), parent_token_id)

                        leaving = mask & ~nxt_mask
                        if leaving:
                            _dec(leaving)
                            _persist_rt_join_runtime()

                        gained = nxt_mask & ~mask
                        if gained:
                            _inc(gained)
                            _persist_rt_join_runtime()

                        pending_tokens.add(t)
                        scheduled_q.put(t)
                        _persist_rt_join_runtime()
                        first = False
                    else:
                        # fanout child gets a NEW token_id
                        child_token_id = stable_id("token_id", f"{token_id}/{step_seq_current}:{i_fanout}:{nxt}").hex #uuid.uuid4().hex
                        t = (str(nxt), int(nxt_mask), child_token_id, str(token_id))

                        try:
                            mq.put_nowait({"type": "token.spawn", "parent_token_id": str(token_id), "child_token_id": child_token_id, "from_node_id": node_id, "to_node_id": str(nxt)})
                        except queue.Full:
                            pass
                        # trace token spawn
                        try:
                            tc_spawn = TraceContext(
                                run_id=str(run_id),
                                token_id=str(token_id),
                                step_seq=int(step_seq_current),
                                node_id=str(node_id),
                                attempt=1,
                                conversation_id=str(conversation_id) if conversation_id is not None else None,
                                turn_node_id=str(turn_node_id) if turn_node_id is not None else None,
                            )
                            self.emitter.emit(
                                type="token_spawned",
                                ctx=tc_spawn,
                                payload={"parent_token_id": str(token_id), "child_token_id": str(child_token_id), "to_node_id": str(nxt)},
                            )
                        except Exception:
                            pass

                        # fanout creates a new token (new outstanding obligations)
                        _inc(nxt_mask)

                        pending_tokens.add(t)
                        scheduled_q.put(t)
                        _persist_rt_join_runtime()

                if (step_seq_current % self.checkpoint_every_n_steps) == 0:
                    with self._maybe_step_uow():
                            self._persist_checkpoint(
                                conversation_id=conversation_id,
                                workflow_id=workflow_id,
                                run_id=run_id,
                                step_seq=step_seq_current,
                                state=state,
                                last_exec_node=last_exec_node,
                            )
                    try:

                        tc_ck = TraceContext(run_id=str(run_id), token_id=str(token_id), step_seq=int(step_seq_current), node_id=str(node_id), attempt=1, conversation_id=str(conversation_id), turn_node_id=str(turn_node_id))

                        self.emitter.emit(type="checkpoint_saved", ctx=tc_ck, payload={"step_seq": int(step_seq_current)})

                    except Exception:

                        pass
        raise Exception('unreacheable')

    def _route_next(
        self,
        edges: List[WorkflowEdge],
        state: WorkflowState,
        last_result: StepRunResult,
        fanout: bool,
    ) -> tuple[List[str], RouteDecision]:
        """
        Waterfall routing:

          1) Evaluate explicit edge predicates (e.predicate != None) in order.
             - Record *all* predicate evaluations (true/false) into RouteDecision.evaluated.
             - Select the first matching edge unless fanout/multiplicity allows multiple.

          2) If no predicate edges match, evaluate unconditional edges (e.predicate == None)
             using BasePredicate (node-level "next_step_names" decision).

          3) If still nothing matches, pick any is_default edge.

        Returns:
          (next_node_ids, route_decision)

        NOTE: This function is pure with respect to tracing; the caller is responsible
        for emitting RouteDecision into the trace sink.
        """
        matched: List[str] = []
        decision = RouteDecision(evaluated=[], selected=[])

        def _edge_id(e: WorkflowEdge) -> str:
            return str(getattr(e, "id", None) or getattr(e, "edge_id", None) or f"{getattr(e, 'predicate', None)}->{(getattr(e, 'target_ids', None) or [''])[0]}")

        def _first_target(e: WorkflowEdge) -> Optional[str]:
            tids = getattr(e, "target_ids", None) or []
            if not tids:
                return None
            return str(tids[0])

        def _stop_on_first(e: WorkflowEdge) -> bool:
            # stop if not fanout and edge multiplicity is not 'many'
            return (not fanout) and (getattr(e, "multiplicity", "one") != "many")

        # (1) explicit predicates
        for e in edges:
            if getattr(e, "predicate", None) is None:
                continue
            tgt = _first_target(e)
            if tgt is None:
                continue

            pred_name = str(getattr(e, "predicate", ""))
            pred = self.predicate_registry.get(pred_name)
            if pred is None:
                # record missing predicate as rejection (False)
                decision.evaluated.append((f"{_edge_id(e)}:{pred_name}", False))
                continue

            workflow_info = WorkflowEdgeInfo.from_workflow_edge(e)
            try:
                ok = bool(pred(workflow_info, state, last_result))
            except Exception:
                ok = False

            decision.evaluated.append((f"{_edge_id(e)}:{pred_name}", ok))
            if ok:
                matched.append(tgt)
                decision.selected.append((_edge_id(e), tgt, "predicate"))
                if _stop_on_first(e):
                    return matched[0:1], decision

        if matched:
            return (matched if fanout else matched[0:1]), decision

        # (2) unconditional edges (node-level decision)
        from typing import cast
        from .contract import BasePredicate

        node_decider = cast(Predicate, BasePredicate())
        for e in edges:
            if getattr(e, "predicate", None) is not None:
                continue
            tgt = _first_target(e)
            if tgt is None:
                continue

            workflow_info = WorkflowEdgeInfo.from_workflow_edge(e)
            try:
                ok = bool(node_decider(workflow_info, state, last_result))
            except Exception:
                ok = False

            # Record unconditional evaluations too (use a synthetic name)
            decision.evaluated.append((f"{_edge_id(e)}:<base>", ok))
            if ok:
                matched.append(tgt)
                decision.selected.append((_edge_id(e), tgt, "base"))
                if _stop_on_first(e):
                    return matched[0:1], decision

        if matched:
            return (matched if fanout else matched[0:1]), decision

        # (3) default edge
        for e in edges:
            if bool(getattr(e, "is_default", False)):
                tids = [str(x) for x in (getattr(e, "target_ids", None) or [])]
                if not tids:
                    continue
                picked = tids if fanout else tids[0:1]
                # record default selection once
                decision.selected.append((_edge_id(e), picked[0], "default"))
                return picked, decision

        return [], decision

    # --------------------
    # Persistence helpers (conversation_engine)
    # --------------------

    def _persist_workflow_run(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        turn_node_id: str,
        status: str,
    ) -> bool: # return success failure result
        # current engine always persist on edit, no batch persist mode needed
        from . import serialize  # keep runtime import-light
        from graph_knowledge_engine.models import WorkflowRunNode, Grounding, Span  # adjust import path to your package layout

        excerpt = f"workflow_run {workflow_id} {run_id} status={status}"
        span = Span(**_make_trace_span(conversation_id=conversation_id, excerpt=excerpt, doc_id=f"conv:{conversation_id}"))
        n = WorkflowRunNode(
            id=f"wf_run|{run_id}",
            label=f"Workflow run {workflow_id}",
            type="entity",
            doc_id=f"wf_run|{run_id}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            metadata={
                "entity_type": "workflow_run",
                "workflow_id": workflow_id,
                "workflow_version": "v1",
                "run_id": run_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_node_id,
                "status": status,
                "level_from_root": 0,
            },
            level_from_root = 0,
            domain_id = None,
            canonical_entity_id = None,
            embedding = None,
        )
        self.conversation_engine.add_node(n)
        return n

    def _update_workflow_run_status(self, conversation_id: str, run_id: str, status: str) -> None:
        # minimal approach: add an update node/event rather than mutate-in-place
        # (your engine likely doesn’t do partial updates easily)
        # You can later add a redirect/tombstone model update.
        return

    def _persist_step_exec(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        workflow_node_id: str,
        op: str,
        status: str,
        duration_ms: int,
        result: StepRunResult,
        state: WorkflowState,
        token_id: str | None = None,
        parent_token_id: str | None = None,
        join_mask: int | None = None,
        last_exec_node: Optional[WorkflowStepExecNode| WorkflowRunNode] = None
    ) -> WorkflowStepExecNode:
        # from graph_knowledge_engine.models import WorkflowStepExecNode, Grounding, Span  # adjust import path

        result_json = try_serialize_with_ref(result)
        
        excerpt = f"{dict(step=step_seq, op=op, status=status, duration_ms=duration_ms)}"
        span = Span(**_make_trace_span(conversation_id=conversation_id, excerpt=excerpt, doc_id=f"conv:{conversation_id}"))

        n = WorkflowStepExecNode(
            id=f"wf_step|{run_id}|{step_seq}",
            label=f"WF step {step_seq}: {op}",
            type="entity",
            doc_id=f"wf_step|{run_id}|{step_seq}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            
            level_from_root = 0,
            domain_id = None,
            canonical_entity_id= None,
            embedding=None,
            metadata={
                "entity_type": "workflow_step_exec",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "workflow_node_id": workflow_node_id,
                "step_seq": step_seq,
                "op": op,
                "status": status,
                "duration_ms": duration_ms,
                "result_json": result_json,
                "token_id": token_id,
                "parent_token_id": parent_token_id,
                "join_mask": join_mask,
                "conversation_id": conversation_id,
                "char_distance_from_last_summary": 0,
                "turn_distance_from_last_summary": 0,
                # "tail_turn_index": state["prev_turn_meta_summary"]["tail_turn_index"]
            },
        )
        self.conversation_engine.add_node(n)
        if result.conversation_node_id:
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{n.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="step_exec",
                page_number=1,
                start_char=0,
                end_char=len(n.summary),
                excerpt=n.summary,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system",
                    is_verified=True,
                    score=1.0,
                    notes="step run result",
                ),
            )
            # eid = f"wf_interstep_edge|{run_id}|{step_seq}|{result.conversation_node_id}"
            # e = ConversationEdge(id = eid, type = 'relationship', summary = f"results during {result.conversation_node_id}",
            #                      domain_id=None, label='run_result', 
            #                      properties={}, 
            #                      mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
            #                      source_ids=[n.safe_get_id()], target_ids=[result.conversation_node_id],
            #                      relation="run_result", source_edge_ids = [], target_edge_ids = [], embedding = None,
            #                      doc_id = f"wf_step|{run_id}|{step_seq}",
            #                      metadata={"relation":"run_result",
            #                                "source_id":[n.id],
            #                                "target_id":result.conversation_node_id,
            #                                "char_distance_from_last_summary": 0,
            #                                "turn_distance_from_last_summary": 0,
            #                             
            #                                },
            #                      )
            # self.conversation_engine.add_edge(e)
        if last_exec_node:
            content = f"{last_exec_node.safe_get_id()} next is {n.safe_get_id()}"
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{n.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="step_exec",
                page_number=1,
                start_char=0,
                end_char=len(content),
                excerpt=content,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system",
                    is_verified=True,
                    score=1.0,
                    notes="step run execution",
                ),
            )
            eid = f"wf_next_step_exec|{run_id}|{step_seq}|last::{last_exec_node.safe_get_id()}|to::{n.safe_get_id()}"
            e = ConversationEdge(id = eid, type = 'relationship', summary = f"wf_next_step_exec {step_seq=}",
                                 domain_id=None, label=f'wf_next_step_exec {step_seq=}', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[last_exec_node.safe_get_id()], target_ids=[n.safe_get_id()],
                                 relation="wf_next_step_exec", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"wf_next_step_exec|{run_id}|{step_seq}",
                                 metadata={"relation":"wf_next_step_exec",
                                           "source_id":[last_exec_node.safe_get_id()],
                                           "target_id":[n.safe_get_id()],
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                        
                                           },
                                 )
            self.conversation_engine.add_edge(e)
        if result.conversation_node_id:
            content = f"{n.safe_get_id()} created {result.conversation_node_id} durign execution"
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{n.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="step_exec",
                page_number=1,
                start_char=0,
                end_char=len(content),
                excerpt=content,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system",
                    is_verified=True,
                    score=1.0,
                    notes="step execution created node",
                ),
            )
            eid = f"conv:{conversation_id}|wfexe:{n.safe_get_id()}|created:{result.conversation_node_id}"
            e = ConversationEdge(id = eid, type = 'relationship', summary = f"created node during {step_seq=}",
                                 domain_id=None, label=f'created node during {step_seq=}', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[n.safe_get_id()], target_ids=[result.conversation_node_id],
                                 relation="created_child", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"wf_next_step_exec|{run_id}|{step_seq}",
                                 metadata={"relation":"created_child",
                                           "source_id":[n.safe_get_id()],
                                           "target_id":[result.conversation_node_id],
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                        
                                           },)
            
            self.conversation_engine.add_edge(e)
        return n
    def _persist_checkpoint(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        state: WorkflowState,
        last_exec_node: WorkflowStepExecNode
    ) -> None:
        from graph_knowledge_engine.models import WorkflowCheckpointNode, Grounding, Span  # adjust import path
        
        state_copy = {k:v for k, v in state.items() if k != '_deps'}
        state_json = try_serialize_with_ref(state_copy)
        excerpt = f"checkpoint step_seq={step_seq}"
        span = Span(**_make_trace_span(conversation_id=conversation_id, excerpt=excerpt, doc_id=f"conv:{conversation_id}"))

        n = WorkflowCheckpointNode(
            id=f"wf_ckpt|{run_id}|{step_seq}",
            label=f"WF checkpoint {step_seq}",
            type="entity",
            doc_id=f"wf_ckpt|{run_id}|{step_seq}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            metadata={
                "entity_type": "workflow_checkpoint",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "step_seq": step_seq,
                "state_json": state_json,
                "level_from_root": 0,
                "conversation_id":conversation_id
            },
            domain_id=None,canonical_entity_id=None,embedding=None,level_from_root=0
        )
        self.conversation_engine.add_node(n)
        
        if last_exec_node:
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{n.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="persist_checkpoint",
                page_number=1,
                start_char=0,
                end_char=len(n.summary),
                excerpt=n.summary,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="system",
                    is_verified=True,
                    score=1.0,
                    notes="persist_checkpoint",
                ),
            )
            eid = f"persist_checkpoint|{run_id}|{step_seq}|last::{last_exec_node.safe_get_id()}|to::{n.safe_get_id()}"
            e = ConversationEdge(id = eid, type = 'relationship', summary = f"persist_checkpoint during {step_seq=}",
                                 domain_id=None, label=f'persist_checkpoint during {step_seq=}', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[n.safe_get_id()], target_ids=[last_exec_node.safe_get_id()],
                                 relation="persist_checkpoint during", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"wf_next_step_exec|{run_id}|{step_seq}",
                                 metadata={"relation":"wf_next_step_exec",
                                           "source_id":[last_exec_node.safe_get_id()],
                                           "target_id":[n.safe_get_id()],
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                        
                                           },
                                 )
        self.conversation_engine.add_edge(e)