from __future__ import annotations

import time
import uuid
import queue
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..conversation_orchestrator import get_id_for_conversation_turn_edge
from graph_knowledge_engine.models import WorkflowNode, MentionVerification, ConversationEdge, WorkflowEdge

from .design import validate_workflow_design, Predicate
from .serialize import try_serialize_with_ref

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


class RunSuccess(BaseModel):
    conversation_node_id: Optional[str]
    state_update: list[StateUpdate]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["success"] = "success"

class RunFailure(BaseModel):
    conversation_node_id: Optional[str]
    state_update: list[StateUpdate] # can still update, append an error message
    errors: list[str]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["failure"] = "failure"
RunResult: TypeAlias = RunSuccess | RunFailure
StepFn: TypeAlias = Callable[["StepContext"], RunResult]
from ..conversation_state_contracts import WorkflowState

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, TypedDict, Iterator, TypeVar
from types import MappingProxyType
import threading
import queue


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

@dataclass
class StepContext:
    run_id: str
    workflow_id: str
    workflow_node_id: str
    op: str

    # Accept `state=` in __init__ but don't store it as a field
    state: InitVar["WorkflowState"]

    message_queue: "queue.Queue[Dict[str, Json]]"

    # real storage
    _state: "WorkflowState" = field(init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self, state: "WorkflowState") -> None:
        self._state = state

    @property
    def state_view(self) -> Mapping[str, Json]:
        return MappingProxyType(self._state)

    @property
    def state_write(self) -> "_StateWriteTxn":
        return _StateWriteTxn(self)

    def publish(self, msg: Dict[str, Json]) -> None:
        self.message_queue.put(msg)

    def drain(self, max_items: int = 200) -> List[Dict[str, Json]]:
        out: List[Dict[str, Json]] = []
        for _ in range(max_items):
            try:
                out.append(self.message_queue.get_nowait())
            except queue.Empty:
                break
        return out

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
    Executes a workflow design from workflow_engine and persists run traces to conversation_engine.
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
    ) -> None:
        from graph_knowledge_engine.engine import GraphKnowledgeEngine
        self.workflow_engine: GraphKnowledgeEngine = workflow_engine
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.step_resolver: Callable[[str], Callable[..., RunResult]] = step_resolver
        self.predicate_registry = predicate_registry
        self.checkpoint_every_n_steps = max(1, int(checkpoint_every_n_steps))
        self.max_workers = max_workers
        self.state_lock : dict[RunID, Lock] = {} # look up of run specific state lock
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
    ) -> Tuple[WorkflowState, str]:
        """
        Returns (final_state, run_id).

        Design lives in workflow_engine.
        Traces/checkpoints live in conversation_engine.
        """
        run_id = run_id or f"run|{uuid.uuid4()}"
        self.state_lock[str(run_id)] = Lock()
        mq: queue.Queue[Dict[str, Json]] = queue.Queue()

        start, nodes, adj = validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
        )
        # start, nodes, adj = load_workflow_design(workflow_engine=self.workflow_engine, workflow_id=workflow_id)

        state: WorkflowState = initial_state
        step_seq = 0

        # Persist workflow_run node in conversation_engine
        self._persist_workflow_run(
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            run_id=run_id,
            turn_node_id=turn_node_id,
            status="running",
        )

        scheduled_q: queue.Queue[str] = queue.Queue()
        scheduled_q.put(start.safe_get_id())
        done_q: queue.Queue[Tuple[str, RunResult, int, str]] = queue.Queue()  # (node_id, result, duration_ms, status)

        def worker(node_id: str, state) -> None:
            wn: WorkflowNode
            wn = nodes[node_id]
            op = nodes[node_id].op
            t0 = _now_ms()
            status = "ok"
            try:
                fn: Callable[..., RunResult] = self.step_resolver(op)
                ctx = StepContext(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    workflow_node_id=node_id,
                    op=op,
                    state=state,
                    message_queue=mq,
                )
                res: RunResult = fn(ctx)
            except Exception as e:
                status = "error"
                res = RunFailure(conversation_node_id = None, status = "failure", errors = [str(e)], state_update = [])
            try:
                mq.put_nowait(res.model_dump())
            except queue.Full as _e:
                raise
                
            t1 = _now_ms()
            done_q.put((node_id, res, max(0, t1 - t0), status))

        inflight: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while True:
                # schedule while capacity
                while len(inflight) < self.max_workers:
                    try:
                        nid = scheduled_q.get_nowait()
                    except queue.Empty:
                        break
                    inflight[nid] = pool.submit(worker, nid, state)

                if not inflight and scheduled_q.empty():
                    # done
                    self._update_workflow_run_status(conversation_id, run_id, "done")
                    return state, run_id

                node_id, run_result, dur_ms, status = done_q.get()
                run_state_lock: Lock = self.state_lock[str(run_id)]
                with run_state_lock:
                    self.apply_state_update(mute_state=state, state_update = run_result.state_update)
                    
                inflight.pop(node_id, None)

                wn = nodes[node_id]

                # persist step exec trace node
                self._persist_step_exec(
                    conversation_id=conversation_id,
                    workflow_id=workflow_id,
                    run_id=run_id,
                    step_seq=step_seq,
                    workflow_node_id=node_id,
                    op=wn.op,
                    status=status,
                    duration_ms=dur_ms,
                    result=run_result,
                    state=state
                )

                # single-writer state apply:
                # default behavior: store under result.<op> and allow ops to also write into state
                # state[f"result.{wn.op}"] = result result only visible 

                # checkpoint
                if (step_seq % self.checkpoint_every_n_steps) == 0:
                    self._persist_checkpoint(
                        conversation_id=conversation_id,
                        workflow_id=workflow_id,
                        run_id=run_id,
                        step_seq=step_seq,
                        state=state,
                    )

                step_seq += 1

                # route
                if wn.terminal or len(adj.get(node_id, [])) == 0:
                    continue

                edges = adj.get(node_id, [])
                next_nodes = self._route_next(edges, state, run_result, wn.fanout)
                for nxt in next_nodes:
                    scheduled_q.put(nxt)
        self.state_lock.pop(run_id)

    def _route_next(
        self,
        edges: List[WorkflowEdge],
        state: WorkflowState,
        last_result: RunResult,
        fanout: bool,
    ) -> List[str]:
        # waterfall, edge predicate, if no next step, inspect node decision -> finally check any default
        matched: List[str] = []
        for e in edges:
            workflow_info = WorkflowEdgeInfo.from_workflow_edge(e)
            if e.predicate is None:
                continue
            pred = self.predicate_registry.get(e.predicate)
            if pred is None:
                continue
            ok = False
            try:
                ok = bool(pred(workflow_info, state, last_result))
            except Exception:
                ok = False
            if ok:
                matched.append(e.target_ids[0])
                if not fanout and e.multiplicity != "many":
                    return matched

        if matched and (fanout or any(getattr(ed, "multiplicity", "one") == "many" for ed in edges)):
            return matched
        
        
        from typing import cast
        from .contract import BasePredicate
        # fall back if no predicate ever gove, then let node decide
        if not matched:
            # node decide logic
            for e in edges:
                # if e.predicate is None:
                pred = cast(Predicate, BasePredicate())
                
                workflow_info = WorkflowEdgeInfo.from_workflow_edge(e)
                # else:
                    # should not run
                    # pred = self.predicate_registry.get(e.predicate)
                try:
                    ok = bool(pred(workflow_info, state, last_result))
                except Exception:
                    ok = False
                if ok:
                    matched.append(e.target_ids[0])
                    if not fanout and e.multiplicity != "many":
                        return matched
                
            if matched:
                return matched
            for e in edges:
                if e.is_default:
                    if fanout:
                        return e.target_ids
                    else:
                        return e.target_ids[0:1]
        # not matched and no default set
        
        return matched

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
        return True

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
        result: RunResult,
        state: WorkflowState,
    ) -> None:
        from graph_knowledge_engine.models import WorkflowStepExecNode, Grounding, Span  # adjust import path

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
                insertion_method="summary_turn",
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
            eid = f"wf_interstep_edge|{run_id}|{step_seq}|{result.conversation_node_id}"
            e = ConversationEdge(id = eid, type = 'relationship', summary = f"results during {result.conversation_node_id}",
                                 domain_id=None, label='run_result', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[n.safe_get_id()], target_ids=[result.conversation_node_id],
                                 relation="run_result", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"wf_step|{run_id}|{step_seq}",
                                 metadata={"relation":"run_result",
                                           "source_id":[n.id],
                                           "target_id":result.conversation_node_id,
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                        #    "tail_turn_index": state["prev_turn_meta_summary"]["tail_turn_index"]
                                           },
                                 )
            self.conversation_engine.add_edge(e)

    def _persist_checkpoint(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        state: WorkflowState,
    ) -> None:
        from graph_knowledge_engine.models import WorkflowCheckpointNode, Grounding, Span  # adjust import path

        state_json = try_serialize_with_ref(state)
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
