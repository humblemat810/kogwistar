from __future__ import annotations

import time
import uuid
import queue
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from graph_knowledge_engine.models import WorkflowNode, MentionVerification, ConversationEdge

from .design import validate_workflow_design, Predicate
from .serialize import try_serialize_with_ref


Json = Any
State = Dict[str, Json]
# Result = Json
from typing import TypedDict, Literal, TypeAlias, Any

from dataclasses import dataclass
from pydantic import BaseModel

class RunSuccess(BaseModel):
    conversation_node_id: Optional[str]
    outputs: list[Any]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["success"] = "success"

class RunFailure(BaseModel):
    conversation_node_id: Optional[str]
    
    errors: list[str]
    next_step_names: list[str] = []  # empty will by default fan out all
    status: Literal["failure"] = "failure"
RunResult: TypeAlias = RunSuccess | RunFailure
StepFn: TypeAlias = Callable[["StepContext"], RunResult]


@dataclass
class StepContext:
    run_id: str
    workflow_id: str
    workflow_node_id: str
    op: str
    state: State
    message_queue: "queue.Queue[Dict[str, Json]]"

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

    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: State,
        run_id: Optional[str] = None,
    ) -> Tuple[State, str]:
        """
        Returns (final_state, run_id).

        Design lives in workflow_engine.
        Traces/checkpoints live in conversation_engine.
        """
        run_id = run_id or f"run|{uuid.uuid4()}"
        mq: queue.Queue[Dict[str, Json]] = queue.Queue()

        start, nodes, adj = validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
        )
        # start, nodes, adj = load_workflow_design(workflow_engine=self.workflow_engine, workflow_id=workflow_id)

        state: State = dict(initial_state)
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

        def worker(node_id: str) -> None:
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
                res = RunFailure(conversation_node_id = None, status = "failure", errors = [str(e)])
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
                    inflight[nid] = pool.submit(worker, nid)

                if not inflight and scheduled_q.empty():
                    # done
                    self._update_workflow_run_status(conversation_id, run_id, "done")
                    return state, run_id

                node_id, run_result, dur_ms, status = done_q.get()
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

    def _route_next(
        self,
        edges: List[Any],
        state: State,
        last_result: RunResult,
        fanout: bool,
    ) -> List[str]:
        matched: List[str] = []

        for e in edges:
            if e.predicate is None:
                continue
            pred = self.predicate_registry.get(e.predicate)
            if pred is None:
                continue
            ok = False
            try:
                ok = bool(pred(state, last_result))
            except Exception:
                ok = False
            if ok:
                matched.append(e.dst)
                if not fanout and e.multiplicity != "many":
                    return matched

        if matched and (fanout or any(getattr(ed, "multiplicity", "one") == "many" for ed in edges)):
            return matched

        if not matched:
            for e in edges:
                if e.is_default:
                    return [e.dst]

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
        return True  # current engine always persist on edit, no batch persist mode needed
        # from . import serialize  # keep runtime import-light
        # from graph_knowledge_engine.models import WorkflowRunNode, Grounding, Span  # adjust import path to your package layout

        # excerpt = f"workflow_run {workflow_id} {run_id} status={status}"
        # span = Span(**_make_trace_span(conversation_id=conversation_id, excerpt=excerpt, doc_id=f"conv:{conversation_id}"))
        # n = WorkflowRunNode(
        #     id=f"wf_run|{run_id}",
        #     label=f"Workflow run {workflow_id}",
        #     type="entity",
        #     doc_id=f"wf_run|{run_id}",
        #     summary=excerpt,
        #     mentions=[Grounding(spans=[span])],
        #     properties={},
        #     metadata={
        #         "entity_type": "workflow_run",
        #         "workflow_id": workflow_id,
        #         "workflow_version": "v1",
        #         "run_id": run_id,
        #         "conversation_id": conversation_id,
        #         "turn_node_id": turn_node_id,
        #         "status": status,
        #         "level_from_root": 0,
        #     },
        # )
        # self.conversation_engine.add_node(n)

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
    ) -> None:
        from graph_knowledge_engine.models import WorkflowStepExecNode, Grounding, Span  # adjust import path

        result_json = try_serialize_with_ref(result)
        excerpt = f"step {step_seq} op={op} status={status} dur={duration_ms}ms"
        span = Span(**_make_trace_span(conversation_id=conversation_id, excerpt=excerpt, doc_id=f"conv:{conversation_id}"))

        n = WorkflowStepExecNode(
            id=f"wf_step|{run_id}|{step_seq}",
            label=f"WF step {step_seq}: {op}",
            type="entity",
            doc_id=f"wf_step|{run_id}|{step_seq}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
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
                "level_from_root": 0,
            },
        )
        self.conversation_engine.add_node(n)
        if result["conversation_node_id"]:
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
                    notes=f"step run result",
                ),
            )    
            e = ConversationEdge(type = 'relationship', summary = f"results during {result['conversation_node_id']}",
                                 domain_id=None, label='run_result', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[n.safe_get_id()], target_ids=[result['conversation_node_id']],
                                 relation="run_result", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"wf_step|{run_id}|{step_seq}",
                                 metadata={"relation":"run_result",
                                           "source_id":[n.id],
                                           "target_id":result['conversation_node_id']},
                                 )
            self.conversation_engine.add_edge(e)

    def _persist_checkpoint(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        state: State,
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
            },
        )
        self.conversation_engine.add_node(n)
