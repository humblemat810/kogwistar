"""Client-suspended workflow continuation operations."""

from __future__ import annotations

import queue
from collections.abc import Callable, Mapping
from os import PathLike
from typing import TYPE_CHECKING, Any, Protocol, cast

from kogwistar.id_provider import stable_id
from kogwistar.runtime.models import (
    RunFailure,
    RunSuspended,
    StateUpdate,
    StepRunResult,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRunNode,
    WorkflowState,
    WorkflowStepExecNode,
)
from kogwistar.runtime.telemetry import TraceContext

from .checkpoint_state import decode_checkpoint_state
from .contract import Predicate
from .design import validate_workflow_design
from .join_state import JoinToken, RuntimeJoinSnapshot, restore_runtime_join_snapshot
from .join_topology import _compute_may_reach_join_bitsets, _iter_bits
from .runtime_conversation import _json_int
from .runtime_checkpoints import CheckpointRuntimeHost
from .step_context import RouteDecision, RunResult, StepResolver
from .telemetry import EventEmitter

if TYPE_CHECKING:
    from ..engine_core.engine import GraphKnowledgeEngine


class SuspendedResumeHost(CheckpointRuntimeHost, Protocol):
    @property
    def workflow_engine(self) -> GraphKnowledgeEngine: ...

    @property
    def predicate_registry(self) -> dict[str, Predicate]: ...

    @property
    def step_resolver(self) -> StepResolver: ...

    @property
    def emitter(self) -> EventEmitter: ...

    @property
    def checkpoint_every_n_steps(self) -> int: ...

    def apply_state_update(
        self,
        mute_state: WorkflowState,
        state_update: list[tuple[str, dict[str, Any]]] | list[StateUpdate],
        update: dict[str, Any] | None = None,
    ) -> None: ...

    def _route_next(
        self,
        edges: list[WorkflowEdge],
        state: WorkflowState,
        last_result: StepRunResult,
        fanout: bool,
        nodes: Mapping[str, WorkflowNode] | None = None,
    ) -> tuple[list[str], RouteDecision]: ...

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
        last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
    ) -> WorkflowStepExecNode: ...

    def _persist_checkpoint(
        self,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        state: WorkflowState,
        last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
        trace_context: TraceContext | None = None,
    ) -> None: ...

    def _persist_failed_terminal(
        self,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        accepted_step_seq: int,
        errors: list[str] | None,
        last_processed_node_id: str | None = None,
    ) -> str: ...

    def _persist_completed_terminal(
        self,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        accepted_step_seq: int,
        last_processed_node_id: str | None = None,
    ) -> str: ...

    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
        _trace_context: TraceContext | None = None,
        _authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult: ...


class SuspendedRunResumeMixin:
    """Resume client-suspended runs through the existing workflow runtime."""

    def resume_run(
        self: SuspendedResumeHost,
        *,
        run_id: str,
        suspended_node_id: str,
        suspended_token_id: str,
        client_result: StepRunResult,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        cache_dir: str | PathLike[str] | None = None,
        _parent_trace_context: TraceContext | None = None,
    ) -> RunResult:
        """
        Resumes a suspended workflow run using the results of an externally executed task
        (e.g. from a client-side sandbox).

        This assumes that:
        1. A checkpoint exists for the suspended state.
        2. The suspended step was saved in _rt_join_snapshot as pending with mask logic intact.
        """
        from .runtime import WorkflowRuntime
        from .rust_runtime_authority import (
            RustRuntimeAuthority,
            rust_runtime_authority_url,
        )

        authority_url = rust_runtime_authority_url()
        if authority_url is not None:
            authority = RustRuntimeAuthority(
                runtime=cast(WorkflowRuntime, self),
                base_url=authority_url,
                cache_dir=cache_dir,
            )
            try:
                return authority.resume(
                    run_id=str(run_id),
                    suspended_node_id=str(suspended_node_id),
                    suspended_token_id=str(suspended_token_id),
                    client_result=client_result,
                    workflow_id=str(workflow_id),
                    conversation_id=str(conversation_id),
                    turn_node_id=str(turn_node_id),
                )
            finally:
                authority.close()

        # Load the latest checkpoint for the run to reconstruct initial_state.
        # Runtime-maintained named projections are the serving path; graph scans
        # are retained for older stores and projection repair scenarios.
        latest_ckpt = self._latest_checkpoint_for_run(
            conversation_id=conversation_id,
            run_id=str(run_id),
        )
        if latest_ckpt is None:
            raise ValueError(f"Cannot resume run {run_id}: no checkpoints found.")

        checkpoint_metadata = latest_ckpt.metadata
        initial_state = decode_checkpoint_state(
            checkpoint_metadata.get("state_json"), run_id=str(run_id)
        )

        persisted_trace = checkpoint_metadata.get("trace_id")
        persisted_span = checkpoint_metadata.get("run_execution_span_id")
        resume_trace_context = _parent_trace_context
        if resume_trace_context is None and persisted_trace and persisted_span:
            candidate = TraceContext(
                run_id=str(run_id),
                token_id=str(suspended_token_id),
                step_seq=_json_int(checkpoint_metadata.get("step_seq"), default=0),
                node_id=str(suspended_node_id),
                trace_id=str(persisted_trace),
                span_id=str(persisted_span),
            )
            if candidate.has_valid_w3c_ids:
                resume_trace_context = candidate

        lifecycle_run_id = str(run_id)
        lifecycle_conversation_id = str(conversation_id)
        lifecycle_turn_node_id = str(turn_node_id)

        def _resume_lifecycle_trace_context(
            *,
            token_id: str,
            step_seq: int,
            node_id: str,
            attempt: int = 1,
            run_id: str | None = None,
            conversation_id: str | None = None,
            turn_node_id: str | None = None,
        ) -> TraceContext:
            effective_run_id = str(run_id or lifecycle_run_id)
            effective_conversation_id = str(
                conversation_id or lifecycle_conversation_id
            )
            effective_turn_node_id = str(turn_node_id or lifecycle_turn_node_id)
            base = resume_trace_context or TraceContext(
                run_id=effective_run_id,
                token_id=str(token_id),
                step_seq=int(step_seq),
                node_id=str(node_id),
                attempt=int(attempt),
                conversation_id=effective_conversation_id,
                turn_node_id=effective_turn_node_id,
            )
            return base.child_span(
                token_id=str(token_id),
                step_seq=int(step_seq),
                node_id=str(node_id),
                attempt=int(attempt),
            )

        # Apply the client's state update directly to the initial state for all resumable result types.
        self.apply_state_update(
            mute_state=initial_state,
            state_update=client_result.state_update,
            update=client_result.update,
        )

        # Start the runtime again
        # The runtime will pick up the `pending` tokens from `_rt_join` stored in `initial_state`
        # Because the token was parked at `suspended_node_id`, it will try to re-execute it.
        # But wait! We don't want to re-execute the suspended node. We want to route *from* it.

        # To fix this, we manually route from the suspended node and update the `_rt_join`
        # pending state before calling run().
        _start, nodes, adj = validate_workflow_design(
            workflow_engine=self.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.predicate_registry,
            resolver=self.step_resolver,
        )

        node_ids_list = list(nodes.keys())
        join_node_ids: list[str] = []
        for _nid, _wn in nodes.items():
            _md = _wn.metadata
            if bool(_md.get("wf_join", False)) or _wn.op == "join":
                join_node_ids.append(str(_nid))

        _may_reach_join = (
            _compute_may_reach_join_bitsets(
                node_ids=node_ids_list, adj=adj, join_ids=join_node_ids
            )
            if join_node_ids
            else {nid: 0 for nid in node_ids_list}
        )

        wn = nodes[suspended_node_id]
        edges = adj.get(suspended_node_id, [])
        step_seq_current = (
            _json_int(checkpoint_metadata.get("step_seq"), default=0) + 1
        )
        next_nodes: list[str] = []
        route_decision = RouteDecision(evaluated=[], selected=[])
        client_status = client_result.status
        if client_status != "suspended":
            next_nodes, route_decision = self._route_next(
                edges, initial_state, client_result, wn.fanout, nodes
            )
        rt_join = restore_runtime_join_snapshot(
            initial_state,
            join_node_ids,
            allow_legacy_lengths=True,
            allow_single_mask_waiters=True,
        )
        if rt_join is None:
            rt_join = RuntimeJoinSnapshot(
                join_node_ids=list(join_node_ids),
                join_outstanding=[0 for _ in join_node_ids],
                join_waiters={join_id: [] for join_id in join_node_ids},
                pending=[],
                suspended=[],
            )
        pending: list[JoinToken] = list(rt_join["pending"])
        suspended: list[JoinToken] = list(rt_join["suspended"])

        new_pending: list[JoinToken] = []
        new_suspended: list[JoinToken] = []
        token_found = False
        mask_to_distribute = 0
        parent_token_id = None
        resumed_exec_node: WorkflowStepExecNode | None = None

        previous_exec_node_id = f"wf_step|{run_id}|{step_seq_current - 1}"
        previous_exec_nodes = self.conversation_engine.read.get_nodes(
            ids=[previous_exec_node_id],
            limit=1,
        )
        if previous_exec_nodes:
            previous_exec_node = previous_exec_nodes[0]
        else:
            fallback_exec_nodes = self.conversation_engine.read.get_nodes(
                ids=[f"wf_run|{run_id}"],
                limit=1,
            )
            previous_exec_node = fallback_exec_nodes[0] if fallback_exec_nodes else None

        def _extract(items: list[JoinToken], keep: list[JoinToken]) -> None:
            nonlocal token_found, mask_to_distribute, parent_token_id
            for item in items:
                nid, tok_mask, tok_id, tok_parent = item
                if (
                    (not token_found)
                    and nid == str(suspended_node_id)
                    and tok_id == str(suspended_token_id)
                ):
                    token_found = True
                    mask_to_distribute = int(tok_mask)
                    parent_token_id = tok_parent
                else:
                    keep.append(item)

        _extract(suspended, new_suspended)
        _extract(pending, new_pending)

        if not token_found:
            raise ValueError(
                f"Token {suspended_token_id} for node {suspended_node_id} not found in suspended runtime state."
            )
        token_id = suspended_token_id
        node_id = suspended_node_id
        # Trace routing decision (includes rejections) at orchestration choke point
        try:
            tc = _resume_lifecycle_trace_context(
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
                    "evaluated": [
                        (name, value)
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
        # Modify the parked suspended tokens in state["_rt_join"]["suspended"] first.
        # Older checkpoints may still have the token in pending, so we accept both for compatibility.
        # Add next tokens
        _join_pos: dict[str, int] = {
            join_id: index for index, join_id in enumerate(join_node_ids)
        }
        jo: list[int] = list(rt_join["join_outstanding"])

        def _inc(m: int) -> None:
            for bi in _iter_bits(m):
                jo[bi] += 1

        def _dec(m: int) -> None:
            for bi in _iter_bits(m):
                jo[bi] = max(0, jo[bi] - 1)

        first = True

        if client_status == "suspended":
            new_suspended.append(
                (
                    str(suspended_node_id),
                    int(mask_to_distribute),
                    str(suspended_token_id),
                    parent_token_id,
                )
            )
        elif not next_nodes:
            _dec(mask_to_distribute)
        else:
            for i_fanout, nxt in enumerate(next_nodes):
                nxt = str(nxt)
                nxt_mask = int(_may_reach_join.get(nxt, 0))

                if first:
                    t = (nxt, nxt_mask, suspended_token_id, parent_token_id)
                    leaving = mask_to_distribute & ~nxt_mask
                    if leaving:
                        _dec(leaving)
                    gained = nxt_mask & ~mask_to_distribute
                    if gained:
                        _inc(gained)
                    new_pending.append(t)
                    first = False
                else:
                    child_token_id = stable_id(
                        "token_id",
                        f"{suspended_token_id}/{step_seq_current}:{i_fanout}:{nxt}",
                    ).hex
                    t = (nxt, nxt_mask, child_token_id, suspended_token_id)
                    _inc(nxt_mask)
                    new_pending.append(t)

        rt_join["pending"] = new_pending
        rt_join["suspended"] = new_suspended
        rt_join["join_outstanding"] = jo
        initial_state["_rt_join"] = rt_join

        # Log the step exec for the external execution so it's recorded
        resumed_exec_node = self._persist_step_exec(
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            run_id=run_id,
            step_seq=step_seq_current,
            workflow_node_id=suspended_node_id,
            op="client_sandbox_resume",
            status=(
                "suspended"
                if client_status == "suspended"
                else "failure"
                if client_status == "failure"
                else "ok"
            ),
            duration_ms=0,
            result=client_result,
            state=initial_state,
            token_id=suspended_token_id,
            parent_token_id=parent_token_id,
            join_mask=mask_to_distribute,
            last_exec_node=cast(
                WorkflowStepExecNode | WorkflowRunNode | None,
                previous_exec_node,
            ),
        )
        if (step_seq_current % self.checkpoint_every_n_steps) == 0:
            self._persist_checkpoint(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                step_seq=step_seq_current,
                state=initial_state,
                last_exec_node=resumed_exec_node,
            )
            try:
                tc_ck = _resume_lifecycle_trace_context(
                    run_id=str(run_id),
                    token_id=str(suspended_token_id),
                    step_seq=int(step_seq_current),
                    node_id=str(suspended_node_id),
                    attempt=1,
                    conversation_id=str(conversation_id),
                    turn_node_id=str(turn_node_id),
                )
                self.emitter.emit(
                    type="checkpoint_saved",
                    ctx=tc_ck,
                    payload={"step_seq": int(step_seq_current)},
                )
            except Exception:
                pass

        if client_status == "suspended":
            assert isinstance(client_result, RunSuspended)
            try:
                self.emitter.emit(
                    type="workflow_step_suspended",
                    ctx=_resume_lifecycle_trace_context(
                        run_id=str(run_id),
                        token_id=str(suspended_token_id),
                        step_seq=int(step_seq_current),
                        node_id=str(suspended_node_id),
                        attempt=1,
                        conversation_id=str(conversation_id),
                        turn_node_id=str(turn_node_id),
                    ),
                    payload=client_result.resume_payload,
                )
            except Exception:
                pass
            return RunResult(
                final_state=initial_state,
                run_id=run_id,
                mq=queue.Queue(),
                status="suspended",
            )

        if new_pending:
            return self.run(
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                turn_node_id=str(turn_node_id or ""),
                initial_state=initial_state,
                run_id=run_id,
                cache_dir=cache_dir,
                _resume_step_seq=int(step_seq_current) + 1,
                _resume_last_exec_node=resumed_exec_node,
            )

        if new_suspended:
            return RunResult(
                final_state=initial_state,
                run_id=run_id,
                mq=queue.Queue(),
                status="suspended",
            )

        if client_status == "failure":
            assert isinstance(client_result, RunFailure)
            try:
                self._persist_failed_terminal(
                    conversation_id=str(conversation_id),
                    workflow_id=str(workflow_id),
                    run_id=str(run_id),
                    accepted_step_seq=int(step_seq_current),
                    errors=client_result.errors,
                    last_processed_node_id=str(suspended_node_id),
                )
                tc_done = _resume_lifecycle_trace_context(
                    run_id=str(run_id),
                    token_id=str(suspended_token_id),
                    step_seq=int(step_seq_current),
                    node_id=str(suspended_node_id),
                    attempt=1,
                    conversation_id=str(conversation_id),
                    turn_node_id=str(turn_node_id),
                )
                self.emitter.emit(
                    type="workflow_run_failed",
                    ctx=tc_done,
                    payload={
                        "workflow_id": str(workflow_id),
                        "errors": [
                            str(error)[-4_000:]
                            for error in client_result.errors
                        ],
                    },
                )
            except Exception:
                pass
            return RunResult(
                final_state=initial_state,
                run_id=run_id,
                mq=queue.Queue(),
                status="failure",
                errors=[
                    str(error)
                    for error in client_result.errors
                ],
            )

        self._persist_completed_terminal(
            conversation_id=str(conversation_id),
            workflow_id=str(workflow_id),
            run_id=str(run_id),
            accepted_step_seq=int(step_seq_current),
            last_processed_node_id=str(suspended_node_id),
        )
        try:
            tc_done = _resume_lifecycle_trace_context(
                run_id=str(run_id),
                token_id=str(suspended_token_id),
                step_seq=int(step_seq_current),
                node_id=str(suspended_node_id),
                attempt=1,
                conversation_id=str(conversation_id),
                turn_node_id=str(turn_node_id),
            )
            self.emitter.emit(
                type="workflow_run_completed",
                ctx=tc_done,
                payload={"workflow_id": str(workflow_id)},
            )
        except Exception:
            pass
        return RunResult(
            final_state=initial_state,
            run_id=run_id,
            mq=queue.Queue(),
            status="succeeded",
        )
