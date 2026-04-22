from __future__ import annotations

import asyncio
import contextvars
import inspect
import queue
import uuid
from contextlib import nullcontext
from typing import Any, Awaitable, Callable, TypeAlias

from ..id_provider import stable_id
from .models import RunFailure, StepRunResult, WorkflowState
from .executor import TerminalStatus, WorkflowExecutor
from .runtime import (
    apply_state_update_inplace,
    RunResult,
    StepContext,
    WorkflowRuntime,
    _compute_may_reach_join_bitsets,
    validate_initial_state,
)
from .design import validate_workflow_design

SyncStepFn: TypeAlias = Callable[[StepContext], StepRunResult]
AsyncStepFn: TypeAlias = Callable[[StepContext], Awaitable[StepRunResult]]


_CANCEL_REQUESTED_CTX: contextvars.ContextVar[Callable[[str], bool] | None] = (
    contextvars.ContextVar("kogwistar_async_cancel_requested", default=None)
)


def _as_sync_step_fn(fn: Callable[[StepContext], Any]) -> SyncStepFn:
    """Adapt step handler to sync callable expected by WorkflowRuntime.

    If handler returns an awaitable, execute it on a short-lived event loop in
    worker thread context.
    """

    def _wrapped(ctx: StepContext) -> StepRunResult:
        out = fn(ctx)
        if inspect.isawaitable(out):
            return asyncio.run(out)
        return out

    return _wrapped


class _SyncResolverAdapter:
    """Preserve resolver metadata while adapting handlers for sync runtime.

    WorkflowRuntime reads resolver attributes like `nested_ops` and
    `_state_schema`. This adapter proxies those attributes from the original
    resolver and only adapts the call result shape (awaitable -> concrete).
    """

    def __init__(self, resolver: Any):
        self._resolver = resolver
        self.nested_ops = getattr(resolver, "nested_ops", set())
        self._state_schema = getattr(resolver, "_state_schema", {})

    def close_sandbox_run(self, run_id: str) -> None:
        close_run = getattr(self._resolver, "close_sandbox_run", None)
        if callable(close_run):
            close_run(run_id)

    def __call__(self, op: str) -> SyncStepFn:
        return _as_sync_step_fn(self._resolver(op))


class AsyncWorkflowRuntime(WorkflowExecutor):
    """Async facade preserving WorkflowRuntime semantics for first async slice.

    Current implementation delegates scheduling/state semantics to the existing
    WorkflowRuntime, while allowing async step handlers through resolver
    adaptation.
    """

    def __init__(
        self,
        *,
        workflow_engine: Any,
        conversation_engine: Any,
        step_resolver: Callable[[str], Callable[[StepContext], Any]],
        predicate_registry: dict[str, Any],
        checkpoint_every_n_steps: int = 1,
        max_workers: int = 4,
        transaction_mode: str | None = None,
        trace: bool = True,
        events: Any | None = None,
        sink: Any | None = None,
        cancel_requested: Callable[[str], bool] | None = None,
        fast_trace_persistence: bool | None = None,
        experimental_native_scheduler: bool = False,
    ) -> None:
        self._raw_step_resolver = step_resolver
        self._predicate_registry = predicate_registry
        self._workflow_engine = workflow_engine
        self._conversation_engine = conversation_engine
        self.max_workers = max_workers
        self.cancel_requested = cancel_requested
        self.experimental_native_scheduler = bool(experimental_native_scheduler)
        self._resolver_adapter = _SyncResolverAdapter(step_resolver)

        self._sync_runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=self._resolver_adapter,
            predicate_registry=predicate_registry,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            max_workers=max_workers,
            transaction_mode=transaction_mode,
            trace=trace,
            events=events,
            sink=sink,
            cancel_requested=cancel_requested,
            fast_trace_persistence=fast_trace_persistence,
        )
        # Contract anchors for tests/docs: async runtime reuses sync runtime context/result shape.
        self.step_context_type = StepContext
        self.step_result_type = StepRunResult
        self.terminal_status_values: set[TerminalStatus] = {
            "succeeded",
            "failed",
            "cancelled",
            "suspended",
        }

    @property
    def sync_runtime(self) -> WorkflowRuntime:
        return self._sync_runtime

    async def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: Any | None = None,
    ) -> RunResult:
        if _resume_step_seq is not None or _resume_last_exec_node is not None:
            return self._sync_runtime.run(
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                initial_state=initial_state,
                run_id=run_id,
                cache_dir=cache_dir,
                _resume_step_seq=_resume_step_seq,
                _resume_last_exec_node=_resume_last_exec_node,
            )
        if self.experimental_native_scheduler:
            return await self._run_native_async(
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                initial_state=initial_state,
                run_id=run_id,
                cache_dir=cache_dir,
            )
        # First async slice keeps exact sync runtime semantics and avoids
        # introducing scheduler differences yet; this call is intentionally
        # direct (blocking) until async scheduler core lands.
        return self._sync_runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=initial_state,
            run_id=run_id,
            cache_dir=cache_dir,
            _resume_step_seq=_resume_step_seq,
            _resume_last_exec_node=_resume_last_exec_node,
        )

    def _resolve_async_step_fn(self, op: str):
        resolver = self._raw_step_resolver
        resolve_async = getattr(resolver, "resolve_async", None)
        if callable(resolve_async):
            return resolve_async(op)
        fn = resolver(op)
        if inspect.iscoroutinefunction(fn):
            return fn

        async def _wrapped(ctx: StepContext):
            return fn(ctx)

        return _wrapped

    def _child_workflow_initial_state(
        self,
        *,
        parent_state: WorkflowState,
        invocation: Any,
    ) -> WorkflowState:
        return self._sync_runtime._child_workflow_initial_state(  # type: ignore[attr-defined]
            parent_state=parent_state,
            invocation=invocation,
        )

    def _apply_workflow_invocation_result(
        self,
        *,
        state: WorkflowState,
        invocation: Any,
        child_result: RunResult,
    ) -> None:
        self._sync_runtime._apply_workflow_invocation_result(  # type: ignore[attr-defined]
            state=state,
            invocation=invocation,
            child_result=child_result,
        )

    async def _run_workflow_invocation_async(
        self,
        *,
        invocation: Any,
        parent_state: WorkflowState,
        conversation_id: str,
        turn_node_id: str,
        parent_run_id: str,
        cache_dir: str | None = None,
    ) -> RunResult:
        if getattr(invocation, "workflow_design", None) is not None:
            wf_design = getattr(invocation, "workflow_design")
            if str(getattr(wf_design, "workflow_id", "")) != str(
                getattr(invocation, "workflow_id", "")
            ):
                raise ValueError(
                    "workflow_design.workflow_id must match workflow_id on the invocation"
                )
            self._sync_runtime._persist_workflow_design_artifact(wf_design)  # type: ignore[attr-defined]

        child_state = self._child_workflow_initial_state(
            parent_state=parent_state,
            invocation=invocation,
        )
        child_run_id = getattr(invocation, "run_id", None) or str(
            stable_id(
                "workflow.child_run",
                parent_run_id,
                str(getattr(invocation, "workflow_id", "")),
                str(getattr(invocation, "result_state_key", "") or ""),
                str(getattr(invocation, "turn_node_id", "") or turn_node_id),
            )
        )
        inherited_cancel = _CANCEL_REQUESTED_CTX.get() or self.cancel_requested

        def _child_cancel_requested(child_id: str) -> bool:
            parent_cancelled = bool(inherited_cancel and inherited_cancel(parent_run_id))
            child_cancelled = bool(inherited_cancel and inherited_cancel(child_id))
            return parent_cancelled or child_cancelled

        token = _CANCEL_REQUESTED_CTX.set(_child_cancel_requested)
        try:
            return await self.run(
                workflow_id=str(getattr(invocation, "workflow_id", "")),
                conversation_id=str(
                    getattr(invocation, "conversation_id", None) or conversation_id
                ),
                turn_node_id=str(getattr(invocation, "turn_node_id", None) or turn_node_id),
                initial_state=child_state,
                run_id=child_run_id,
                cache_dir=cache_dir,
            )
        finally:
            _CANCEL_REQUESTED_CTX.reset(token)

    @staticmethod
    def _edge_priority(edge: Any) -> int:
        md = getattr(edge, "metadata", {}) or {}
        try:
            return int(md.get("wf_priority", 100))
        except Exception:
            return 100

    @staticmethod
    def _select_next_edges(node: Any, edges: list[Any], state: WorkflowState, result: StepRunResult, predicate_registry: dict[str, Any]) -> list[Any]:
        if not edges:
            return []
        edges = sorted(list(edges), key=AsyncWorkflowRuntime._edge_priority)
        fanout = bool((getattr(node, "metadata", {}) or {}).get("wf_fanout", False))
        route_names = list(getattr(result, "next_step_names", []) or [])
        if route_names:
            routed = [
                e
                for e in edges
                if str(getattr(e, "label", "")) in set(str(n) for n in route_names)
            ]
            if routed:
                if fanout:
                    return routed
                many = [
                    e
                    for e in routed
                    if (getattr(e, "metadata", {}) or {}).get("wf_multiplicity")
                    == "many"
                ]
                if many:
                    return many
                return [routed[0]]
        matched: list[Any] = []
        defaults: list[Any] = []
        for edge in edges:
            md = getattr(edge, "metadata", {}) or {}
            pred_name = md.get("wf_predicate")
            if pred_name is None:
                if bool(md.get("wf_is_default", False)):
                    defaults.append(edge)
                continue
            pred = predicate_registry.get(str(pred_name))
            if pred is None:
                continue
            try:
                ok = bool(pred(edge, state, result))
            except Exception:
                ok = False
            if ok:
                matched.append(edge)
        picked = matched if matched else defaults
        if not picked:
            return []
        if fanout:
            return picked
        # Non-fanout: still allow explicit "many" edge fanout.
        many = [e for e in picked if (getattr(e, "metadata", {}) or {}).get("wf_multiplicity") == "many"]
        if many:
            return many
        return [picked[0]]

    async def _run_native_async(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: str | None,
        cache_dir: str | None,
    ) -> RunResult:
        state: WorkflowState = dict(initial_state)
        validate_initial_state(state)
        run_id = str(run_id or f"run|{uuid.uuid4()}")
        mq: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10000)

        start, nodes, adj = validate_workflow_design(
            workflow_engine=self._workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self._predicate_registry,
            resolver=self._raw_step_resolver,
        )

        incoming: dict[str, int] = {str(nid): 0 for nid in nodes.keys()}
        for _src, edges in adj.items():
            for e in list(edges or []):
                dst = str((getattr(e, "target_ids", None) or [None])[0] or "")
                if dst:
                    incoming[dst] = int(incoming.get(dst, 0)) + 1

        join_merge_nodes: set[str] = set()
        join_expected: dict[str, int] = {}
        join_waiters: dict[str, list[tuple[int, str, str | None]]] = {}
        for nid, node in nodes.items():
            md = getattr(node, "metadata", {}) or {}
            is_join = bool(md.get("wf_join", False)) or str(getattr(node, "op", "")) == "join"
            if not is_join or not bool(md.get("wf_join_is_merge", True)):
                continue
            sid = str(nid)
            join_merge_nodes.add(sid)
            join_expected[sid] = max(1, int(incoming.get(sid, 0)))
            join_waiters[sid] = []

        join_node_ids = sorted(join_merge_nodes)
        may_reach_join = (
            _compute_may_reach_join_bitsets(
                node_ids=list(nodes.keys()),
                adj=adj,
                join_ids=join_node_ids,
            )
            if join_node_ids
            else {str(nid): 0 for nid in nodes.keys()}
        )

        sem = asyncio.Semaphore(max(1, int(self.max_workers)))
        pending: list[tuple[int, int, str, str, str | None]] = [
            (0, int(may_reach_join.get(str(start.id), 0)), str(start.id), str(uuid.uuid4()), None)
        ]
        inflight: set[asyncio.Task] = set()
        inflight_records: dict[asyncio.Task, tuple[int, int, str, str, str | None]] = {}
        suspended_tokens: dict[str, tuple[str, int, str, str | None]] = {}
        seq = 0
        dispatch_seq = 0
        status: TerminalStatus = "succeeded"
        last_processed_node_id: str | None = None
        accepted_step_seq = -1
        next_accept_seq = 0
        ready_by_launch_seq: dict[int, tuple[int, int, str, str, str | None, StepRunResult]] = {}
        last_exec_node: Any | None = None
        cancel_requested = _CANCEL_REQUESTED_CTX.get() or self.cancel_requested

        def _cancel_requested() -> bool:
            return bool(cancel_requested and cancel_requested(str(run_id)))

        async def _run_one(item: tuple[int, int, str, str, str | None]):
            nonlocal seq
            launch_seq, mask, node_id, token_id, parent_token_id = item
            node = nodes[node_id]
            step_seq = seq
            seq += 1
            ctx = StepContext(
                run_id=run_id,
                workflow_id=str(workflow_id),
                workflow_node_id=str(node_id),
                op=str(node.op),
                token_id=str(token_id),
                attempt=1,
                step_seq=int(step_seq),
                conversation_id=str(conversation_id),
                turn_node_id=str(turn_node_id),
                state=state,
                message_queue=mq,
                events=None,
                cache_dir=cache_dir,
            )
            fn = self._resolve_async_step_fn(str(node.op))
            should_step_uow = getattr(self._sync_runtime, "_should_step_uow", None)
            maybe_step_uow = getattr(self._sync_runtime, "_maybe_step_uow", None)
            try:
                use_step_uow = bool(
                    callable(should_step_uow)
                    and should_step_uow(str(node.op), state)
                )
            except Exception:
                use_step_uow = False
            uow_ctx = (
                maybe_step_uow()
                if use_step_uow and callable(maybe_step_uow)
                else nullcontext()
            )
            async with sem:
                with uow_ctx:
                    out = await fn(ctx)
            return launch_seq, step_seq, mask, node_id, token_id, parent_token_id, out

        def _apply_run_result(target_state: WorkflowState, result: StepRunResult) -> None:
            apply_state_update_inplace(
                target_state,
                list(getattr(result, "state_update", []) or []),
                getattr(result, "update", None),
                state_schema=getattr(self._resolver_adapter, "_state_schema", None),
            )

        def _persist_rt_join_snapshot() -> None:
            state["_rt_join"] = {
                "join_node_ids": list(join_node_ids),
                "join_outstanding": [len(join_waiters.get(jid, [])) for jid in join_node_ids],
                "join_waiters": {
                    jid: [
                        (
                            int(mask),
                            str(token_id),
                            (str(parent_token_id) if parent_token_id is not None else None),
                        )
                        for mask, token_id, parent_token_id in join_waiters.get(jid, [])
                    ]
                    for jid in join_node_ids
                },
                "pending": [
                    (
                        str(nid),
                        int(mask),
                        str(token_id),
                        (str(parent_token_id) if parent_token_id is not None else None),
                    )
                    for launch_seq, mask, nid, token_id, parent_token_id in list(pending) + list(inflight_records.values())
                ],
                "suspended": [
                    (
                        str(nid),
                        int(mask),
                        str(token_id),
                        (str(parent_token_id) if parent_token_id is not None else None),
                    )
                    for nid, mask, token_id, parent_token_id in suspended_tokens.values()
                ],
            }

        def _persist_step_exec_compat(**kwargs: Any) -> Any:
            persist = getattr(self._sync_runtime, "_persist_step_exec", None)
            if callable(persist):
                return persist(**kwargs)
            return None

        def _persist_checkpoint_compat(**kwargs: Any) -> Any:
            persist = getattr(self._sync_runtime, "_persist_checkpoint", None)
            if callable(persist):
                return persist(**kwargs)
            return None

        async def _cancel_and_drain_inflight(tasks: set[asyncio.Task]) -> None:
            if not tasks:
                return
            for t in list(tasks):
                t.cancel()
            await asyncio.gather(*list(tasks), return_exceptions=True)
            tasks.clear()
            inflight_records.clear()

        while pending or inflight:
            if _cancel_requested():
                status = "cancelled"
                _persist_rt_join_snapshot()
                pending.clear()
                await _cancel_and_drain_inflight(inflight)
                break

            while pending and len(inflight) < max(1, int(self.max_workers)):
                item = pending.pop(0)
                task = asyncio.create_task(_run_one(item))
                inflight.add(task)
                inflight_records[task] = item

            if not inflight:
                break

            done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)

            if _cancel_requested():
                status = "cancelled"
                _persist_rt_join_snapshot()
                pending.clear()
                await _cancel_and_drain_inflight(inflight)
                break

            completed: list[tuple[int, int, int, str, str, str | None, StepRunResult]] = []
            for task in done:
                item = inflight_records.pop(task, None)
                if task.cancelled() or item is None:
                    continue
                try:
                    completed.append(task.result())
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    status = "failure"
                    pending.clear()
                    await _cancel_and_drain_inflight(inflight)
                    completed.append(
                        (
                            -1,
                            -1,
                            0,
                            "",
                            "",
                            None,
                            RunFailure(
                                conversation_node_id=None,
                                state_update=[],
                                errors=[str(exc)],
                            ),
                        )
                    )
                    break

            for item in completed:
                launch_seq = int(item[0])
                ready_by_launch_seq[launch_seq] = item

            while next_accept_seq in ready_by_launch_seq:
                _launch_seq, _step_seq, mask, node_id, token_id, parent_token_id, out = ready_by_launch_seq.pop(next_accept_seq)
                next_accept_seq += 1
                if not node_id:
                    continue
                accepted_step_seq = max(accepted_step_seq, int(_step_seq))
                last_processed_node_id = str(node_id)
                step_exec_node: Any | None = last_exec_node
                child_failed = False

                invocations = list(getattr(out, "workflow_invocations", []) or [])
                for invocation in invocations:
                    child_result = await self._run_workflow_invocation_async(
                        invocation=invocation,
                        parent_state=state,
                        conversation_id=str(conversation_id),
                        turn_node_id=str(turn_node_id),
                        parent_run_id=str(run_id),
                        cache_dir=cache_dir,
                    )
                    child_status = str(getattr(child_result, "status", None))
                    if child_status != "succeeded":
                        if child_status == "cancelled" and _cancel_requested():
                            status = "cancelled"
                            out = RunFailure(
                                conversation_node_id=None,
                                status="failure",
                                errors=["Run cancelled"],
                                state_update=[],
                            )
                        else:
                            status = "failure"
                            out = RunFailure(
                                conversation_node_id=None,
                                status="failure",
                                errors=[
                                    "Nested workflow "
                                    f"{getattr(invocation, 'workflow_id', '')!r} "
                                    f"returned status {child_status!r}"
                                ],
                                state_update=[],
                            )
                        pending.clear()
                        await _cancel_and_drain_inflight(inflight)
                        child_failed = True
                        break
                    self._apply_workflow_invocation_result(
                        state=state,
                        invocation=invocation,
                        child_result=child_result,
                    )
                if status != "succeeded":
                    _persist_rt_join_snapshot()
                    step_exec_node = _persist_step_exec_compat(
                        conversation_id=str(conversation_id),
                        workflow_id=str(workflow_id),
                        run_id=str(run_id),
                        step_seq=int(_step_seq),
                        workflow_node_id=str(node_id),
                        op=str(nodes[node_id].op),
                        status=(
                            "suspended"
                            if getattr(out, "status", None) == "suspended"
                            else "failure"
                            if getattr(out, "status", None) == "failure"
                            else "ok"
                        ),
                        duration_ms=0,
                        result=out,
                        state=state,
                        token_id=str(token_id),
                        parent_token_id=parent_token_id,
                        join_mask=int(mask),
                        last_exec_node=last_exec_node,
                    )
                    if child_failed:
                        _persist_checkpoint_compat(
                            conversation_id=str(conversation_id),
                            workflow_id=str(workflow_id),
                            run_id=str(run_id),
                            step_seq=int(_step_seq),
                            state=state,
                            last_exec_node=step_exec_node,
                        )
                    break

                _apply_run_result(state, out)

                step_exec_node = _persist_step_exec_compat(
                    conversation_id=str(conversation_id),
                    workflow_id=str(workflow_id),
                    run_id=str(run_id),
                    step_seq=int(_step_seq),
                    workflow_node_id=str(node_id),
                    op=str(nodes[node_id].op),
                    status=(
                        "suspended"
                        if getattr(out, "status", None) == "suspended"
                        else "failure"
                        if getattr(out, "status", None) == "failure"
                        else "ok"
                    ),
                    duration_ms=0,
                    result=out,
                    state=state,
                    token_id=str(token_id),
                    parent_token_id=parent_token_id,
                    join_mask=int(mask),
                    last_exec_node=last_exec_node,
                )
                last_exec_node = step_exec_node

                if getattr(out, "status", None) == "failure":
                    status = "failure"
                    pending.clear()
                    await _cancel_and_drain_inflight(inflight)
                    _persist_rt_join_snapshot()
                    _persist_checkpoint_compat(
                        conversation_id=str(conversation_id),
                        workflow_id=str(workflow_id),
                        run_id=str(run_id),
                        step_seq=int(_step_seq),
                        state=state,
                        last_exec_node=step_exec_node,
                    )
                    break

                if getattr(out, "status", None) == "suspended":
                    status = "suspended"
                    suspended_tokens[str(token_id)] = (str(node_id), int(mask), str(token_id), parent_token_id)
                    pending.clear()
                    await _cancel_and_drain_inflight(inflight)
                    _persist_rt_join_snapshot()
                    _persist_checkpoint_compat(
                        conversation_id=str(conversation_id),
                        workflow_id=str(workflow_id),
                        run_id=str(run_id),
                        step_seq=int(_step_seq),
                        state=state,
                        last_exec_node=step_exec_node,
                    )
                    break

                md = getattr(nodes[node_id], "metadata", {}) or {}
                if bool(md.get("wf_terminal", False)):
                    _persist_rt_join_snapshot()
                    _persist_checkpoint_compat(
                        conversation_id=str(conversation_id),
                        workflow_id=str(workflow_id),
                        run_id=str(run_id),
                        step_seq=int(_step_seq),
                        state=state,
                        last_exec_node=step_exec_node,
                    )
                    continue

                next_edges = self._select_next_edges(
                    nodes[node_id],
                    list(adj.get(node_id, [])),
                    state,
                    out,
                    self._predicate_registry,
                )
                for idx, e in enumerate(next_edges):
                    dst = str((getattr(e, "target_ids", None) or [None])[0])
                    if not dst:
                        continue

                    next_token = token_id if idx == 0 else str(uuid.uuid4())
                    next_parent = None if idx == 0 else token_id
                    next_mask = int(may_reach_join.get(dst, 0))

                    if idx > 0:
                        try:
                            mq.put_nowait(
                                {
                                    "type": "token.spawn",
                                    "parent_token_id": str(token_id),
                                    "child_token_id": str(next_token),
                                    "from_node_id": str(node_id),
                                    "to_node_id": str(dst),
                                }
                            )
                        except queue.Full:
                            pass

                    if dst in join_merge_nodes:
                        waiters = join_waiters.setdefault(dst, [])
                        waiters.append((int(next_mask), str(next_token), next_parent))
                        _persist_rt_join_snapshot()
                        if len(waiters) < int(join_expected.get(dst, 1)):
                            continue
                        merged_mask = int(waiters[0][0])
                        next_token, next_parent = waiters[0][1], waiters[0][2]
                        for wm, _tok, _parent in waiters[1:]:
                            merged_mask &= int(wm)
                        waiters.clear()
                        next_mask = merged_mask
                        _persist_rt_join_snapshot()

                    dispatch_seq += 1
                    pending.append((dispatch_seq, int(next_mask), str(dst), str(next_token), next_parent))

                _persist_rt_join_snapshot()
            _persist_checkpoint_compat(
                conversation_id=str(conversation_id),
                workflow_id=str(workflow_id),
                run_id=str(run_id),
                step_seq=int(accepted_step_seq if accepted_step_seq >= 0 else 0),
                state=state,
                last_exec_node=last_exec_node,
            )

            if status != "succeeded":
                break

        if status != "succeeded":
            await _cancel_and_drain_inflight(inflight)
            if status == "cancelled":
                persist_cancelled = getattr(self._sync_runtime, "_persist_cancelled_terminal", None)
                if callable(persist_cancelled):
                    try:
                        persist_cancelled(
                            conversation_id=str(conversation_id),
                            workflow_id=str(workflow_id),
                            run_id=str(run_id),
                            accepted_step_seq=int(accepted_step_seq),
                            cancel_info={
                                "source": "native_async",
                                "node_id": last_processed_node_id,
                                "seq": accepted_step_seq,
                                "watermark": None,
                            },
                            last_processed_node_id=last_processed_node_id,
                        )
                    except Exception:
                        pass
        if status != "cancelled":
            _persist_rt_join_snapshot()

        return RunResult(run_id=run_id, final_state=state, mq=mq, status=status)

    def run_sync(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | None = None,
        _resume_step_seq: int | None = None,
        _resume_last_exec_node: Any | None = None,
    ) -> RunResult:
        return self._sync_runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=initial_state,
            run_id=run_id,
            cache_dir=cache_dir,
            _resume_step_seq=_resume_step_seq,
            _resume_last_exec_node=_resume_last_exec_node,
        )

    async def resume_run(
        self,
        *,
        run_id: str,
        suspended_node_id: str,
        suspended_token_id: str,
        client_result: StepRunResult,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        cache_dir: str | None = None,
    ) -> RunResult:
        return self._sync_runtime.resume_run(
            run_id=run_id,
            suspended_node_id=suspended_node_id,
            suspended_token_id=suspended_token_id,
            client_result=client_result,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            cache_dir=cache_dir,
        )
