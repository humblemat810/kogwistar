from __future__ import annotations

import json
import pathlib
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.runtime.budget import RateBudgetWindow
from kogwistar.runtime.models import RunSuccess, RunSuspended, WorkflowEdge, WorkflowNode
from kogwistar.runtime.runtime import StepContext, WorkflowRuntime


@dataclass
class FakeClock:
    now_ms: int = 0

    def advance(self, amount_ms: int) -> int:
        self.now_ms += int(amount_ms or 0)
        return self.now_ms


class FakeTokenWindow:
    def __init__(self, *, limit: int, window_ms: int, clock: FakeClock) -> None:
        self.clock = clock
        self.window = RateBudgetWindow(
            limit=int(limit or 0),
            used=0,
            window_ms=int(window_ms or 0),
            window_started_ms=int(clock.now_ms),
        )
        self._lock = threading.Lock()

    def remaining(self) -> int:
        with self._lock:
            return self.window.remaining(now_ms=self.clock.now_ms)

    def consume(self, amount: int) -> bool:
        with self._lock:
            try:
                self.window.debit(int(amount or 0), now_ms=self.clock.now_ms)
                return True
            except Exception:
                return False

    def next_refresh_ms(self) -> int | None:
        with self._lock:
            return self.window.next_refresh_ms()

    def pinned(self) -> bool:
        with self._lock:
            return self.window.is_pinned_until_refresh(now_ms=self.clock.now_ms)


class _TinyEmbeddingFunction:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in str(text))
            out.append([(seed % 13) / 13.0, (seed % 17) / 17.0, (seed % 19) / 19.0])
        return out


def _dummy_grounding() -> Grounding:
    sp = Span(
        collection_page_url="demo",
        document_page_url="demo",
        doc_id="demo",
        insertion_method="demo",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="demo"),
    )
    return Grounding(spans=[sp])


def _add_node(engine: GraphKnowledgeEngine, wf_id: str, node_id: str, op: str, *, start: bool = False, terminal: bool = False, fanout: bool = False, join: bool = False) -> None:
    engine.write.add_node(
        WorkflowNode(
            id=node_id,
            label=op,
            type="entity",
            doc_id=node_id,
            summary=op,
            properties={},
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_node",
                "workflow_id": wf_id,
                "wf_op": op,
                "wf_start": start,
                "wf_terminal": terminal,
                "wf_version": "v1",
                "wf_fanout": fanout,
                "wf_join": join,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _add_edge(engine: GraphKnowledgeEngine, wf_id: str, src: str, dst: str) -> None:
    engine.write.add_edge(
        WorkflowEdge(
            id=f"{src}->{dst}",
            label="wf_next",
            type="entity",
            doc_id=f"{src}->{dst}",
            summary="next",
            properties={},
            source_ids=[src],
            target_ids=[dst],
            source_edge_ids=[],
            target_edge_ids=[],
            relation="wf_next",
            mentions=[_dummy_grounding()],
            metadata={
                "entity_type": "workflow_edge",
                "workflow_id": wf_id,
                "wf_predicate": None,
                "wf_priority": 100,
                "wf_is_default": True,
                "wf_multiplicity": "one",
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
    )


def _build_branch_workflow(engine: GraphKnowledgeEngine) -> None:
    wf_id = "budget_branch_pin_demo"
    _add_node(engine, wf_id, "start", "start", start=True, fanout=True)
    _add_node(engine, wf_id, "heavy_1", "heavy_1")
    _add_node(engine, wf_id, "heavy_2", "heavy_2")
    _add_node(engine, wf_id, "light", "light")
    _add_node(engine, wf_id, "join", "join", join=True)
    _add_node(engine, wf_id, "end", "end", terminal=True)
    _add_edge(engine, wf_id, "start", "heavy_1")
    _add_edge(engine, wf_id, "start", "light")
    _add_edge(engine, wf_id, "heavy_1", "heavy_2")
    _add_edge(engine, wf_id, "heavy_2", "join")
    _add_edge(engine, wf_id, "light", "join")
    _add_edge(engine, wf_id, "join", "end")


def _build_engine(base_dir: pathlib.Path, graph_type: str) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(base_dir),
        kg_graph_type=graph_type,
        embedding_function=_TinyEmbeddingFunction(),
        backend_factory=build_in_memory_backend,
    )


def _latest_checkpoint_state(conv_engine: GraphKnowledgeEngine, run_id: str) -> dict[str, Any]:
    ckpts = conv_engine.read.get_nodes(
        where={"$and": [{"entity_type": "workflow_checkpoint"}, {"run_id": run_id}]}
    )
    latest = max(
        ckpts, key=lambda c: int(getattr(c, "metadata", {}).get("step_seq", -1))
    )
    state_json = latest.metadata.get("state_json", {})
    if isinstance(state_json, str):
        state_json = json.loads(state_json)
    return dict(state_json or {})


def run_budget_branch_pin_demo() -> dict[str, Any]:
    clock = FakeClock(now_ms=1000)
    gate = FakeTokenWindow(limit=2, window_ms=10_000, clock=clock)
    timeline: list[dict[str, Any]] = []
    order: list[str] = []
    lock = threading.Lock()
    heavy_ready = threading.Event()
    light_done = threading.Event()
    heavy_done = threading.Event()

    def _record(event: str, **extra: Any) -> None:
        with lock:
            timeline.append(
                {
                    "event": event,
                    "ts_ms": int(clock.now_ms),
                    "remaining": gate.remaining(),
                    **extra,
                }
            )

    def _heavy_1(ctx: StepContext):
        if not gate.consume(2):
            raise AssertionError("heavy branch should have enough tokens at step 1")
        with ctx.state_write as state:
            state.setdefault("op_log", []).append("heavy_1")
        _record("heavy.1.finished")
        order.append("heavy_1")
        heavy_ready.set()
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"heavy_1": True})])

    def _heavy_2(ctx: StepContext):
        _record("heavy.2.started")
        if not gate.consume(1):
            _record("heavy.2.paused")
            return RunSuspended(
                conversation_node_id=None,
                state_update=[],
                resume_payload={
                    "type": "recoverable_error",
                    "op": "heavy_2",
                    "category": "token_window_exhausted",
                    "message": "token window pinned until refresh",
                    "errors": ["token window exhausted"],
                    "repair_payload": {
                        "next_refresh_ms": gate.next_refresh_ms(),
                    },
                },
            )
        with ctx.state_write as state:
            state.setdefault("op_log", []).append("heavy_2")
        _record("heavy.2.finished")
        order.append("heavy_2")
        heavy_done.set()
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"heavy_2": True})])

    def _light(ctx: StepContext):
        with ctx.state_write as state:
            state.setdefault("op_log", []).append("light")
        _record("light.finished")
        order.append("light")
        light_done.set()
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"light": True})])

    def _join(ctx: StepContext):
        with ctx.state_write as state:
            state.setdefault("op_log", []).append("join")
        _record("join.finished")
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"joined": True})])

    from kogwistar.runtime.resolvers import MappingStepResolver

    resolver = MappingStepResolver()
    resolver.register("heavy_1")(_heavy_1)
    resolver.register("heavy_2")(_heavy_2)
    resolver.register("light")(_light)
    resolver.register("join")(_join)
    resolver.register("start")(lambda ctx: RunSuccess(conversation_node_id=None, state_update=[("u", {"started": True})]))
    resolver.register("end")(
        lambda ctx: RunSuccess(conversation_node_id=None, state_update=[("u", {"ended": True})])
    )

    base = pathlib.Path.cwd() / ".tmp_budget_branch_pin_demo" / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    wf_engine = _build_engine(base / "wf", "workflow")
    conv_engine = _build_engine(base / "conv", "conversation")
    _build_branch_workflow(wf_engine)
    runtime = WorkflowRuntime(
        workflow_engine=wf_engine,
        conversation_engine=conv_engine,
        step_resolver=resolver,
        predicate_registry={},
        checkpoint_every_n_steps=1,
        max_workers=2,
    )

    run_id = "run_budget_branch_pin"
    conv_id = "conv_budget_branch_pin"
    initial_state = {
        "conversation_id": conv_id,
        "user_id": "user-budget",
        "turn_node_id": "turn_1",
        "turn_index": 0,
        "role": "user",
        "user_text": "",
        "mem_id": "mem_1",
        "budget": {
            "token_budget": 2,
            "token_used": 0,
            "time_budget_ms": 0,
            "time_used_ms": 0,
            "cost_budget": 0.0,
            "cost_used": 0.0,
            "budget_kind": "token",
            "budget_scope": "run",
            "rate_limit": 2,
            "rate_used": 0,
            "rate_window_ms": 10_000,
            "rate_window_started_ms": clock.now_ms,
        },
    }

    res1 = runtime.run(
        workflow_id="budget_branch_pin_demo",
        conversation_id=conv_id,
        turn_node_id="turn_1",
        initial_state=initial_state,  # type: ignore[arg-type]
        run_id=run_id,
    )
    assert res1.status == "suspended"
    assert heavy_ready.is_set()
    assert light_done.is_set()
    pinned_before_refresh = gate.pinned() is True
    assert pinned_before_refresh is True
    wait_until = gate.next_refresh_ms()
    if wait_until is None:
        raise AssertionError("missing refresh bound")
    _record("budget.pinned", wait_until=wait_until)

    resume_blocked_before_refresh = clock.now_ms < wait_until
    if resume_blocked_before_refresh:
        _record("resume.blocked_before_refresh")

    clock.advance(wait_until - clock.now_ms + 1)
    _record("token_window.refreshed", wait_until=wait_until)

    state1 = _latest_checkpoint_state(runtime.conversation_engine, run_id)
    suspended_token_id = state1["_rt_join"]["suspended"][0][2]

    res2 = runtime.resume_run(
        run_id=run_id,
        suspended_node_id="heavy_2",
        suspended_token_id=suspended_token_id,
        client_result=RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"heavy_2_resumed": True})],
        ),
        workflow_id="budget_branch_pin_demo",
        conversation_id=conv_id,
        turn_node_id="turn_1",
    )
    assert res2.status == "succeeded"
    order.append("heavy_2")
    heavy_done.set()
    _record("heavy.2.resumed")

    return {
        "order": order,
        "timeline": timeline,
        "result": {
            "branch_pinned_until_refresh": pinned_before_refresh,
            "resume_blocked_before_refresh": resume_blocked_before_refresh,
            "heavy_after_light": order.index("light") < order.index("heavy_2"),
        },
        "resume": {
            "first": res1.status,
            "second": res2.status,
        },
    }


def main() -> None:
    print(json.dumps(run_budget_branch_pin_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
