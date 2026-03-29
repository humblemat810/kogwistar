from __future__ import annotations

import json
import os
import time
from typing import Any

from kogwistar.runtime.models import RunSuccess
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.server.chat_service import RuntimeRunRequest


def _predicate_always_false(_workflow_info, _state, _last_result) -> bool:
    return False


def _debug_log(record: dict[str, Any]) -> None:
    log_path = str(os.getenv("KOGWISTAR_RUNTIME_SSE_DEBUG_LOG") or "").strip()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
    except Exception:
        pass


def make_looping_sleep_runtime_runner():
    def _runner(req: RuntimeRunRequest) -> dict[str, Any]:
        resolver = MappingStepResolver()

        @resolver.register("start")
        def _start(ctx):
            with ctx.state_write as state:
                state["sleep_ticks"] = int(ctx.state_view.get("sleep_ticks") or 0)
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    ("u", {"sleep_ticks": int(ctx.state_view.get("sleep_ticks") or 0)})
                ],
            )

        @resolver.register("sleep")
        def _sleep(ctx):
            deps = dict(ctx.state_view.get("_deps") or {})
            publish = deps.get("publish")
            tick = int(ctx.state_view.get("sleep_ticks") or 0) + 1
            if callable(publish):
                _debug_log(
                    {
                        "ts_ms": int(time.time() * 1000),
                        "stage": "runner_publish_before",
                        "run_id": req.run_id,
                        "workflow_id": ctx.workflow_id,
                        "workflow_node_id": ctx.workflow_node_id,
                        "tick": tick,
                    }
                )
                publish(
                    "sleep.tick",
                    {
                        "workflow_id": ctx.workflow_id,
                        "workflow_node_id": ctx.workflow_node_id,
                        "tick": tick,
                    },
                )
                _debug_log(
                    {
                        "ts_ms": int(time.time() * 1000),
                        "stage": "runner_publish_after",
                        "run_id": req.run_id,
                        "workflow_id": ctx.workflow_id,
                        "workflow_node_id": ctx.workflow_node_id,
                        "tick": tick,
                    }
                )
            time.sleep(1.0)
            with ctx.state_write as state:
                state["sleep_ticks"] = tick
            return RunSuccess(
                conversation_node_id=None,
                state_update=[("u", {"sleep_ticks": tick})],
            )

        @resolver.register("end")
        def _end(ctx):
            return RunSuccess(
                conversation_node_id=None,
                state_update=[
                    (
                        "u",
                        {
                            "done": True,
                            "sleep_ticks": int(ctx.state_view.get("sleep_ticks") or 0),
                        },
                    )
                ],
            )

        initial_state = dict(req.initial_state or {})
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("publish", req.publish)
        initial_state["_deps"] = deps

        def _cancel_requested(run_id: str) -> bool:
            _ = run_id
            return req.is_cancel_requested()

        runtime = WorkflowRuntime(
            workflow_engine=req.workflow_engine,
            conversation_engine=req.conversation_engine,
            step_resolver=resolver,
            predicate_registry={"always_false": _predicate_always_false},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=_cancel_requested,
        )
        run_result = runtime.run(
            workflow_id=req.workflow_id,
            conversation_id=req.conversation_id,
            turn_node_id=req.turn_node_id,
            initial_state=initial_state,
            run_id=req.run_id,
        )
        final_state = dict(getattr(run_result, "final_state", {}) or {})
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(getattr(run_result, "status", "succeeded") or "succeeded"),
            "final_state": final_state,
        }

    return _runner
