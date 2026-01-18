from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union


Json = Any

# Import your real RunResult types from runtime/models
from graph_knowledge_engine.workflow.runtime import RunFailure, RunResult, RunSuccess, StepContext  # adjust import path

RawStepFn = Callable[[StepContext], Union[Json, RunResult]]

@dataclass
class MappingStepResolver:
    handlers: Dict[str, RawStepFn]
    default: Optional[RawStepFn] = None

    def __init__(self, handlers: Optional[Mapping[str, RawStepFn]] = None, *, default: Optional[RawStepFn] = None) -> None:
        self.handlers = dict(handlers or {})
        self.default = default

    def register(self, op: str) -> Callable[[RawStepFn], RawStepFn]:
        def _decorator(fn: RawStepFn) -> RawStepFn:
            self.handlers[op] = fn
            return fn
        return _decorator

    def resolve(self, op: str) -> Callable[[StepContext], RunResult]:
        raw = self.handlers.get(op) or self.default
        if raw is None:
            raise KeyError(f"No step handler registered for op={op!r}")

        def _wrapped(ctx: StepContext) -> RunResult:
            try:
                out = raw(ctx)
                if isinstance(out, (RunSuccess, RunFailure)):
                    return out
                else:
                    raise TypeError("Resolver must return RunResult")
                # return out  # field name might be `data`/`payload` in your codebase
            except Exception as e:
                return RunFailure(errors=[str(e)])  # match your RunFailure fields
        return _wrapped

    def __call__(self, op: str) -> Callable[[StepContext], RunResult]:
        return self.resolve(op)
    
    
default_resolver = MappingStepResolver()


@default_resolver.register("start")
def _start(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("start")
    ctx.state["started"] = True
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"started": True}])

@default_resolver.register("memory_retrieve")
def _memory_retrieve(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("memory_retrieve")
    ctx.state["memory"] = {"selected_ids": ["m1"], "text": "memory context"}
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"ok": True}])
@default_resolver.register("kg_retrieve")
def _kg_retrieve(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("kg_retrieve")
    ctx.state["kg"] = {"selected_ids": ["k1"], "facts": ["f1"]}
    # return {"ok": True}
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"ok": True}])
@default_resolver.register("memory_pin")
def _memory_pin(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("memory_pin")
    ctx.state["memory_pin"] = {"pinned_ids": ["m1"]}
    # return {"ok": True}
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"ok": True}])
@default_resolver.register("kg_pin")
def _kg_pin(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("kg_pin")
    ctx.state["kg_pin"] = {"pinned_ids": ["k1"]}
    # return {"ok": True}
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"ok": True}])
@default_resolver.register("answer")
def _answer(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("answer")
    ctx.state["answer"] = {"text": "answer text", "llm_decision_need_summary": True}
    # return ctx.state["answer"]
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"answer": ctx.state["answer"]}])
@default_resolver.register("decide_summarize")
def _decide(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("decide_summarize")
    need = bool(ctx.state.get("answer", {}).get("llm_decision_need_summary"))
    ctx.state["decide"] = {"need_summary": need}
    # return ctx.state["decide"]
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[ctx.state["decide"]])
@default_resolver.register("summarize")
def _summarize(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("summarize")
    ctx.state["summary"] = {"text": "summary text"}
    # return ctx.state["summary"]
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"ok": True}])
@default_resolver.register("end")
def _end(ctx: StepContext):
    ctx.state.setdefault("op_log", []).append("end")
    # return {"done": True}
    conversation_node_id_created_during_process = None
    return RunSuccess(conversation_node_id=conversation_node_id_created_during_process, 
                        outputs=[{"done": True}])    