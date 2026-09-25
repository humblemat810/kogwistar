"""Deterministic model/tool bindings using existing resolver result contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from kogwistar.acl.derivation import (
    ACLInput,
    Declassifier,
    derive_acl,
    normalize_derivation_policy,
)
from kogwistar.runtime.budget import (
    BudgetEvent,
    BudgetExhaustedError,
    StateBackedBudgetLedger,
)
from kogwistar.runtime.models import (
    RunFailure,
    RunSuccess,
    WorkflowDesignArtifact,
    WorkflowInvocationRequest,
)
from kogwistar.runtime.resolvers import MappingStepResolver

from .limits import refresh_budget_hints
from .read_tools import AgentReadTools, ReadScope


class FakeModel(Protocol):
    def complete(self, prompt: str, context: Mapping[str, Any]) -> Any: ...


class FakeTool(Protocol):
    def invoke(self, arguments: Mapping[str, Any]) -> Any: ...


class SequenceFakeModel:
    """Deterministic model fixture; no network or provider dependency."""

    def __init__(self, payloads: Sequence[Any]) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []
        self.contexts: list[dict[str, Any]] = []

    def complete(self, prompt: str, context: Mapping[str, Any]) -> Any:
        self.prompts.append(str(prompt))
        self.contexts.append(dict(context))
        if not self._payloads:
            raise RuntimeError("fake model payload sequence exhausted")
        return self._payloads.pop(0)


class FunctionFakeTool:
    def __init__(self, function: Callable[[Mapping[str, Any]], Any]) -> None:
        self.function = function
        self.calls: list[dict[str, Any]] = []

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        payload = dict(arguments)
        self.calls.append(payload)
        return self.function(payload)


def _ledger(ctx: Any) -> StateBackedBudgetLedger:
    state = ctx._state
    budget_state = state.get("budget")
    if not isinstance(budget_state, dict):
        budget_state = state
    deps = state.setdefault("_deps", {})
    if not isinstance(deps, dict):
        raise ValueError("workflow state _deps must be a dict")
    ledger = deps.get("budget_ledger")
    if (
        not isinstance(ledger, StateBackedBudgetLedger)
        or ledger.state is not budget_state
    ):
        ledger = StateBackedBudgetLedger(budget_state)
        deps["budget_ledger"] = ledger
    return ledger


def _failure(ctx: Any, message: str) -> RunFailure:
    return RunFailure(
        conversation_node_id=None,
        state_update=[],
        errors=[message],
    )


def _trusted_acl_inputs(ctx: Any, key: str) -> list[Any]:
    """Read ACL descriptors from the runtime authority carrier only.

    Mutable workflow state may carry provenance for display, but cannot widen
    ACL. If no authority descriptor is present, any state-side descriptors are
    represented as private inputs and therefore fail closed.
    """
    authority = getattr(ctx, "authority_context", None)
    if isinstance(authority, Mapping):
        supplied = authority.get("acl_inputs")
        if supplied is not None:
            if isinstance(supplied, Mapping) or isinstance(supplied, ACLInput):
                return [supplied]
            return list(supplied)
    state_supplied = ctx.state_view.get(key)
    if state_supplied is None:
        return []
    if isinstance(state_supplied, Mapping) or isinstance(state_supplied, ACLInput):
        state_supplied = [state_supplied]
    return [
        ACLInput(
            object_id=str(item.get("object_id") or item.get("id") or "untrusted-state-acl"),
            mode="private",
            source_kind="untrusted_state_acl",
        )
        for item in state_supplied
        if isinstance(item, Mapping)
    ]


def register_model_step(
    resolver: MappingStepResolver,
    model: FakeModel,
    *,
    op: str = "agent.model_call",
    prompt_key: str = "agent_prompt",
    output_key: str = "agent_model_output",
    estimated_tokens: int | None = None,
    estimated_cost: float | None = None,
    acl_policy: str = "STRICT",
    declassifier: Declassifier | None = None,
    acknowledge_llm_guarded: bool = False,
    acl_inputs_key: str = "agent_acl_inputs",
    prompt_acl_mode: str = "private",
    prompt_object_id: str = "agent_prompt",
    output_acl_key: str = "agent_model_output_acl",
    output_provenance_key: str = "agent_model_output_provenance",
) -> None:
    """Register one model call; call/token ceilings are enforced before exposure."""
    selected_acl_policy = normalize_derivation_policy(acl_policy)
    if prompt_acl_mode not in {"private", "shared", "scope", "group", "public"}:
        raise ValueError("prompt_acl_mode must be private, shared, scope, group, or public")
    if selected_acl_policy == "LLM_GUARDED" and not acknowledge_llm_guarded:
        raise PermissionError("LLM_GUARDED requires explicit acknowledgement")

    @resolver.register(op)
    def _model(ctx: Any) -> RunSuccess | RunFailure:
        ledger = _ledger(ctx)
        if ledger.should_suspend_for_budget():
            refresh_budget_hints(ctx._state, ledger=ledger)
            return _failure(ctx, "budget exhausted before model dispatch")
        if estimated_tokens is not None and ledger.total and ledger.remaining < estimated_tokens:
            refresh_budget_hints(ctx._state, ledger=ledger)
            return _failure(ctx, "token budget insufficient for model estimate")
        if (
            estimated_cost is not None
            and ledger.cost_budget > 0
            and ledger.cost_budget - ledger.cost_used < estimated_cost
        ):
            refresh_budget_hints(ctx._state, ledger=ledger)
            return _failure(ctx, "cost budget insufficient for model estimate")
        try:
            ledger.debit_call(run_id=str(ctx.run_id), reason=op)
        except BudgetExhaustedError as exc:
            refresh_budget_hints(ctx._state, ledger=ledger)
            return _failure(ctx, str(exc))
        prompt = str(ctx.state_view.get(prompt_key, ""))
        acl_inputs = [
            ACLInput(object_id=prompt_object_id, mode=prompt_acl_mode, source_kind="prompt"),
            *_trusted_acl_inputs(ctx, acl_inputs_key),
        ]
        acl_result = derive_acl(
            acl_inputs,
            policy=selected_acl_policy,
            declassifier=declassifier,
            acknowledge_llm_guarded=acknowledge_llm_guarded,
            object_id=output_key,
            generation_id=str(ctx.run_id),
        )
        model_context: dict[str, Any] = {
            "budget": dict(ctx.state_view.get("agent_budget_hints") or {})
        }
        if selected_acl_policy == "LLM_GUARDED" or isinstance(
            getattr(ctx, "authority_context", None), Mapping
        ) and "acl_inputs" in getattr(ctx, "authority_context", {}):
            model_context["acl"] = {
                "mode": acl_result.final_mode,
                "source_ids": list(acl_result.source_ids),
                "policy": acl_result.policy,
            }
        try:
            payload = model.complete(prompt, model_context)
        except Exception as exc:
            refresh_budget_hints(ctx._state, ledger=ledger)
            return _failure(ctx, f"fake model failed: {exc}")
        usage = payload.get("usage") if isinstance(payload, Mapping) else None
        if isinstance(usage, Mapping):
            try:
                ledger.debit(
                    int(usage.get("total_tokens") or usage.get("output_tokens") or 0),
                    reason=op,
                    run_id=str(ctx.run_id),
                )
            except BudgetExhaustedError as exc:
                refresh_budget_hints(ctx._state, ledger=ledger)
                return _failure(ctx, str(exc))
            actual_cost = usage.get("total_cost")
            if actual_cost is not None:
                ledger.ingest(
                    BudgetEvent(
                        run_id=str(ctx.run_id),
                        source="agent",
                        kind="cost",
                        amount=float(actual_cost),
                        unit="total_cost",
                        meta={"reason": op},
                    )
                )
                if ledger.cost_budget and ledger.cost_used > ledger.cost_budget:
                    refresh_budget_hints(ctx._state, ledger=ledger)
                    return _failure(ctx, "cost budget exhausted by model usage")
        refresh_budget_hints(ctx._state, ledger=ledger)
        return RunSuccess(
            state_update=[
                (
                    "u",
                    {
                        output_key: payload,
                        output_acl_key: acl_result.to_metadata(),
                        output_provenance_key: acl_result.to_dict(),
                    },
                )
            ]
        )


def register_tool_step(
    resolver: MappingStepResolver,
    tool: FakeTool,
    *,
    required_capability: str,
    op: str = "agent.tool_call",
    arguments_key: str = "agent_tool_arguments",
    output_key: str = "agent_tool_output",
    max_attempts: int = 1,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    acl_policy: str = "STRICT",
    declassifier: Declassifier | None = None,
    acknowledge_llm_guarded: bool = False,
    acl_inputs_key: str = "agent_acl_inputs",
    output_acl_mode: str = "private",
    output_acl_key: str = "agent_tool_output_acl",
    output_provenance_key: str = "agent_tool_output_provenance",
) -> None:
    """Register bounded retry using caller-supplied capabilities only.

    Retries stay inside one ordinary step attempt.  Durable external effects
    still require an idempotent tool; this helper does not claim exactly once.
    """

    capability = str(required_capability).strip()
    if not capability:
        raise ValueError("required_capability must be non-empty")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive")
    if not retry_exceptions:
        raise ValueError("retry_exceptions must be non-empty")
    selected_acl_policy = normalize_derivation_policy(acl_policy)
    if output_acl_mode not in {"private", "shared", "scope", "group", "public"}:
        raise ValueError("output_acl_mode must be private, shared, scope, group, or public")
    if selected_acl_policy == "LLM_GUARDED" and not acknowledge_llm_guarded:
        raise PermissionError("LLM_GUARDED requires explicit acknowledgement")

    @resolver.register(op)
    def _tool(ctx: Any) -> RunSuccess | RunFailure:
        # Persisted workflow state is evidence, not authority.  A resolver
        # invoked without the runtime carrier must fail closed, including in
        # direct/native worker integrations.
        trusted = getattr(ctx, "authority_context", None)
        effective = (
            trusted.get("effective_capabilities", ())
            if isinstance(trusted, Mapping)
            else ()
        )
        if capability not in {str(item) for item in effective}:
            return _failure(ctx, f"capability denied: {capability}")
        arguments = ctx.state_view.get(arguments_key, {})
        if not isinstance(arguments, Mapping):
            return _failure(ctx, f"{arguments_key} must be an object")
        last_error: BaseException | None = None
        for attempt in range(1, int(max_attempts) + 1):
            try:
                result = tool.invoke(arguments)
                acl_result = derive_acl(
                    [
                        *_trusted_acl_inputs(ctx, acl_inputs_key),
                        ACLInput(
                            object_id=f"{op}:output",
                            mode=output_acl_mode,  # type: ignore[arg-type]
                            source_kind="tool_output",
                        ),
                    ],
                    policy=selected_acl_policy,
                    declassifier=declassifier,
                    acknowledge_llm_guarded=acknowledge_llm_guarded,
                    object_id=output_key,
                    generation_id=str(ctx.run_id),
                )
                return RunSuccess(
                    state_update=[
                        (
                            "u",
                            {
                                output_key: result,
                                "agent_tool_attempts": attempt,
                                output_acl_key: acl_result.to_metadata(),
                                output_provenance_key: acl_result.to_dict(),
                            },
                        )
                    ]
                )
            except retry_exceptions as exc:
                last_error = exc
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"agent_tool_attempts": int(max_attempts)})],
            errors=[f"fake tool failed after {max_attempts} attempts: {last_error}"],
        )


def register_catalog_search_step(
    resolver: MappingStepResolver,
    reads: AgentReadTools,
    scope: ReadScope,
    *,
    op: str = "agent.catalog_search",
    query_key: str = "agent_catalog_query",
    output_key: str = "agent_catalog_results",
) -> None:
    """Use Goal B descriptor-first catalog search from an ordinary step."""

    @resolver.register(op)
    def _search(ctx: Any) -> RunSuccess:
        query = str(ctx.state_view.get(query_key, ""))
        page = reads.catalog_search(query, scope=scope, limit=20)
        return RunSuccess(state_update=[("u", {output_key: page.to_dict()})])


def make_nested_invocation_handler(
    invocation_factory: Callable[[Any], WorkflowInvocationRequest],
) -> Callable[[Any], RunSuccess]:
    """Create an ordinary resolver handler returning one nested invocation."""

    def _invoke(ctx: Any) -> RunSuccess:
        request = validate_invocation_request(invocation_factory(ctx))
        return RunSuccess(state_update=[], workflow_invocations=[request])

    return _invoke


def validate_invocation_request(
    request: WorkflowInvocationRequest,
) -> WorkflowInvocationRequest:
    """Validate static/dynamic action identity before returning a step result."""

    if not str(request.workflow_id).strip():
        raise ValueError("workflow_id must be non-empty")
    if request.workflow_design is not None and str(
        request.workflow_design.workflow_id
    ) != str(request.workflow_id):
        raise ValueError("workflow_design.workflow_id must match workflow_id")
    return request


def static_invocation(workflow_id: str, *, result_state_key: str = "agent_action_result") -> Callable[[Any], WorkflowInvocationRequest]:
    def _factory(ctx: Any) -> WorkflowInvocationRequest:
        return validate_invocation_request(WorkflowInvocationRequest(
            workflow_id=workflow_id,
            result_state_key=result_state_key,
            invocation_key=f"{ctx.run_id}:{ctx.step_seq}:{workflow_id}",
        ))

    return _factory


def dynamic_invocation(
    design: WorkflowDesignArtifact, *, result_state_key: str = "agent_action_result"
) -> Callable[[Any], WorkflowInvocationRequest]:
    def _factory(ctx: Any) -> WorkflowInvocationRequest:
        return validate_invocation_request(WorkflowInvocationRequest(
            workflow_id=design.workflow_id,
            workflow_design=design,
            result_state_key=result_state_key,
            invocation_key=f"{ctx.run_id}:{ctx.step_seq}:{design.workflow_id}",
        ))

    return _factory
