# AI Runtime Workflow Guide

This is the short version to give an AI before asking it to write or edit native `WorkflowRuntime` workflows.

## What The Runtime Expects

- Workflow nodes and edges live in the workflow graph.
- Step handlers are resolved by `MappingStepResolver`.
- A handler must return one of:
  - `RunSuccess(...)`
  - `RunFailure(...)`
  - `RunSuspended(...)`
- Do not return raw dicts for native runtime handlers.

Preferred success shape:

```python
return RunSuccess(
    conversation_node_id=None,
    state_update=[],
)
```

## Routing Rules

- `_route_next` is the alias field for `next_step_names`.
- Use `_route_next` when the step wants to explicitly choose downstream nodes.
- A single `_route_next` target means route to that target.
- Multiple `_route_next` targets mean explicit fanout.
- Prefer destination node ids or the node's short suffix such as `"left"` for `wf|my_flow|left`.

Good:

```python
return RunSuccess(
    conversation_node_id=None,
    state_update=[],
    _route_next=["left", "right"],
)
```

Avoid:

```python
# Edge labels are not the thing to route to.
_route_next=["wf_next"]
```

## Fanout And Join

- If multiple branches should run, return multiple `_route_next` targets.
- If multiple static edges may fire through graph semantics, mark the node with `wf_fanout=True` or use edge multiplicity `"many"` where appropriate.
- If converging branches must execute the downstream step exactly once, add an explicit join node.
- Join nodes are declared with `wf_join=True` or `op == "join"`.
- Without a join node, converging branches can execute the same downstream node more than once.

Practical rule:

- Fan out freely.
- Join explicitly.
- Keep branch state writes disjoint unless you intentionally want reducer-style merging.

## Nested Workflow Invocation

- Use `workflow_invocations=[WorkflowInvocationRequest(...)]` from a `RunSuccess`.
- If you synthesize a child design on the fly, provide `workflow_design=...`.
- Put the child result in `result_state_key`.
- Child failure propagates up and fails the parent run.

Example:

```python
return RunSuccess(
    conversation_node_id=None,
    state_update=[],
    workflow_invocations=[
        WorkflowInvocationRequest(
            workflow_id="wf_child",
            workflow_design=child_design,
            result_state_key="child_result",
        )
    ],
    _route_next=["end"],
)
```

## State Rules

- Use `ctx.state_write` for direct in-place state mutation when needed.
- Use `state_update` for normal runtime-applied updates.
- Do not write runtime-owned keys such as `_rt_join`.
- `_deps` is dependency injection plumbing, not normal workflow state.
- Prefer stable, explicit state keys like `left_seen`, `child_result`, `answer_text`.

## Failure And Suspension

- Use `RunFailure(...)` for a structured step failure.
- Use `RunSuspended(...)` when a client or human must resume the run later.
- Do not raise exceptions for expected business outcomes if you want workflow-level control over the result.
- Raised exceptions are treated as resolver/runtime failures and typically become `RunFailure`.

## Minimal Example

```python
resolver = MappingStepResolver()

@resolver.register("start")
def _start(ctx):
    return RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["left", "right"],
    )

@resolver.register("left")
def _left(ctx):
    with ctx.state_write as st:
        st["left_seen"] = True
    return RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["join"],
    )

@resolver.register("right")
def _right(ctx):
    with ctx.state_write as st:
        st["right_seen"] = True
    return RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["join"],
    )

@resolver.register("join")
def _join(ctx):
    return RunSuccess(
        conversation_node_id=None,
        state_update=[],
        _route_next=["end"],
    )

@resolver.register("end")
def _end(ctx):
    with ctx.state_write as st:
        st["ended"] = True
    return RunSuccess(conversation_node_id=None, state_update=[])
```

Workflow design intent for that example:

- `start` routes to `left` and `right`
- `join` should be an explicit join node in graph metadata
- `end` is terminal

## Prompt Snippet For AI

Use this when asking an AI to generate a runtime workflow:

```text
Write native kogwistar WorkflowRuntime code, not LangGraph code.
Handlers must return RunSuccess, RunFailure, or RunSuspended.
Use conversation_node_id=None unless the step creates conversation graph nodes.
Use _route_next with downstream node ids or short node suffixes, never edge labels.
If multiple _route_next targets are returned, that is intentional fanout.
Add explicit join nodes where converging branches must execute once.
Use WorkflowInvocationRequest for nested workflows.
Do not mutate _rt_join. Do not treat _deps as normal business state.
```

## Read Next

- [runtime-level-0-basics.md](./tutorials/runtime-level-0-basics.md)
- [07_branch_join_workflows.md](./tutorials/07_branch_join_workflows.md)
- [runtime-level-2-pause-resume.md](./tutorials/runtime-level-2-pause-resume.md)
- [ARD-workflowruntime-token-nesting.md](../kogwistar/docs/ARD-workflowruntime-token-nesting.md)
- [ARD-wisdom-layer-and-dynamic-workflow-orchestration.md](../kogwistar/docs/ARD-wisdom-layer-and-dynamic-workflow-orchestration.md)
