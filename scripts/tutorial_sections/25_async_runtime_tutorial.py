# %% [markdown]
# # 25 Async Runtime Basics
# Notebook-style companion for the async runtime tutorial.
#
# This file shows the async runtime contract in three small steps:
# 1. an async resolver handler
# 2. a `StepContext` call
# 3. `AsyncWorkflowRuntime` construction

# %%
import asyncio

from _helpers import banner, show
from kogwistar.runtime import AsyncMappingStepResolver, AsyncWorkflowRuntime, StepContext
from kogwistar.runtime.models import RunSuccess

# %%
banner("Define async resolver.")

resolver = AsyncMappingStepResolver()


@resolver.register("hello_async")
async def hello_async(ctx: StepContext):
    return RunSuccess(
        conversation_node_id=None,
        state_update=[
            ("u", {"hello": "async"}),
            ("a", {"run_id": ctx.run_id, "step_seq": ctx.step_seq}),
        ],
    )


ctx = StepContext(
    run_id="run-async-tutorial",
    workflow_id="wf-async-tutorial",
    workflow_node_id="node-async-tutorial",
    op="hello_async",
    token_id="tok-async-tutorial",
    attempt=1,
    step_seq=1,
    cache_dir=None,
    state={},
)

result = asyncio.run(resolver.resolve_async("hello_async")(ctx))
show(
    "resolver call",
    {
        "ops": sorted(resolver.ops),
        "status": result.status,
        "state_update": result.state_update,
    },
)

# %%
banner("Construct async runtime contract.")

runtime = AsyncWorkflowRuntime(
    workflow_engine={"name": "demo-workflow-engine"},
    conversation_engine={"name": "demo-conversation-engine"},
    step_resolver=resolver,
    predicate_registry={},
    trace=False,
)

show(
    "runtime contract",
    {
        "runtime_class": type(runtime).__name__,
        "sync_runtime_class": type(runtime.sync_runtime).__name__,
        "workflow_engine": runtime.sync_runtime.workflow_engine,
        "conversation_engine": runtime.sync_runtime.conversation_engine,
        "step_context_type": runtime.step_context_type.__name__,
        "step_result_type": str(runtime.step_result_type),
    },
)

# %%
# Invariant:
# - async resolver returns same StepRunResult shape
# - async runtime wraps sync runtime contract
# - execution style changes, workflow meaning does not
