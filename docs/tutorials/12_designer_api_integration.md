# 12 Designer API Integration

Audience: Builder / integrator
Time: 15-20 minutes

## What You Will Build

You will build a working map of the repo's external integration surfaces: workflow design, runtime capability discovery, and UI-facing API affordances.

## Why This Matters

Integrators need to know what can be discovered dynamically and what should remain native to the runtime. This page keeps that boundary explicit.

## Run or Inspect

- Inspect `kogwistar/conversation/designer.py` for workflow assembly helpers.
- Inspect `kogwistar/server/chat_service.py` and `kogwistar/server/runtime_api.py` for externally visible runtime and conversation surfaces.
- Compare with [06 First Workflow](./06_first_workflow.md) and [07 Branch Join Workflows](./07_branch_join_workflows.md) so the API discussion stays grounded in actual execution.

## Inspect The Result

- Identify which workflow capabilities can be discovered from the design/runtime surfaces.
- Notice that runtime state shape is intentional, especially around checkpoints and traces.
- Notice where the repo leaves room for future user-defined operations without pretending all behavior is already generic.

## Invariant Demonstrated

External tooling can discover supported capabilities without replacing the native runtime as the source of truth for execution semantics.

## Next Tutorial

Continue to [13 How to Test This Repo](./13_how_to_test_this_repo.md).
