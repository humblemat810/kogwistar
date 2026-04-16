# 06 First Workflow

Audience: Builder / integrator
Time: 15-20 minutes
Companion: [scripts/tutorial_sections/06_first_workflow.py](../../scripts/tutorial_sections/06_first_workflow.py)

## What You Will Build

You will run the smallest useful workflow example, inspect its state updates, and confirm checkpoint artifacts are persisted.

## Why This Matters

The runtime is a core subsystem, not an accessory. This tutorial introduces it with the smallest example that still proves deterministic execution and persisted scheduler state.

## Run or Inspect

- Run the companion file or the script-backed level:

```bash
python scripts/runtime_tutorial_ladder.py reset --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level0 --data-dir .gke-data/runtime-tutorial-ladder
```

- Follow up with [Runtime Level 0 - WorkflowRuntime Basics](./runtime-level-0-basics.md).

## Inspect The Result

- Confirm the run status is `suspended`.
- Inspect the persisted `workflow_step_exec` and `workflow_checkpoint` counts.
- Inspect the trace DB path reported by the tutorial output.

## Invariant Demonstrated

Workflow execution is persisted, not ephemeral. The runtime records enough state to inspect where a run stopped and what state changes had already landed.

## Next Tutorial

Continue to [07 Branch Join Workflows](./07_branch_join_workflows.md).
