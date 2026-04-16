# Runtime Level 0: WorkflowRuntime Basics

Goal: run the smallest useful workflow runtime example and inspect the native state updates it produces.

## What You Will Build

You will execute the canonical workflow until its first suspension point and inspect the persisted run, step, checkpoint, and trace artifacts.

## Why This Matters

This is the minimal proof that the runtime is real. It demonstrates that execution state is durable and inspectable before introducing resolver customization or resume logic.

## Run or Inspect

## Quick Run

```bash
python scripts/runtime_tutorial_ladder.py reset \
  --data-dir .gke-data/runtime-tutorial-ladder

python scripts/runtime_tutorial_ladder.py level0 \
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"status": "suspended"`: the canonical workflow pauses on `branch_a`
- `"final_state"`: includes `started`, `fanout_seen`, and `branch_b_done`
- `"step_exec_count"` / `"checkpoint_count"`: persisted runtime artifacts
- `"trace_db_path"`: SQLite trace sink path
- `"checkpoint_pass": true`

## Inspect The Result

- Confirm the run stopped at the suspension point instead of running to silent completion.
- Confirm `branch_b` already completed while `branch_a` is parked.
- Inspect the trace DB and checkpoint counts before moving on to resume logic.

## What This Level Teaches

- `WorkflowRuntime.run(...)` executes workflow nodes stored in the workflow graph.
- `MappingStepResolver.register(...)` maps `wf_op` values to Python handlers.
- Resolver handlers return `RunSuccess` or `RunSuspended`, not raw dicts.
- `state_update` drives the workflow state machine and checkpoint snapshots.

## Canonical Example

This track reuses one workflow throughout:

`start -> fork -> branch_a(suspends) + branch_b(completes) -> join -> end`

Level 0 intentionally stops at the first suspension so you can inspect the state and persisted traces before introducing resume logic.

## Checkpoint

Pass when:

- the run suspends at `branch_a`
- `branch_b` already completed
- at least one `workflow_step_exec` node and one `workflow_checkpoint` node were persisted

## Invariant Demonstrated

Workflow state is persisted as it executes. Suspension is a first-class state, not a test-only special case.

## Troubleshooting

- If Chroma is missing: install with `pip install -e ".[chroma]"`.
- If old state is confusing the result: rerun `reset`.
- Run from repo root so the script can import the package and locate the bundled viewer asset.

## Next Tutorial

Continue to [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md) or return to [06 First Workflow](./06_first_workflow.md).
