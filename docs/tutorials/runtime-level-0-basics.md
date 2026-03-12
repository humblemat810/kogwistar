# Runtime Level 0: WorkflowRuntime Basics

Goal: run the smallest useful workflow runtime example and inspect the native state updates it produces.

## Quick Run

```powershell
python scripts/runtime_tutorial_ladder.py reset `
  --data-dir .gke-data/runtime-tutorial-ladder

python scripts/runtime_tutorial_ladder.py level0 `
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"status": "suspended"`: the canonical workflow pauses on `branch_a`
- `"final_state"`: includes `started`, `fanout_seen`, and `branch_b_done`
- `"step_exec_count"` / `"checkpoint_count"`: persisted runtime artifacts
- `"trace_db_path"`: SQLite trace sink path, typically `.../workflow/wf_trace.sqlite`
- `"checkpoint_pass": true`

Example shape:

```json
{
  "level": 0,
  "status": "suspended",
  "final_state": {
    "started": true,
    "fanout_seen": true,
    "branch_b_done": true
  },
  "step_exec_count": 3,
  "checkpoint_count": 3,
  "trace_db_path": ".gke-data/runtime-tutorial-ladder/workflow/wf_trace.sqlite",
  "checkpoint_pass": true
}
```

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

## Troubleshooting

- If Chroma is missing: install with `pip install -e ".[chroma]"`.
- If old state is confusing the result: rerun `reset`.
- Run from repo root so the script can import the package and locate the bundled viewer asset.
