# Runtime Level 2: Pause, Checkpoint, and Resume

Goal: inspect the suspended token, then continue the same run with `resume_run(...)`.

## Quick Run

```powershell
python scripts/runtime_tutorial_ladder.py level2 `
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"suspended_node_id"` and `"suspended_token_id"`: identify the paused branch
- `"resume_payload"`: payload emitted by `RunSuspended`
- `"checkpoint_step_seqs"`: confirms checkpoints were saved during the run
- `"final_state"`: includes `branch_a_result`, `joined`, and `ended`
- `"checkpoint_pass": true`

## What This Level Teaches

- `RunSuspended` pauses execution without losing scheduler state.
- Checkpoint persistence stores `_rt_join` frontier data in `workflow_checkpoint` nodes.
- `WorkflowRuntime.resume_run(...)` continues a parked token after the client supplies a `RunSuccess`.
- Suspend/resume parity belongs to the native runtime. The LangGraph converter does not replace this behavior.

## Walkthrough

1. Run the canonical workflow until `branch_a` suspends.
2. Read the latest checkpoint and extract `_rt_join.suspended`.
3. Build a client result, here a `RunSuccess` with `branch_a_result`.
4. Call `WorkflowRuntime.resume_run(...)` with the run id, suspended node id, suspended token id, and client result.

## Checkpoint

Pass when:

- the initial run status is `"suspended"`
- the resumed run status is `"succeeded"`
- the resumed state reaches `join` and `end`

## Troubleshooting

- If no suspended token is found, the workflow did not pause where expected.
- If resume fails, confirm the run id and suspended token id came from the latest checkpoint for that run.
