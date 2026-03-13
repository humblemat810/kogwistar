# 07 Branch Join Workflows

Audience: Builder / integrator
Time: 20 minutes
Companion: [scripts/tutorial_sections/07_branch_join_workflows.py](../../scripts/tutorial_sections/07_branch_join_workflows.py)

## What You Will Build

You will run the canonical branch/join workflow, inspect resolver telemetry, and resume a suspended branch to completion.

## Why This Matters

This is the point where the runtime starts to feel like a real orchestration substrate. Fanout, join release, and suspend/resume are the behaviors people usually distrust until they can inspect them.

## Run or Inspect

- Run the companion file to walk through resolver registration, trace events, and resume.
- Or run the script-backed levels:

```powershell
python scripts/runtime_tutorial_ladder.py level1 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level2 --data-dir .gke-data/runtime-tutorial-ladder
```

- Use [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md) and [Runtime Level 2 - Pause and Resume](./runtime-level-2-pause-resume.md) as the execution references.

## Inspect The Result

- Confirm custom resolver events appear in the trace sink.
- Confirm `_deps` is visible to handlers via `dep_echo`.
- Confirm the resumed run reaches `joined` and `ended`.

## Invariant Demonstrated

Branching, joins, and human-in-the-loop resume preserve deterministic state transitions instead of mutating hidden in-memory control flow.

## Next Tutorial

Continue to [08 Storage Backends and Parity](./08_storage_backends_and_parity.md).
