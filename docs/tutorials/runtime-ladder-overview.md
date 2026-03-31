# Runtime Ladder Overview

Goal: give one compact entrypoint for the runtime tutorial track so a human or LLM can understand the flow before reading the level pages.

If you are new to the repo, read [06 First Workflow](./06_first_workflow.md), [07 Branch Join Workflows](./07_branch_join_workflows.md), and [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) first. Come back here when you want the script-backed runtime ladder as a compact execution index.

## Canonical Workflow

All runtime levels use the same workflow:

`start -> fork -> branch_a(suspends) + branch_b(completes) -> join -> end`

This single example is enough to demonstrate:

- resolver registration and dispatch
- dependency injection through `_deps`
- pause and continue via `RunSuspended` and `resume_run(...)`
- persisted checkpoints and step execution nodes
- trace events and CDC viewer integration
- LangGraph export as interoperability, not as the source of truth

## One-Time Setup

```powershell
python scripts/runtime_tutorial_ladder.py reset --data-dir .gke-data/runtime-tutorial-ladder
```

## Run Order

```powershell
python scripts/runtime_tutorial_ladder.py level0 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level1 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level2 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level3 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level4 --data-dir .gke-data/runtime-tutorial-ladder
```

## Levels

- Level 0: prove the native `WorkflowRuntime` executes the workflow and stops at the suspend point.
- Level 1: show how `MappingStepResolver` handlers consume `StepContext`, `_deps`, and `StepContext.events`.
- Level 2: recover the suspended token from checkpoint state and continue the run with `WorkflowRuntime.resume_run(...)`.
- Level 3: inspect runtime trace events, connect them to the CDC viewer, and export the workflow to LangGraph.
- Level 4: treat LLM-generated code as untrusted and execute a sandboxed workflow op inside Docker.

## What To Inspect

- `final_state`: the easiest confirmation that the workflow reached the expected branch or terminal state.
- `workflow_checkpoint` nodes: persisted snapshots, including `_rt_join` frontier state for resume.
- `workflow_step_exec` nodes: persisted per-step execution records.
- `workflow/wf_trace.sqlite`: SQLite trace sink for step, routing, join, checkpoint, and run lifecycle events.
- `kogwistar/scripts/workflow.bundle.cdc.script.hl3.html`: bundled viewer asset for CDC/graph inspection.
- `/api/workflow/runs/{run_id}/events`: hosted event stream path for workflow runs.
- `/api/runs/{run_id}/events`: conversation run event stream path used by the chat API guide.

## Native Runtime vs LangGraph

- Use the native runtime for actual execution, checkpointing, suspend/resume, and event-sourced behavior.
- Use `kogwistar.runtime.langgraph_converter.to_langgraph(...)` for interoperability, visualization, and semantic comparison.
- Do not treat the converter as the authoritative implementation of pause/resume behavior.

## Glossary

- `WorkflowRuntime`: executes a workflow graph persisted in the workflow engine.
- `MappingStepResolver`: registry that maps workflow `wf_op` values to Python handlers.
- `StepContext`: runtime context object passed into native resolver handlers.
- `RunSuccess`: step result that applies state updates and routes forward.
- `RunSuspended`: step result that parks a token and requires external resume.
- `_rt_join`: runtime-owned join/barrier frontier persisted in checkpoint state.

## Next Documents

- [Runtime Level 0 - WorkflowRuntime Basics](./runtime-level-0-basics.md)
- [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md)
- [Runtime Level 2 - Pause and Resume](./runtime-level-2-pause-resume.md)
- [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md)
- [Runtime Level 4 - Sandboxed Ops With Docker](./runtime-level-4-sandboxed-ops.md)
