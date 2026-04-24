# 25 Async Runtime Basics

Audience: Runtime / backend developers  
Time: 20-30 minutes

Companion notebook: [`scripts/tutorial_sections/25_async_runtime_tutorial.py`](../../scripts/tutorial_sections/25_async_runtime_tutorial.py)

## What You Will Build

A tiny async runtime resolver call and a runtime object that preserves sync contract shape.

## What You Will Learn

This tutorial introduces the async runtime track.
You will see how the async runtime keeps the same workflow semantics as the native runtime while handling cancellation, resume, and backend parity.

## Why This Matters

Async runtime is not a new workflow language.
It is the same workflow contract under async execution rules.

That matters because the repo needs:

- parity with sync runtime semantics
- safe cancellation behavior
- PostgreSQL / Chroma backend support
- traceable suspend / resume behavior

## Core Concepts

- `AsyncWorkflowRuntime`: async execution surface
- `parity`: same accepted state and trace shape as sync runtime
- `cancellation`: cooperative async stop behavior
- `suspend / resume`: durable pause and continue
- `trace`: inspectable run history

## Read This First

- [Runtime Ladder Overview](./runtime-ladder-overview.md)
- [Async Runtime ARD](../kogwistar_async_runtime_ard.md)
- [Async Runtime Implementation Checklist](../kogwistar_async_runtime_impl_checklist.md)

## Run or Inspect

Run the notebook companion:

```bash
python scripts/tutorial_sections/25_async_runtime_tutorial.py
```

## What To Inspect

The async runtime track keeps these expectations:

- same workflow graph contract
- same step context shape
- same state merge semantics
- same checkpoint and trace truth
- different execution mechanics, not different meaning

## Practical Entry Points

Use these as the main inspection points:

- `docs/tutorials/runtime-ladder-overview.md`
- `docs/tutorials/runtime-level-0-basics.md`
- `docs/tutorials/runtime-level-2-pause-resume.md`
- `tests/runtime/test_async_runtime_contract.py`
- `tests/runtime/test_workflow_cancel_event_sourced.py`
- `tests/server/test_chat_server_async_events.py`

If you want the broad flow, run the runtime ladder levels in order.
If you want the contract proof, read the async runtime tests beside the ARD.

## Code Example

Run the ladder, then inspect async parity:

```bash
python scripts/runtime_tutorial_ladder.py reset --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level0 --data-dir .gke-data/runtime-tutorial-ladder
python scripts/runtime_tutorial_ladder.py level2 --data-dir .gke-data/runtime-tutorial-ladder
```

Python-side, the async runtime is meant to keep same workflow contract:

```python
from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime

runtime = AsyncWorkflowRuntime(...)
result = await runtime.run(...)
```

Use this shape to check:

- same state contract
- same suspend / resume meaning
- same trace / checkpoint truth

## Inspect The Result

You should be able to answer:

- does async runtime keep same state contract as sync runtime?
- does suspend / resume preserve run identity?
- does cancellation stay cooperative and inspectable?
- do backend parity tests still pass?

## Invariant Demonstrated

Async runtime changes execution style, not workflow meaning.

## Next Tutorial

Continue with [Runtime Level 0: WorkflowRuntime Basics](./runtime-level-0-basics.md) or [Runtime Level 2: Pause and Resume](./runtime-level-2-pause-resume.md).
