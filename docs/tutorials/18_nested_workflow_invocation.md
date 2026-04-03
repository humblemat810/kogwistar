# Tutorial 18: Nested Workflow Invocation

Companion: [scripts/tutorial_sections/18_nested_workflow_invocation.py](../../scripts/tutorial_sections/18_nested_workflow_invocation.py)

## What You Will Build

You will run one parent workflow that invokes two child workflows:

- a predesigned child that already exists in the workflow engine
- a dynamically generated child whose graph is synthesized from predetermined fake planner data, persisted during the run, and then executed

## Why This Matters

This is the smallest golden example that makes the nested-workflow contract concrete instead of abstract:

- predesigned children should remain visible as workflow graphs
- dynamic children should become new workflow graphs when inserted at runtime
- both child runs should leave execution traces in the conversation graph

## Run or Inspect

Run the companion script from the repo root:

```powershell
python scripts/tutorial_sections/18_nested_workflow_invocation.py
```

The script always uses the same resettable data directory:

```text
.gke-data/tutorial-sections/18_nested_workflow_invocation
```

That keeps every run easy to inspect without guessing where artifacts landed.

## Inspect The Result

Look for these artifacts after the script runs:

- In the workflow graph:
  the parent workflow, the predesigned child workflow, and the dynamic child workflow should all have persisted nodes and edges.
- In the conversation graph:
  the parent run and both child runs should have `workflow_run` and `workflow_step_exec` trace nodes.
- In the final JSON payload:
  `predesigned_child` and `dynamic_child` should both appear in parent state, each with its own `__run_id` and `__status`.

The dynamic child uses fake predetermined planner data on purpose. That keeps the example deterministic while still demonstrating the persistence path used for on-the-fly workflow generation.

## Invariant Demonstrated

Nested workflow invocation supports both reuse and insertion:

- reuse an existing child workflow by `workflow_id`
- insert a newly synthesized child workflow graph and run it in the same parent execution

The workflow engine remains the source of workflow designs, while the conversation engine remains the source of workflow execution traces.

## Next Tutorial

Use this example together with [12 Designer API Integration](./12_designer_api_integration.md) when you want to compare interactive design-time persistence with runtime-time child workflow insertion.
