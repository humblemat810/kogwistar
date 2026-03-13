# Runtime Level 1: Custom Resolvers and Injected Dependencies

Goal: show how resolvers consume runtime context, injected dependencies, and `StepContext.events`.

## What You Will Build

You will run the canonical workflow with registered resolver handlers, inspect custom trace emissions, and confirm injected `_deps` reaches the resolver code.

## Why This Matters

This level shows how the runtime becomes programmable without losing its core execution model. Resolver logic is explicit and inspectable, not hidden behind framework magic.

## Run or Inspect

## Quick Run

```powershell
python scripts/runtime_tutorial_ladder.py level1 `
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"dep_echo": "runtime-tutorial"`: confirms `_deps` reached the resolver
- `"state_schema"`: merge modes used for native/runtime updates
- `"registered_ops"`: resolver registry contents
- `"custom_event_payloads"`: emitted from resolver code through `StepContext.events`
- `"checkpoint_pass": true`

## Inspect The Result

- Confirm the resolver reads the injected audience from `_deps`.
- Inspect the custom event payloads recorded in the trace sink.
- Compare this with [07 Branch Join Workflows](./07_branch_join_workflows.md) if you want the higher-level explanation first.

## What This Level Teaches

- Dependencies are injected through `initial_state["_deps"]`.
- Resolver handlers receive `StepContext`, so they can read state, emit telemetry, and return `StepRunResult` variants.
- `StepContext.events` can emit custom trace events alongside built-in step and routing telemetry.
- `MappingStepResolver.set_state_schema(...)` documents preferred merge behavior for keys such as append vs extend.

## Relevant Surfaces

- `MappingStepResolver.register(...)`
- `StepContext.state_view`
- `StepContext.events`
- `RunSuccess`
- `RunSuspended`

## Checkpoint

Pass when:

- the resolver reads the injected audience from `_deps`
- custom `tutorial_resolver_note` events show up in the trace sink
- the run still suspends at the same workflow point as Level 0

## Invariant Demonstrated

Runtime customization stays observable. Resolver logic can emit domain-specific notes without bypassing the runtime's step model.

## Troubleshooting

- If `dep_echo` is empty, inspect the script's `_base_initial_state()` helper.
- If custom events are missing, verify the trace sink file exists under the workflow data directory.

## Next Tutorial

Continue to [Runtime Level 2 - Pause and Resume](./runtime-level-2-pause-resume.md) or return to [07 Branch Join Workflows](./07_branch_join_workflows.md).
