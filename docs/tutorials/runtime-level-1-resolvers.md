# Runtime Level 1: Custom Resolvers and Injected Dependencies

Goal: show how resolvers consume runtime context, injected dependencies, and `StepContext.events`.

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

## Troubleshooting

- If `dep_echo` is empty, inspect the script’s `_base_initial_state()` helper.
- If custom events are missing, verify the trace sink file exists under the workflow data directory.
