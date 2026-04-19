# Tool / Device Subsystem

Tools are first-class resources, not opaque helper functions.

This slice uses a four-part shape:

- `ToolDefinition`
  - stable identity
  - capability requirement
  - kind
  - async flag
  - side-effect tags
- `ToolRegistry`
  - register
  - list
  - revoke
  - requirement lookup
- `ToolReceipt`
  - input
  - output
  - status
  - side-effects
  - execution mode

## What It Means

The substrate can now say:

- which tool exists
- what capability it needs
- what it did
- whether it changed state

That is the first step toward device-like tool governance.

## Current Limits

- registry is still light and in-process
- sync vs async contract is enforced at runner boundary
- subworkflow tools can be wrapped as child-process style executions
- long-running / human-approval tools are governed by receipt kind + capability path

## Smoke Shape

The current smoke shape now proves registry + runner behavior:

```text
register -> resolve requirement -> revoke
sync tool -> async tool -> side-effecting tool -> subworkflow tool
```

That keeps the slice small while preserving the right abstraction seam.
