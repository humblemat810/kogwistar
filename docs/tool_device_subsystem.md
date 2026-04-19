# Tool / Device Subsystem

Tools are first-class resources, not opaque helper functions.

This slice uses a small three-part shape:

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

## What It Means

The substrate can now say:

- which tool exists
- what capability it needs
- what it did
- whether it changed state

That is the first step toward device-like tool governance.

## Current Limits

- registry is still light and in-process
- sync vs async contract is only modeled, not fully enforced
- long-running tool execution still rides existing conversation/workflow runtime paths

## Smoke Shape

The current smoke case only proves registry behavior:

```text
register -> resolve requirement -> revoke
```

That keeps the slice small while preserving the right abstraction seam.
