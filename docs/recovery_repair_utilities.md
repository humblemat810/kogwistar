# Recovery / Repair Utilities

`Slice 12` makes serving projections repairable from authoritative truth.

## Main Actions

```text
repair service projection
  -> rebuild one service view from service_definition + service_event

repair all service projections
  -> scan and rebuild all service views

repair orphaned claims
  -> move expired claimed messages back to pending

replay run history
  -> inspect authoritative run timeline

dead-letter inspect / replay
  -> inspect scheduler dead letters
  -> resume paused or failed work when appropriate
```

## Rules

- authoritative truth first
- projection can be dropped and rebuilt
- repair APIs are operator/admin only
- no repair path should become new truth source

## What It Buys

- projection corruption recoverable
- partial failures tolerable
- serving state stays a view, not source of truth

