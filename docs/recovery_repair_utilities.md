# Recovery / Repair Utilities

`Slice 12` makes serving projections repairable from authoritative truth.

`engine.recovery` is the core restart-recovery coordinator. Apps call it on
daemon startup and pass only app-specific output probes such as projection
manifest/vault state.

## Main Actions

```text
repair service projection
  -> rebuild one service view from workflow_service_definition + service_event

repair all service projections
  -> scan and rebuild all service views

repair service health
  -> rebuild latest health from service_health_definition + service_health_event

repair orphaned claims
  -> move expired claimed messages back to pending

repair lane message projection
  -> rematerialize projected lane-message rows from graph/entity-event truth

recover startup
  -> safely repair lane projections
  -> inspect queues, lane rows, checkpoints, run history, dead letters, daemon health
  -> include app output surfaces in one operator report

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
- lane-message repair is explicit; there is no default time-windowed crash reprojection loop
- startup recovery is bounded and non-destructive by default
- workflow service supervision and service health are separate surfaces
- recovery may read and repair service health, but it does not orchestrate workflow services
- heartbeat updates may refresh latest-state projections, but service-health repair reconstructs `last_seen_ms` only from persisted lifecycle facts
- repaired `last_seen_ms` is best available persisted observation time, not a recreated heartbeat stream
- once the latest-state row is lost, exact heartbeat freshness cannot be reconstructed unless those heartbeat observations were themselves persisted as truth
- an immediate post-repair stale classification can be a valid coarse visibility outcome; it is not proof that fresh heartbeats were absent after the last persisted lifecycle fact
- workflow auto-resume requires an explicit restartable marker and caller-provided resume hook
- delivery is at-least-once; handlers and output projections should converge through idempotency keys, stable artifact IDs, completion markers, and versioned replacement

## What It Buys

- projection corruption recoverable
- partial failures tolerable
- serving state stays a view, not source of truth
- repaired service health stays an operator-facing latest-state view, not a fully faithful heartbeat history
- operators get one report for queue, lane, checkpoint, run, dead-letter, daemon, and app-output surfaces

