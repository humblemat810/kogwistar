# 26 Recovery and Durable Operational State

Audience: Advanced / contributor  
Time: 20-30 minutes

Companion notebook: [`scripts/tutorial_sections/26_recovery_and_durable_operational_state.py`](../../scripts/tutorial_sections/26_recovery_and_durable_operational_state.py)

## What You Will Build

You will inspect a small recovery walkthrough that damages serving projections on purpose and then restores them through the core recovery APIs.

The tutorial keeps durable truth and latest-state convenience separate on purpose:

- graph and entity-event history remain authoritative
- queue and lane leases remain the redelivery mechanism
- named projections hold latest operational state for fast inspection
- recovery performs bounded repair and reporting, not orchestration

## Why This Matters

Recovery semantics change how careful users operate the system.

This repo does not treat restart behavior as an afterthought. It makes a sharp distinction between:

- what survives restart
- what can be repaired from durable truth
- what becomes claimable again through lease expiry
- what still needs explicit policy before resuming

Without that distinction, restart handling sounds simpler than it really is, and operators end up trusting the wrong surface.

## Run or Inspect

Run the notebook companion:

```bash
python scripts/tutorial_sections/26_recovery_and_durable_operational_state.py
```

Read these alongside it:

- [23 Lane Messaging Contract](./23_lane_messaging_contract.md)
- [20 Generic Named Projection Meta Layer](./20_generic_named_projection_meta_layer.md)
- [14 Architecture Deep Dive](./14_architecture_deep_dive.md)

## Durable Surfaces

After restart, these are the important durable surfaces in the current design:

- durable jobs in the job queue
- lane-message graph truth and entity events
- projected lane-message rows
- workflow checkpoints
- run history
- service health lifecycle facts plus durable latest-state rows

The important split is:

- sparse lifecycle truth belongs in graph/oplog
- latest operational state belongs in named projections

That is why lane rows and service health latest state are rebuildable without pretending that the projection itself was the source of truth.

## Inspect Versus Recover

`engine.recovery.inspect(...)` is a read-only operator view.

It reports queues, lanes, checkpoints, run history, dead letters, service health, and app surfaces without mutating them.

`engine.recovery.recover_startup(...)` is the bounded repair entrypoint.

In the current implementation it may repair:

- missing lane-message projection rows
- missing service-health latest-state rows

It still does not:

- restart daemons
- schedule work
- authorize actions
- route messages
- blindly resume every interrupted workflow

Checkpoint resume remains policy-gated and is off by default.

## Lane And Service Repair

Lane-message projection repair works by rebuilding serving rows from authoritative lane-message graph/entity-event truth.

Service-health repair works by rebuilding durable latest-state rows from sparse lifecycle facts such as:

- service registered
- instance started
- error changed
- stale / degraded / failed
- recovered
- stopped

Neither path turns projections into primary truth. The repair logic exists precisely because projections are allowed to be disposable.

## Lease Redelivery

Recovery is not exactly-once processing.

Queue jobs and claimed lane rows use lease-based redelivery:

- if a process stops after claim and before ack, the item becomes claimable again after lease expiry
- handlers must remain idempotent
- recovery improves convergence and visibility, but it does not erase at-least-once semantics

This is why restart handling has two different mechanisms:

- bounded projection repair for missing latest-state views
- lease expiry for interrupted in-flight work

## Inspect The Result

After running the companion, confirm these points:

- deleting a projected lane row does not delete authoritative lane truth
- `inspect(...)` reports the current state without repairing it
- `recover_startup(...)` restores missing lane projection rows
- deleting latest service-health projection state does not delete lifecycle truth
- `recover_startup(...)` rebuilds latest service health from sparse lifecycle events
- recovery remains observability plus bounded repair, not hidden supervision

## Invariant Demonstrated

If recovery needs to know it after restart, it must be persisted.

If replay or provenance needs it, it belongs in graph/oplog.

If only latest operational visibility needs it, it belongs in a durable projection.

That is why recovery in this repo is intentionally narrow: it repairs views from truth and lets lease semantics handle interrupted delivery.

## Next Tutorial

Return to [14 Architecture Deep Dive](./14_architecture_deep_dive.md) for the larger subsystem picture, or revisit [23 Lane Messaging Contract](./23_lane_messaging_contract.md) and [20 Generic Named Projection Meta Layer](./20_generic_named_projection_meta_layer.md) for the two substrate patterns that make this recovery model work.
