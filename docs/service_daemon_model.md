# Service Daemon Model

`Slice 10` makes workflow-backed services behave like managed services while
keeping narrow service-health visibility separate.

## Core Shape

```text
service_definition truth
  -> service_event truth
  -> service_registry projection
  -> operator view / process table

service_health_definition truth
  -> service_health_event truth
  -> service_health projection
  -> operator health view
```

## Main Rules

- `workflow_service_definition` is authoritative truth for supervisor-managed services.
- `service_health_definition` is authoritative truth for durable latest service health.
- `service_event` records lifecycle, heartbeat, trigger, and restart facts.
- `service_health_event` records sparse lifecycle facts for health repair.
- `meta_sqlite` stores latest projection only.
- `workflow_run` remains child execution truth.
- workflow service supervision lives above backend semantics.
- service health records durable latest state, not scheduling or restart policy.

## Trigger Types

- `schedule`
- `message arrival`
- `graph change`
- `external event`

All four flow through one service-trigger contract.

## Lifecycle

- `enabled`
- `starting`
- `healthy`
- `degraded`
- `restarting`
- `stopped`

## Operator Surface

- service table
- health snapshot
- last heartbeat
- restart count
- current child run

## Test Meaning

- declare service
- heartbeat updates health
- restart policy honors backoff and max restarts
- schedule/message/graph/external triggers all map to same service path
- process table shows service row plus child workflow row

