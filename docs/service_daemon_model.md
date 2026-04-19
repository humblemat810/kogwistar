# Service Daemon Model

`Slice 10` makes long-running agents and workers behave like managed services.

## Core Shape

```text
service_definition truth
  -> service_event truth
  -> service_registry projection
  -> operator view / process table
```

## Main Rules

- `service_definition` is authoritative truth.
- `service_event` records lifecycle, heartbeat, trigger, and restart facts.
- `meta_sqlite` stores latest projection only.
- `workflow_run` remains child execution truth.
- service supervision lives above backend semantics.

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

