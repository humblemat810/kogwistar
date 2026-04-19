# Operator Views / System Introspection

`Slice 11` makes system state legible without diving into code.

## Main Views

```text
process table
  -> workflow runs
  -> service rows

operator inbox
  -> visible lane messages

queue snapshot
  -> depth by status
  -> failed rows

blocked graph
  -> waiting / suspended runs

capability snapshot
  -> current subject
  -> effective capabilities

service dashboard
  -> health
  -> restart count
  -> current child run

resource snapshot
  -> scheduler
  -> runs
  -> services
  -> storage / cost
```

## Rules

- internal/admin only for early version
- operator views are projection, not second truth
- no recursion / no hidden auto-tick in read path

## Why It Matters

- operator can answer "what is happening now?"
- stuck flows visible
- service health visible
- budget and queue pressure visible

