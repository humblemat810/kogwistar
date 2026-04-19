# Budget Accounting Flow

Budget accounting is canonical at the OS layer, but provider-specific at the adapter layer.

## Canonical shape

```text
BudgetEvent
  run_id
  source
  kind        # token | cost | time
  amount
  unit
  scope       # run | step | tool | provider
  ts_ms
  meta
```

## Flow

```text
Provider response / tool result
   |
   v
Provider adapter
   |
   |- maps provider usage schema
   |- normalizes into BudgetEvent list
   `- preserves raw metadata in meta
   v
BudgetLedger / CostLedger
   |
   |- debits run budget
   |- records canonical events
   `- raises BudgetExhaustedError when exhausted
```

## Rule

- OS policy reads canonical events only
- provider-specific usage stays in adapters
- no provider format becomes core truth
- persisted `state["budget"]` is authoritative budget state
- `_deps["budget_ledger"]` is a state-backed helper, not second truth
- rate windows use lazy refresh on read/debit; no cron is required unless paused runs must auto-wake
- runtime-token pinning is branch-aware: one suspended branch can wait while sibling branches finish, then resume after refresh
