# Budget Rate Switch Demo

This demo shows the intended budget semantics with a fake token generator and a
single-lane scheduler.

## What it proves

- A token-window budget can block a heavy workflow before it finishes.
- While the heavy workflow is paused, other work can continue.
- Low-token work can consume leftover tokens before the window refreshes.
- Zero-token work can still run even when the token window is exhausted.
- After the window refreshes, the paused workflow can resume and finish.

## Model

The demo uses:

- `RunScheduler(max_active=1)` for single-lane execution.
- `FakeClock` so refresh timing stays deterministic.
- `FakeTokenGenerator` backed by `RateBudgetWindow`.

The token window is:

- `limit = 3`
- `window_ms = 10000`

The runs are:

- `heavy-run`
  - step 1 needs `2` tokens
  - step 2 needs `2` tokens
  - it pauses after step 1 because only `1` token remains
- `tiny-run`
  - needs `1` token
  - it can finish before refresh by using that leftover token
- `free-run`
  - needs `0` tokens
  - it can also finish before refresh

After `clock.advance(10001)`, the token window refreshes and `heavy-run` is
submitted again and finishes.

## Expected order

```text
heavy.started
heavy.step1
heavy.rate_blocked
tiny.started
tiny.finished
free.started
free.finished
token_window.refreshed
heavy.finished
```

The completion order should be:

```text
tiny
free
heavy
```

## Run

```powershell
.\.venv\Scripts\python.exe -m kogwistar.demo.budget_rate_switch_demo
.\.venv\Scripts\pytest.exe tests/core/test_budget_rate_switch_demo.py -q
```

## Why no cron is required here

This demo uses lazy refresh:

- the token window refreshes when work checks or debits the budget
- no background timer is required just to recompute remaining tokens

If you want paused runs to wake up automatically at refresh time without a new
submission or explicit resume, that would require an active scheduler tick or a
timer-driven wakeup path.
