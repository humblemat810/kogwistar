# Scheduler Priority Demo

This demo shows the exact Slice 7 follow-up case:

- one scheduler lane only: `max_active=1`
- a low-priority workflow starts first
- the low-priority workflow reaches a cooperative block and yields the lane
- a high-priority workflow enters later and finishes before the low-priority workflow resumes

Run it with:

```bash
python -m kogwistar.demo.scheduler_priority_demo
```

## Why the cooperative block matters

The scheduler does not hard-preempt an already-running step.
That means a high-priority run cannot cut in front of a low-priority run that is still actively consuming the only lane.

The handoff only works when the low-priority run reaches a safe boundary and yields.
In this demo, that boundary is modeled by a manual boolean switch:

- `allow_low_finish = False` when the low-priority run first starts
- the low-priority run sees the switch is false, records `low.blocked`, and marks itself paused
- the lane becomes free
- the high-priority run is submitted and finishes
- the switch is then set to `True`
- the low-priority run is resumed and finishes last

## Expected outcome

The result should report:

- `order == ["high", "low"]`
- `high_finished_before_low == true`
- `low_blocked_before_high_started == true`

## Tiny timeline sketch

```text
low.started
low.blocked
high.started
high.finished
low.resumed
low.finished
```

## Related verification

- Scheduler behavior test:
  - `tests/server/test_run_scheduler.py::test_run_scheduler_high_priority_can_finish_before_low_after_cooperative_block`
- Demo smoke test:
  - `tests/core/test_scheduler_priority_demo.py::test_scheduler_priority_demo_smoke`
