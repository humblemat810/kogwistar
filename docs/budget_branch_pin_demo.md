# Budget Branch Pin Demo

This demo shows runtime-token pinning on a branching workflow.

It does not talk about LLM token usage. It talks about workflow runtime tokens:

- a branch token can be pinned by budget exhaustion
- pinned branch token does not rerun until refresh
- other branches can keep moving if they do not need the pinned budget

## What it proves

- Heavy branch consumes available runtime tokens.
- Heavy branch then suspends when next step needs more tokens.
- Light branch still completes in same run.
- Heavy branch stays pinned until token window refresh.
- After refresh, heavy branch resumes and the workflow finishes.

## Model

- `RunScheduler` is not main actor here.
- `WorkflowRuntime` runs branching workflow.
- `FakeClock` gives deterministic refresh time.
- `FakeTokenWindow` models runtime-token budget, not LLM token budget.

The workflow shape is:

```text
start
  |- heavy_1 -> heavy_2 -> join -> end
  '- light    -----------/
```

The budget shape is:

- `limit = 2`
- `window_ms = 10000`

`heavy_1` spends window. `heavy_2` cannot continue until refresh. `light`
finishes while heavy stays pinned.

## Expected order

```text
heavy_1
light
heavy_2
```

The key contract:

- `heavy_2.paused` happens before `token_window.refreshed`
- `heavy_2` does not resume before refresh
- `heavy_2` resumes only after refresh

## Run

```powershell
.\.venv\Scripts\python.exe -m kogwistar.demo.budget_branch_pin_demo
.\.venv\Scripts\pytest.exe tests/core/test_budget_branch_pin_demo.py -q
```
