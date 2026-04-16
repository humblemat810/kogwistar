# Runtime Level 3: CDC Viewer and LangGraph Interop

Goal: inspect the runtime's emitted events, connect them to the bundled CDC viewer, and export the same workflow to LangGraph.

## What You Will Build

You will complete the canonical workflow, inspect its trace-event inventory, confirm the bundled CDC viewer path, and observe how LangGraph export is positioned as interop rather than the source of truth.

## Why This Matters

This level connects runtime telemetry, viewer surfaces, and external graph export into one story. It makes the runtime observable without weakening its native execution model.

## Run or Inspect

## Quick Run

```bash
python scripts/runtime_tutorial_ladder.py level3 \
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"trace_event_types"`: includes step lifecycle, routing, join, checkpoint, and suspension events
- `"cdc_viewer_html"`: bundled viewer asset path
- `"runtime_event_endpoint"`: hosted event stream path for a workflow run
- `"langgraph"`: converter status and output summary
- `"checkpoint_pass": true`

## Inspect The Result

- Confirm the trace event inventory includes run, step, edge, join, and checkpoint lifecycle.
- Confirm the viewer asset path exists on disk.
- Confirm the runtime event endpoint exposes the workflow run as a stream-friendly surface.

## What This Level Teaches

- Runtime telemetry is written to the SQLite trace sink at `workflow/wf_trace.sqlite`.
- Useful event types include `step_attempt_started`, `edge_selected`, `join_released`, `checkpoint_saved`, and `workflow_run_completed`.
- The script also confirms suspension by reading checkpoint frontier state, because some environments may not surface `workflow_step_suspended` in the trace sink even when the run really paused and resumed.
- Hosted consumers can read the same workflow run stream from `/api/workflow/runs/{run_id}/events`.

## LangGraph Positioning

Use `kogwistar.runtime.langgraph_converter.to_langgraph(...)` for interoperability and export.

- `execution="visual"` favors a cleaner diagram.
- `execution="semantics"` preserves more fanout and join behavior.
- Native `WorkflowRuntime` remains the source of truth for suspend/resume and event-sourced execution.

## Checkpoint

Pass when:

- required runtime event types are present in the trace sink
- the bundled viewer asset exists
- workflow run, step exec, and checkpoint nodes were all persisted

## Invariant Demonstrated

Observability is attached to the native runtime path. Export and visualization do not replace event-sourced execution semantics.

## Troubleshooting

- If LangGraph conversion is unavailable: install the optional extra that provides `langgraph`.
- If trace events are missing, confirm the workflow run used the default trace-enabled runtime path.

## Next Tutorial

Return to [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) or revisit [Runtime Ladder Overview](./runtime-ladder-overview.md).
