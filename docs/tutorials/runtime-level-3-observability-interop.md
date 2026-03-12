# Runtime Level 3: CDC Viewer and LangGraph Interop

Goal: inspect the runtime's emitted events, connect them to the bundled CDC viewer, and export the same workflow to LangGraph.

## Quick Run

```powershell
python scripts/runtime_tutorial_ladder.py level3 `
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields:

- `"trace_event_types"`: includes step lifecycle, routing, join, checkpoint, and suspension events
- `"cdc_viewer_html"`: bundled viewer asset path
- `"runtime_event_endpoint"`: hosted event stream path for a run
- `"langgraph"`: converter status and output summary
- `"checkpoint_pass": true`

Example shape:

```json
{
  "level": 3,
  "trace_event_types": [
    "checkpoint_saved",
    "edge_selected",
    "join_waiting",
    "join_released",
    "predicate_evaluated",
    "step_attempt_started",
    "step_attempt_completed",
    "workflow_run_completed"
  ],
  "cdc_viewer_html": "graph_knowledge_engine/scripts/workflow.bundle.cdc.script.hl3.html",
  "runtime_event_endpoint": "/api/workflow/runs/<run_id>/events",
  "saw_suspended_checkpoint": true,
  "langgraph": {
    "available": true,
    "visual_compiled": true,
    "semantics_compiled": true
  },
  "checkpoint_pass": true
}
```

## What This Level Teaches

- Runtime telemetry is written to the SQLite trace sink at `workflow/wf_trace.sqlite`.
- Useful event types include:
  - `step_attempt_started`
  - `step_attempt_completed`
  - `predicate_evaluated`
  - `edge_selected`
  - `join_waiting`
  - `join_released`
  - `checkpoint_saved`
  - `workflow_run_completed`
- The script also confirms suspension by reading checkpoint frontier state, because some environments may not surface `workflow_step_suspended` in the trace sink even when the run really paused and resumed.
- The bundled viewer lives at [workflow.bundle.cdc.script.hl3.html](/c:/Users/chanh/Documents/graphrag_v2_working_tree/graph_knowledge_engine/scripts/workflow.bundle.cdc.script.hl3.html).
- Hosted consumers can read the same run stream from `/api/workflow/runs/{run_id}/events`.

## LangGraph Positioning

Use `graph_knowledge_engine.runtime.langgraph_converter.to_langgraph(...)` for interoperability and export.

- `execution="visual"` favors a cleaner diagram.
- `execution="semantics"` preserves more fanout and join behavior.
- Native `WorkflowRuntime` remains the source of truth for suspend/resume and event-sourced execution.

The tutorial script attempts both converter modes. If the optional `langgraph` dependency is not installed, it reports that instead of failing.

## Checkpoint

Pass when:

- required runtime event types are present in the trace sink
- the bundled viewer asset exists
- workflow run, step exec, and checkpoint nodes were all persisted

## Troubleshooting

- If LangGraph conversion is unavailable: install the optional extra that provides `langgraph`.
- If trace events are missing, confirm the workflow run used the default trace-enabled runtime path.
