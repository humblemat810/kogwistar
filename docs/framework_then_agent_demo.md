# Framework-First Demo

This demo is a small proof of the repo's substrate framing: first define a reusable agent loop framework, then instantiate one concrete agent on top of it.

Run it with:

```bash
python -m kogwistar.demo.framework_then_agent_demo
python -m kogwistar.demo.framework_then_agent_demo --variant all
```

## What It Shows

- The original framework layer defines a generic `Plan -> Approve -> Act -> Observe -> End` loop with explicit state keys and transitions.
- The easy variant removes approval but keeps the same agent unchanged, which shows this repo can host closely related framework swaps.
- The harder variant uses a batch `Collect -> Classify -> Apply -> End` shape plus a small adapter, which shows why a clearer contract matters when the framework changes more materially.
- The runtime layer is still the repo's native `WorkflowRuntime` and `MappingStepResolver` seam, so the demo exercises the same harness concepts the rest of the repo uses.

## Brief Design Note

- The approval step is the smallest useful guard extension point for the original family.
- The no-approval variant keeps the state contract nearly identical, so the same agent can run with minimal change.
- The batch variant intentionally uses a different contract, so the demo adds a tiny adapter instead of pretending one workflow shape fits all shapes.
- The demo stays deterministic and avoids LLM calls, filesystem writes, and destructive operations.
- The point is to show that the substrate can support custom framework construction, not only one-off agent loops.

## Tiny Output Sketch

```json
{
  "framework_step_order": ["plan", "approve", "act", "observe", "end"],
  "runtime_step_ops": ["plan", "approve", "act", "observe", "..."],
  "easy": {
    "framework_step_order": ["plan", "act", "observe", "end"]
  },
  "harder": {
    "framework_step_order": ["collect", "classify_batch", "apply_batch", "end"]
  },
  "staged_moves": [
    {"note_id": "note-1", "destination": "finance/"},
    {"note_id": "note-2", "destination": "meetings/"}
  ],
  "final_state": {"completed": true, "final_status": "completed"}
}
```
