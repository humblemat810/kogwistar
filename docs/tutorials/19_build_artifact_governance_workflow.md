# Tutorial 19: Build Artifact Governance Workflow

Companion: [scripts/tutorial_sections/19_build_artifact_governance_workflow.py](../../scripts/tutorial_sections/19_build_artifact_governance_workflow.py)

## What You Will Build

You will run two deterministic artifact-governance workflows against the same build artifact:

- a safe workflow that applies a real `public` schema mode before publish
- an unsafe workflow that skips filtering and is rejected by the invariant

The build artifact contains internal-only fields such as `source_maps`, `raw_sources`, and sensitive metadata, and the workflow proves that those fields cannot cross the public boundary.

## Why This Matters

This tutorial turns "don't leak source maps" from convention into an encoded system contract:

- schema mode slicing makes `public` a first-class boundary
- the public payload comes from `field_mode="public"` and is validated against `BuildArtifact["public"]`, not from ad hoc field stripping
- workflow validation fails fast if that boundary is skipped
- governance events are appended to the conversation graph so the decision is inspectable and replayable
- replay reconstructs both the published public artifact and the blocked unsafe run

This is the pattern you want when an output crossing a boundary must never contain sensitive build artifacts.

## Run or Inspect

Run the companion script from the repo root:

```powershell
python scripts/tutorial_sections/19_build_artifact_governance_workflow.py
```

The script always uses the same resettable data directory:

```text
.gke-data/tutorial-sections/19_build_artifact_governance_workflow
```

That keeps every run easy to inspect without guessing where artifacts landed.

## Inspect The Result

Look for these artifacts after the script runs:

- In the workflow graph:
  both the safe and unsafe governance workflows should have persisted nodes and edges.
- In the conversation graph:
  each run should have `workflow_run`, `workflow_step_exec`, and `artifact_governance_event` nodes.
- In the safe run:
  the published artifact should have `mode="public"` and should omit `source_maps`, `raw_sources`, `metadata.source_root`, `metadata.source_map_manifest`, and `metadata.internal_notes`.
- In the unsafe run:
  validation should fail before publish, the run should emit `artifact_rejected`, and `published_artifact` should stay empty.
- In replayed state:
  the safe replay should reconstruct the published artifact, while the unsafe replay should reconstruct the rejection state.

The filtering step uses mode slicing as the primary boundary and then validates the result with explicit invariants. That means the workflow does not rely on CI-only checks or developer discipline to prevent leakage.

## Invariant Demonstrated

It is impossible for this workflow to publish an artifact that still contains source maps, raw sources, or sensitive metadata.

The guarantee comes from three encoded layers working together:

- `BuildArtifact` and `ArtifactMetadata` define a `public` schema mode
- the public projection is materialized with `field_mode="public"` and checked against `BuildArtifact["public"]`
- `validate_artifact` rejects any payload that still contains sensitive fields
- governance event nodes plus workflow checkpoints make the decision replayable

## Next Tutorial

Pair this example with [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) when you want to trace how these governance events can feed downstream observers and audit tooling.
