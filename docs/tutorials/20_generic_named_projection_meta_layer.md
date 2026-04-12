# Tutorial 20: Generic Named Projection Meta Layer

Companion: [scripts/tutorial_sections/20_generic_named_projection_meta_layer.py](../../scripts/tutorial_sections/20_generic_named_projection_meta_layer.py)

## What You Will Build

You will run a small bridge-governance projector that uses the new generic named-projection meta API.

The demo keeps two things separate on purpose:

- authoritative append-only history in `entity_events`
- latest-state convenience rows in `named_projections`

Each interaction id gets one named projection row that stores the folded payload plus freshness metadata such as `last_authoritative_seq`, `last_materialized_seq`, `projection_schema_version`, and `materialization_status`.

## Why This Matters

This is the reusable pattern for projections in this repo:

- meta/sql stores projection rows generically by `namespace + key`
- the service layer owns the domain-specific fold algorithm
- append-only history remains authoritative
- projections are disposable and rebuildable
- feature code does not write backend-specific SQL for SQLite vs Postgres

Workflow design remains one compatibility consumer of this substrate, but the bridge-governance example shows the abstraction is not workflow-only.

## Run or Inspect

Run the companion script from the repo root:

```bash
python scripts/tutorial_sections/20_generic_named_projection_meta_layer.py
```

The script always uses the same resettable data directory:

```text
.gke-data/tutorial-sections/20_generic_named_projection_meta_layer
```

## Inspect The Result

Look for these artifacts after the script runs:

- In the authoritative history log:
  `bridge_governance_history` should contain the append-only interaction events for `interaction-alpha` and `interaction-beta`.
- In the named projection namespace:
  `bridge_governance` should contain one latest-state row per interaction before the clear step.
- In the projected payload:
  each row should expose domain state like `latest_status`, `latest_decision`, `participants`, `active_agents`, and `policy_versions`.
- In the freshness metadata:
  each row should expose `last_authoritative_seq`, `last_materialized_seq`, `projection_schema_version`, and `materialization_status`.
- In the rebuild step:
  clearing the `interaction-alpha` projection and refreshing it again should reconstruct the same payload from history.
- In the namespace clear step:
  clearing `bridge_governance` should remove all latest-state rows without touching the authoritative history.

## Invariant Demonstrated

The named projection row is not the source of truth.

The guarantee comes from the split between:

- append-only authoritative history in `entity_events`
- a case-specific service-layer projector
- a generic meta-store row keyed by `namespace + key`

That means latest-state reads can avoid full scans in steady state, while rebuilds can still recover from authoritative history whenever a projection is missing, stale, or intentionally cleared.

## Next Tutorial

Pair this with [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) when you want to see how append-only history feeds other rebuildable projections, or with [12 Designer API Integration](./12_designer_api_integration.md) when you want to compare this generic substrate to a richer workflow-facing service surface.
