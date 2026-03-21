# ADR: Graph-Authoritative Workflow Design History with Serving Projection

## Status
Accepted and implemented.

## Summary
- Workflow design truth lives only in the authoritative workflow-engine event store for `wf_design:{workflow_id}`.
- The authoritative store contains both entity mutation events and explicitly typed `design_control` events within the same authoritative event store.
- The runtime executes only a non-authoritative serving projection of the active workflow graph.
- Sidecar workflow-history SQLite and sidecar server-run SQLite are removed.
- Projection metadata and snapshots are allowed for speed, but they are disposable and rebuildable from authoritative history.

## Context
The current workflow design implementation is hybrid:
- authoritative-seeming history is split between workflow-engine event history and a sidecar SQL pointer/history store
- server run lifecycle is also persisted in a separate sidecar SQLite registry
- abandoned redo branches are represented via masking logic, but history/head state still depends on a mutable side database

This creates split-brain risk, backend inconsistency, and poor operational fit for the project’s backend-respecting architecture.

## Decision

### 1. Authoritative workflow design history
- Workflow design truth is the per-workflow authoritative event stream under namespace `wf_design:{workflow_id}`.
- That stream contains:
  - entity mutation events for workflow node/edge changes
  - typed `design_control` events:
    - `MUTATION_COMMITTED`
    - `UNDO_APPLIED`
    - `REDO_APPLIED`
    - `BRANCH_DROPPED`

### 2. Version semantics
- One `MUTATION_COMMITTED` event defines one undoable committed design version.
- `version` is workflow-local, monotonic, unique, and never reused.
- Version adjacency is defined by `prev_version`, not by numeric contiguity.
- `UNDO_APPLIED` and `REDO_APPLIED` move the selected head only; they do not create versions.
- For `UNDO_APPLIED` and `REDO_APPLIED`, `target_seq` references the commit boundary of the destination version and does not imply new entity mutation events.
- `MUTATION_COMMITTED.target_seq` references the final authoritative seq of the entity mutation events included in that committed version.
- For the first committed version, the lower membership bound is namespace origin seq `0` exclusive.

### 3. Semantic truth vs ordering truth
- Version lineage is semantic truth.
- Seq is ordering/indexing truth only.
- Inactive seq intervals are derived mechanically from version membership boundaries of versions not on the active lineage.
- Seq masking is never an independent truth source.

### 4. Branch semantics
- `BRANCH_DROPPED` marks versions as inactive after `undo -> new edit`.
- `BRANCH_DROPPED` is visibility masking only.
- It does not:
  - tombstone abandoned branch node/edge entities
  - delete authoritative entity events
  - rewrite historical meaning

### 5. Lineage fold contract
Folding authoritative history for one workflow must produce:
- ordered active lineage version list
- current head version
- inactive version set
- undo availability
- redo availability
- active lineage tip version

Field semantics:
- `current_version`: currently selected active head after undo/redo state is applied
- `active_tip_version`: tip version of the active lineage
- `max_version`: retained for API compatibility and means the active lineage tip version id, not the largest historical version id ever allocated

### 6. Serving projection
- Keep one serving projection of the active visible workflow graph:
  - visible active workflow nodes
  - visible active workflow edges
- Default implementation:
  - reuse the existing runtime-facing visible workflow node/edge materialization
  - do not introduce duplicate active-node/active-edge tables in this phase
- Runtime rule:
  - runtime reads only from the serving projection
  - runtime must not interpret raw authoritative workflow-design history directly
  - runtime must not fall back to raw authoritative workflow-design history when serving projection materialization is stale or rebuilding

### 7. Serving projection invariants
- only active nodes are present
- only active edges are present
- no visible edge may exist whose source or target endpoint is absent from the visible node set
- exactly one visible active head node exists in the serving projection for a workflow design lineage
- if the serving projection contains entities that cannot be derived from authoritative history under the current lineage fold, the projection must be discarded and rebuilt

### 8. Projection metadata and snapshots
Backend-respecting projection metadata tables are allowed:
- `workflow_design_projection_head`
- `workflow_design_projection_versions`
- `workflow_design_projection_dropped_ranges`
- `workflow_design_snapshots`

Projection head fields:
- `workflow_id`
- `current_version`
- `active_tip_version`
- `last_authoritative_seq`
- `last_materialized_seq`
- `projection_schema_version`
- `snapshot_schema_version`
- `materialization_status`
- `updated_at_ms`

Staleness detection:
- missing row
- schema mismatch
- `last_authoritative_seq` behind namespace latest seq
- malformed or incomplete projection rows

Snapshots:
- projection-only
- full visible node/edge JSON payload
- config-driven cadence, default `50` committed versions
- missing/corrupt snapshot falls back to replay

### 9. Refresh semantics
- Add an admin/internal refresh only.
- Refresh is idempotent.
- Refresh clears and rebuilds projection metadata, snapshots, and serving projection from authoritative history.
- While `materialization_status=rebuilding`, same-workflow design writes and workflow submissions fail fast with retryable conflict.
- Partial rebuild state must never be treated as authoritative or executable.

### 10. Determinism and authority
- Deterministic rebuild invariant:
  - same authoritative history + same schema version => same lineage fold and same serving projection
- Projection non-authority invariant:
  - if projections disagree with authoritative history, authoritative history always wins and projections are replaced
- Serving projection is disposable; historical truth is not

### 11. Transaction boundaries
- The authoritative SQL transaction for one design mutation must atomically commit:
  - all entity mutation events for that mutation
  - all related `design_control` events
- Projection maintenance may lag and must be recoverable.
- Cross-store ACID with Chroma is not promised.

### 12. Server run state
- Server run persistence is a separate operational model from workflow design history.
- Store it in the workflow engine’s backend-respecting meta backend.
- Use:
  - authoritative current-state run rows
  - append-only run event rows for audit/SSE replay
- Remove `server_runs.sqlite`.

## Implementation Plan

### A. Remove sidecar workflow-history logic
- Delete all workflow-history pointer/table code from [chat_service.py](/c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/server/chat_service.py).
- Remove:
  - `workflow_design_history.sqlite`
  - `workflow_design_history`
  - `workflow_design_pointer`
  - `HISTORY_BACKFILL`
  - any tests depending on backfill or sidecar files

### B. Move workflow design history to event-derived fold
- Replace SQL-backed `current_version/max_version/versions/can_undo/can_redo` derivation with a control-event fold.
- Expand `design_control` payloads so fold logic is decision-complete.
- Keep current REST/MCP history response shape unchanged.
- Keep `BRANCH_DROPPED` as masking-only control state.

### C. Add backend-respecting projection metadata
- Extend [engine_sqlite.py](/c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/engine_core/engine_sqlite.py) and [engine_postgres_meta.py](/c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/engine_core/engine_postgres_meta.py) with projection metadata tables and APIs.
- Projection APIs must support:
  - get head/version state
  - upsert rebuilt projection rows
  - clear one workflow projection
  - manage snapshots
  - report last authoritative seq vs last materialized seq

### D. Make serving projection explicit
- Treat the currently materialized visible workflow node/edge set as the serving projection.
- Rebuild serving projection from:
  - lineage fold
  - nearest valid snapshot
  - forward replay of authoritative entity events
- Enforce no-orphan-edge and single-visible-head invariants during rebuild.

### E. Add internal refresh path
- Add service-level internal/admin refresh for one workflow.
- It must:
  - mark rebuilding
  - clear projection metadata and serving projection
  - rebuild from authoritative history
  - mark stable on success
- Same-workflow writes/submissions must fail fast while rebuild is active.

### F. Remove sidecar server run registry
- Replace [run_registry.py](/c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/server/run_registry.py) SQLite persistence with backend-respecting meta persistence.
- Keep existing REST/MCP/SSE behavior and event taxonomy unchanged.
- Update [server_mcp_with_admin.py](/c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/server_mcp_with_admin.py) wiring so no `server_runs.sqlite` file is created.

## Test Plan
- Lineage fold:
  - first commit lower bound is correct
  - `current_version`, `active_tip_version`, and API `max_version` semantics are correct
  - lineage adjacency follows `prev_version`, not numeric adjacency
  - unique monotonic version ids remain correct across undo/new-edit
- Undo/redo:
  - `UNDO_APPLIED` / `REDO_APPLIED` target the destination committed version boundary only
  - no new entity mutation events are implied by undo/redo
- Branch behavior:
  - `undo -> new edit` emits `BRANCH_DROPPED`
  - abandoned branch remains in audit history
  - abandoned branch never resurfaces in serving projection
  - no branch-drop tombstone sweep occurs
- Serving projection:
  - runtime reads only serving projection
  - no orphan visible edges
  - no runtime fallback to raw history during stale/rebuilding state
  - divergence causes discard and rebuild
- Refresh:
  - refresh is idempotent
  - refresh rebuilds deterministic state
  - rebuild-in-progress blocks same-workflow writes/submissions
- Cleanup:
  - no `workflow_design_history.sqlite`
  - no `server_runs.sqlite`
- Backend coverage:
  - SQLite meta backend
  - Postgres meta backend

## Assumptions
- No migration/backfill is required for old sidecar workflow history.
- Existing REST/MCP response shapes remain stable.
- Cross-store ACID with Chroma is out of scope.
