# ADR-018: Two-Stage Derived Projection and Async Semantic Indexing

**Status:** Proposed
**Date:** 2026-08-30
**Owner:** Maintainers

## Context

Current node writes synchronously create an embedding in
`WriteSubsystem._add_node_impl`, then write through a vector-capable backend.
This couples admission latency and embedding-provider failure to writes.

Kogwistar has `entity_events`, durable `index_jobs`, leases, retries,
coalescing, DLQ behavior, replay, and projection repair. It also has several
storage shapes: Chroma collections, PostgreSQL/pgvector tables, an in-memory
test backend, and SQLite metadata/index-job/FTS facilities. These do not yet
provide one complete two-stage contract.

This ADR separates canonical event-sourced state from *two disposable serving
projections*. It corrects the earlier ambiguity that Stage 1 could be a second,
permanent canonical store.

## Decision

Kogwistar will eventually expose:

```python
GraphKnowledgeEngine(persistence_mode="single_stage")
GraphKnowledgeEngine(persistence_mode="two_stage")
```

`single_stage` remains default and preserves present behavior. `two_stage` is
opt-in and may be enabled only for a backend arrangement that passes the full
capability contract below. Unsupported arrangements fail configuration; they
must not silently embed synchronously.

`add_node()` keeps its public signature. A later engine option and
`KOGWISTAR_PERSISTENCE_MODE` may select the mode after validation.

## Authority and Projection Model

Canonical entity events are intended recovery authority:

```text
canonical event-sourced state
   -> canonical read materialization, when required
   -> Stage 1 transient metadata/reference projection
   -> Stage 2 semantic/vector projection
```

Stage 1 and Stage 2 are projections of one canonical entity revision. Neither
is canonical truth. Both may be discarded and rebuilt from canonical events.

Exclusive Projection Residency

For the same canonical entity revision, Stage 1 and Stage 2 MUST NOT be simultaneously query-visible. At any point in time, zero or one serving projection may be query-visible; never both.

Promotion is therefore a logical handoff, not a period of dual serving:

canonical current revision
    |
    +--> Stage 1 visible
    |       Stage 2 not eligible
    |
    +--> promotion
    |
    +--> Stage 2 visible
            Stage 1 not eligible
            |
            +--> physical Stage-1 cleanup

The implementation MUST establish the readiness transition so that Stage 1 becomes ineligible before, or atomically with, Stage 2 becoming eligible according to the capabilities of the selected backend arrangement.

Physical copies MAY temporarily coexist during crash recovery or cross-store reconciliation. Such physical overlap MUST NOT make both copies query-visible. The projection registry/readiness gate is authoritative for serving eligibility during this interval.

The invariant is therefore serving exclusivity, not physical-storage exclusivity.

There is deliberately no requirement that one projection is always visible. During deletion, invalidation, or recovery, zero projections may be visible:

canonical deleted
    -> Stage 1 ineligible/removed
    -> Stage 2 ineligible/removed
    -> zero serving projections

Stage 1 is a small, bounded, short-lived, non-semantic projection. It MUST NOT become a permanent node store or an immortal tombstone store.

Stage 2 is the longer-lived semantic/vector serving projection and remains rebuildable from canonical state/events.

The canonical read path remains independent of this handoff. ID lookup, payload retrieval, graph traversal, and last/next traversal MUST remain available through canonical read materialization or another explicitly designated durable read projection after Stage 1 is removed.

A nullable vector, a physically present row, or an existing projection record does not by itself make that projection query-eligible.

This does **not** permit removal of the normal canonical read path. ID lookup,
payload retrieval, graph traversal, and last/next traversal must remain
available after Stage 1 cleanup through a canonical read materialization or an
explicit Stage-2 serving representation. A vector row with `NULL` embedding is
not automatically either of those things.

Current repository boundary: non-native node/edge admission and lifecycle
tombstone/redirect mutations now append their required canonical event before
writing the backend projection. A projection failure can therefore be repaired
from the event, while event-store failure prevents an unlogged projection.
Native authority paths retain their own transactional mutation boundary. These
paths are not one uniform cross-store atomic commit, so full event-sourced
reconstruction remains a required prerequisite for claiming this ADR's recovery
guarantees for every path.

## Query Semantics

Before promotion:

| Query | Stage 1 pending | Stage 2 ready |
| --- | --- | --- |
| ID/payload/graph/reference/last-next | yes, normal canonical or Stage-1 path | yes, normal post-promotion path |
| metadata/reference query | Stage 1 may answer | normal canonical or Stage-2 appropriate path |
| semantic/vector/HNSW/FTS/hybrid | no | yes, Stage 2 only |
| Stage 1 representation | present | absent |

Semantic eligibility requires a Stage-2 projection whose captured canonical
revision/fingerprint matches current canonical state. Presence of a node,
nullable embedding, or completed unrelated job is insufficient.

All semantic surfaces obey this rule: `backend.node_query`, `node_index`,
SQLite `semantic_index`/FTS, hybrid search, and resolver/API wrappers. No
direct backend path may bypass readiness/revision filtering.

## Projection Lifecycle

`index_jobs` records asynchronous work, not canonical entity lifecycle:

```text
canonical current
  -> stage1_pending
  -> stage2_pending
  -> stage2_ready and Stage 1 absent

canonical_deleted
  -> Stage 1 absent
  -> Stage 2 absent or semantically ineligible
```

Job terms map to existing durable statuses:

```text
PENDING -> DOING/leased -> DONE
                    -> PENDING  (retry/backoff)
                    -> FAILED   (retry budget/DLQ)
```

No mandatory canonical `embedding_status`, `embedding_error`, or permanent
Stage-1 status field is introduced. Job payload plus projection records carry
the source revision/fingerprint and embedding model/version where needed.

### Promotion

`node_embedding` is first job kind. It may batch compatible jobs, generates
embeddings, and writes all enabled Stage-2 semantic surfaces idempotently.
Only after current-revision Stage 2 succeeds may Stage 1 be removed.

Physical atomicity across stores is not assumed and distributed 2PC is rejected.
Logical completion is:

```text
current Stage 2 visible + obsolete Stage 1 absent
```

The logical readiness transition MUST make Stage 1 ineligible before or in the
same durable decision that makes Stage 2 eligible. Physical Stage-1 deletion
may follow through reconciliation, but cannot leave both query-visible.

If process stops after Stage-2 write but before Stage-1 cleanup, reconciliation
detects matching current Stage 2 and removes Stage 1. If it stops before
Stage-2 write, Stage 1 remains temporary and job/scanner recovery retries.
Both writes and cleanup must be idempotent.

### Revision races and deletion

Every job carries:

```text
node_id + canonical_revision_or_fingerprint + embedding_model/version
```

Before visible Stage-2 write and before acknowledgement, worker compares this
identity with current canonical state. Mismatch means superseded work:

```text
v1 job completes after v2 current     -> cannot expose v1
v1 job completes after canonical delete -> cannot resurrect v1
```

Deletion is canonical event/state, not a Stage-1 tombstone. A delete/revision
reconciliation action removes Stage 1 and invalidates/deletes Stage 2. A
durable delete job may exist until applied, but neither Stage 1 nor Stage 2
needs a permanent tombstone merely to preserve deletion history. Canonical
events retain that history.

```text
Stage 1 exists
  -> canonical delete event/state
  -> remove Stage 1
  -> delete/invalidate Stage 2, if present
  -> stale embedding worker is rejected by revision/deletion check
```

Scanner/replay must reconstruct missing upsert *and deletion* projection work
from canonical current state and events. It must not retain Stage-1 rows as
deletion memory.

## Reused Primitives and Required Seams

Reuse `entity_events`, `index_jobs`, lease/retry/coalescing/DLQ, existing UOW
where physically applicable, search adapters, replay, and reconciliation. No
second event store, queue, or embedding-specific canonical store is added.

Required generic seams:

- `node_embedding` handler receives parsed `IndexJob.payload_json`; unknown
  kinds fail/retry rather than returning and becoming `DONE`.
- canonical revision/fingerprint and deletion visibility can be read reliably
  by worker and scanner.
- The initial generic read seam folds the existing `entity_events` rows for one
  entity into latest `revision`, `active|deleted` state, and payload. It fails
  closed on malformed payloads or unknown operations; it is the authority for
  future Stage-2 promotion/delete gates, not a new materialized state store.
  Existing Phase-1 join-index fingerprints retain their backend-compatible
  behavior until a complete two-stage adapter adopts this event-derived seam.
- a projection registry/readiness check gates every semantic surface.
- a bounded scanner reconstructs missing upsert/delete work and removes stale
  Stage 1 after successful promotion.
- canonical write/event semantics become sufficient to replay current state,
  including deletion, before this ADR claims recovery authority.

The current inline `reconcile_indexes(max_jobs=50)` must not synchronously drain
`node_embedding` in two-stage mode.

## Two-Stage Capability and Adapter Contract

`two_stage` is not a property inferred from a nullable vector column. Engine
configuration MUST obtain both an immutable capability descriptor and an
executable two-stage projection adapter from the selected projection
arrangement. It MUST reject the mode unless every required property is true and
the adapter is present; it MUST NOT fall back to the legacy synchronous
embedding/write path.

The adapter is arrangement-specific. A Chroma deployment may use a dedicated
SQLite Stage-1 projection; PostgreSQL/pgvector and in-memory deployments use
their own co-located or test-only adapters and do not inherit that SQLite
staging store. SQLite is therefore one possible Stage-1 implementation, not a
universal two-stage dependency or a property of `StorageBackend`.

Required descriptor properties:

```text
canonical_event_replay
canonical_read
stage1_strategy: none | transient_projection
stage1_metadata_query          # required when strategy is transient_projection
stage2_semantic_projection
revision_gated_promotion
semantic_readiness_gate
delete_reconciliation
atomic_promotion: same_store | eventual_reconcile
stage1_cleanup                 # required when strategy is transient_projection
```

`canonical_event_replay` means canonical create/update/delete history can
reconstruct the current projection state; it is not satisfied by a best-effort
event append after a separate backend write. `canonical_read` is deliberately
separate from Stage 1. It is the durable
read path for ID, payload, graph, and reference access after Stage 1 has been
removed. `atomic_promotion` describes physical topology, not a promise of
distributed 2PC: `same_store` may use one local transaction; cross-store
arrangements use `eventual_reconcile` with idempotent writes and cleanup.

`stage1_metadata_query` is required only where an actual transient Stage-1
projection is exposed. It proves the Kogwistar metadata/reference query
contract, not merely that the backing store has JSON columns.

The existing `StorageBackend` node/index verbs remain the common low-level
surface. They MUST NOT be expanded so every backend pretends to host Stage 1.
An arrangement that claims `two_stage` implements an executable two-stage
projection adapter, conceptually:

```text
stage1_upsert(revision, payload, metadata, references)
stage1_query(query contract)
promote_stage2(revision, embedding, projection payload)
remove_stage1(revision)
remove_stage2_or_invalidate(revision/delete)
reconcile_projection(canonical current state/events)
apply_embedding_job(entity_kind, entity_id, op, payload_json)
```

Method names and exact Python type remain implementation detail. Their
semantics are mandatory: promotion validates current revision before visibility;
cleanup is idempotent; delete rejects stale promotion; reconciliation converges
after crashes. Admission methods MUST run only after required canonical event
append. The embedding-job handler is part of the executable adapter contract;
having only Stage-1 admission methods is insufficient. `index_jobs` remains
worker coordination and supplies job payload; it is not replaced by this
adapter.

The adapter contract is common, but its physical implementation is
backend-specific. SQLite is not a universal Stage-1 store: a Chroma deployment
may use it because Stage 2 is external, while pgvector and in-memory adapters
own their respective storage and lifecycle implementations.

Backend profiles:

```text
pgvector co-located arrangement:
  stage1_strategy = transient_projection
  canonical_read = relational node/edge read projection after handoff
  atomic_promotion = same_store

Chroma external-vector arrangement:
  stage1_strategy = transient_projection
  canonical_read = event-derived read materialization
  atomic_promotion = eventual_reconcile
```

The pgvector profile does not delete its relational node/edge row after vector
write: that row is the post-handoff Stage-2/read representation, not Stage 1.
PG Stage 1 uses a separate high-churn `gke_stage1_projections` table and the
event plus Stage-1 admission are committed in one PostgreSQL UOW. Promotion
embeds outside the transaction, then rechecks canonical revision and writes
pgvector plus deletes Stage 1 in one local transaction. Chroma may use a dedicated SQLite Stage-1
projection file/table with its own high-churn retention and maintenance policy.
It may reuse the SQLite projection-row/UOW/JSON/CAS implementation behind
generic named projections, but MUST NOT share the `named_projections` table or
pretend that a namespace alone provides isolation. Current named-projection
payload persistence alone is insufficient. The implemented Chroma arrangement
uses the dedicated `stage1_node_projections` table instead; it reuses
projection-row mechanics, not the named-projection table or its mixed-purpose
namespace. Physical Stage-1 keys isolate node and edge identifiers.

## Backend Review

Low-level physical possibility is not full two-stage support. A backend is
capable only if it satisfies all: no Stage-1 embedding; metadata/reference
queryability; delayed Stage 2; readiness exclusion; revision gate; Stage-1
cleanup; delete propagation; stale-worker rejection; and crash reconciliation.

| Backend | Physical support now | Stage 1 / metadata-query basis | Stage 2 basis | Current complete capability |
| --- | --- | --- | --- | --- |
| PostgreSQL/pgvector | dedicated `gke_stage1_projections` plus relational vector tables | Stage-1 flat metadata equality; normal relational read after handoff | pgvector HNSW | supported for synchronous Python PG arrangement; async/native remain unsupported |
| SQLite | `entity_events`, `index_jobs`, JSON-capable SQLite, FTS tables | dedicated `stage1_node_projections` table with narrow equality query seam | current FTS is search projection, not Stage 1 | unsupported pending promotion/read routing/reconciliation |
| ChromaDB | explicit collections and later update APIs | dedicated SQLite `stage1_node_projections` table in the engine arrangement; narrow flat-metadata query | Chroma/HNSW via explicit embedding promotion | supported only for SQLite+Chroma arrangement; bare Chroma remains unsupported |
| In-memory | same volatile row: `embedding=None` then update | row storage/get | in-memory vector query with readiness exclusion | complete test arrangement; volatile only, not durable recovery |
| Rust/native Postgres paths | native projection writes require embeddings today | no staged projection contract or `node_embedding` worker integration | native/Postgres projection | unsupported now |

### PostgreSQL/pgvector

`embedding` being nullable proves only physical representation of no vector.
The implemented synchronous Python arrangement adds a distinct
`gke_stage1_projections` table for short-lived metadata/reference reads. Canonical
event append and Stage-1 admission share one PostgreSQL UOW; `node_embedding`
promotion rechecks event-derived revision, writes pgvector, and deletes Stage 1
in the same local transaction. Delete-before-promotion cleanup and stale-job
rejection are covered by live pgvector acceptance tests. Async PostgreSQL and
Rust/native authority do not inherit this adapter and remain unsupported.

### SQLite

SQLite currently serves metadata/event/job infrastructure. Its `semantic_index`
and FTS5 tables are derived text-search projections; they are not a general
node Stage-1 metadata store. A dedicated high-churn
`stage1_node_projections` table now provides ID/namespace and flat metadata
equality querying through `query_stage1_node_projections()`. Nested operators
are explicitly rejected; this is a narrow adapter seam, not yet full parity
with backend `where` semantics. JSON support alone does not define equality,
scalar, nested, array, namespace/tenant, or filtering semantics compatible with
existing backend `where` APIs.

Option A remains viable for a Chroma profile: a dedicated high-churn SQLite
Stage-1 file/table plus the narrow Kogwistar metadata query adapter, then Chroma
Stage 2. The table may be batch-cleared or rebuilt from canonical state; it
retains no deletion history. It reuses SQLite storage mechanics but is not a
generic named-projection namespace. Its query contract remains intentionally
narrow until full Chroma-facing parity is designed and tested; SQLite JSON is
implementation detail, not API contract. SQLite
maintenance may use cheap query-planner optimization routinely and controlled
rebuild/compaction only for the dedicated Stage-1 store, never as a reason to
truncate canonical events or shared job metadata.

### ChromaDB

Current `nodes` and `nodes_index` collections use
`embedding_function=self._ef`. Current write path embeds first; a simple
omitted embedding therefore cannot be claimed as non-semantic Stage 1. The
implemented Chroma arrangement does not write Chroma during Stage 1. It stores
payload in the dedicated SQLite `stage1_node_projections` table, then promotes
with an explicit embedding into the existing Chroma collection and removes
Stage 1. This support is arrangement-specific: a bare `ChromaBackend` still
rejects `two_stage` and does not automatically acquire SQLite staging. Future
implementations may choose either:

- a dedicated SQLite (or another explicit) Stage-1 metadata/reference
  projection and Chroma/HNSW Stage 2; or
- a Chroma staging collection/path guaranteed to have no automatic embedding,
  plus separate vector Stage 2.

Either must prove no provider call during Stage 1, exclusive projection
residency, cleanup, revision/delete gates, and replay/reconciliation. The
current SQLite+Chroma arrangement satisfies the declared narrow contract;
full metadata-query parity and stronger background reconciliation remain
future work.

### In-memory

The in-memory backend now provides a complete **volatile test arrangement** for
the declared contract: Stage 1 is a row with `embedding=None`, ID/payload reads
remain available, semantic query excludes the row, and the existing
`node_embedding` index job promotes it after a canonical-source fingerprint
check. Missing/deleted or stale revisions are not promoted. This validates
deterministic lifecycle and race semantics only; it does not establish durable
crash recovery or production persistence guarantees.

### Rust/native paths

Current native PostgreSQL projection APIs receive concrete embeddings and
enqueue only existing join-index jobs. No native Stage-1/Stage-2 lifecycle,
revision gate, or `node_embedding` job handler is present. Native support is
explicit future work, not inherited from Python PostgreSQL capability.

## Failure, Batching, and Backpressure

Stage 1 failure rejects canonical admission only when canonical commit itself
fails. Embedding/provider/index failure never invalidates canonical truth.
Pending/failed Stage-2 work remains non-semantic and recoverable through
existing queue retry/DLQ rules.

Worker claims bounded compatible `node_embedding` jobs, grouping by model,
version, namespace/tenant, and provider limits. One bad member cannot block
others; successful members promote independently, failed members retry or DLQ.
Acknowledgement occurs only after current Stage 2 write and required Stage-1
cleanup/reconciliation intent are durable enough for idempotent convergence.

## Recovery and Rebuild

Canonical events/current canonical state reconstruct both projections and their
deletion effects. `index_jobs` aid continuation but are not sole repair source.
Normal scanner must detect:

- current canonical revision lacking Stage 2 or required Stage 1 work;
- Stage 2 whose revision no longer matches canonical state;
- Stage 1 remaining after matching Stage-2 success; and
- Stage 2 remaining for canonical deletion/tombstone.

Until canonical event append and current-state reconstruction are made atomic
enough per backend, this is an implementation constraint, not a guaranteed
property of current code.

## Cross-ADR Interaction

ADR-017 observes lifecycle only; OTel never gates commit, job acknowledgement,
or projection readiness. ADR-019 workflows may create nodes; workflow success
depends on canonical write unless ordinary business policy explicitly waits for
Stage 2 readiness.

## Required Acceptance Tests

Any backend claiming `two_stage` must pass equivalent lifecycle tests:

- canonical create -> Stage 1 metadata/reference query yes; semantic query no;
- successful promotion -> Stage 2 semantic query yes; Stage 1 absent;
- crash before Stage 2 -> job/scanner recovers, then Stage 1 removed;
- crash after Stage 2 before cleanup -> reconciliation removes Stage 1;
- delete during Stage 1 -> no Stage-1 tombstone; Stage 2 absent/ineligible;
- delete during v1 embedding -> worker cannot resurrect v1;
- v1 job A, v2 job B, A finishes late -> v1 never becomes visible;
- lost enqueue -> scanner reconstructs needed work from canonical state/events;
- replay/rebuild reconstructs current Stage 2 and deletions without retaining
  Stage 1; and
- every exposed vector/HNSW/FTS/hybrid/resolver path excludes pending, failed,
  stale, and deleted projections.

`single_stage` compatibility tests remain required. In-memory may cover
deterministic semantics but not durability; production claims require durable
backend parity.

## Backward Compatibility

`single_stage` is the backward-compatible default. Existing deployments that do not explicitly enable `two_stage` MUST retain the current single-stage write and serving semantics.

In particular:

* `GraphKnowledgeEngine()` without an explicit persistence mode MUST behave as `single_stage`.
* Existing `add_node()` callers require no API or signature change.
* Existing synchronous embedding behavior remains unchanged in `single_stage`.
* Existing backend configurations remain valid unless they explicitly opt into `two_stage`.
* Existing persisted data MUST NOT require migration merely to continue operating in `single_stage`.
* `two_stage` MUST NOT be silently selected because a backend happens to support nullable embeddings, asynchronous jobs, or another partial capability.
* Switching to `two_stage` is an explicit deployment/configuration decision and MUST pass the complete capability contract before accepting writes.
* A failed or unsupported `two_stage` capability check MUST fail configuration rather than silently falling back to `single_stage`, unless the caller explicitly requests such fallback behavior.
* `single_stage` and `two_stage` MUST preserve the same public entity/write API and canonical entity semantics; the difference is the timing and lifecycle of derived semantic projections.
* Semantic search results in `two_stage` may temporarily exclude newly admitted entities until Stage 2 becomes ready. This is an intentional consistency characteristic of `two_stage`, not a change to canonical entity existence.
* Existing applications that require synchronous semantic availability MUST continue using `single_stage` unless they explicitly adopt and handle two-stage eventual semantic readiness.

The migration from `single_stage` to `two_stage` is therefore an explicit operational transition, not an implicit behavioral change caused by upgrading Kogwistar.

No requirement is imposed that an existing `single_stage` projection be converted in place into Stage 1. Existing projections may continue operating under `single_stage` until an explicit migration/rebuild procedure for the selected backend is defined and executed.

## Non-Goals

- permanent Stage-1 tombstone/state table;
- second event store or durable queue;
- embedding-specific canonical lifecycle fields;
- distributed 2PC;
- backend-specific Goal logic;
- changing `add_node()` signature; or
- treating OpenTelemetry as persistence/CDC authority.

## Future Work

- make canonical event/current-state commit and replay guarantees explicit per
  backend;
- projection registry and readiness API;
- generic SQLite metadata Stage-1 adapter, if selected;
- broader Chroma metadata-query parity and background reconciliation;
- PostgreSQL and Rust/native staged-projection implementation;
- model/version migration and controlled historic re-embedding.
