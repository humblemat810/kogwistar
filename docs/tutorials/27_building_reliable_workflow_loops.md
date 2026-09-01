# 27 Building Reliable Workflow Loops

This tutorial covers the semantics recorded in ADR-017, ADR-018, and ADR-019.

Audience: advanced contributors

This ladder teaches the three recent architectural decisions as one system.
Read one level at a time. Each level adds one semantic boundary and one
failure question.

## The Authority Rule

Keep this order in mind throughout the tutorial:

```text
workflow/runtime state
        -> canonical graph/entity state
        -> derived indexing projections
        -> OpenTelemetry observation
```

OTel observes execution. Stage 1 and Stage 2 serve derived data. Neither can
replace canonical state, workflow checkpoints, or event history.

## Level 0: Map the Three ADRs

Read:

- [ADR-017: Optional OpenTelemetry Sink](../../kogwistar/docs/ADR-017-optional-otel-observability-sink.md)
- [ADR-018: Two-Stage Semantic Indexing](../../kogwistar/docs/ADR-018-two-stage-persistence-and-async-semantic-indexing.md)
- [ADR-019: Cyclic Workflow Pattern](../../kogwistar/docs/ADR-019-goal-mode-reactive-control-over-workflow-runtime.md)

Classify each surface before changing code:

| Surface | Role | Authoritative? |
| --- | --- | --- |
| `WorkflowRun` and checkpoints | workflow control and recovery | yes for runtime state |
| canonical events/entity state | graph/entity truth | yes for knowledge state |
| Stage 1 | temporary metadata/reference projection | no |
| Stage 2 | semantic/vector serving projection | no |
| OTel sink | external observation | no |
| Goal-shaped workflow graph | ordinary workflow control pattern | runtime uses existing semantics |

Checkpoint: state which surface can be deleted and rebuilt. The answer is
Stage 1, Stage 2, and OTel data; canonical events and required workflow state
must remain recoverable.

## Level 1: ADR-017, Observe Without Becoming Truth

`TraceContext` carries correlation. It does not own workflow meaning.
`EventEmitter` emits the existing runtime event dictionaries to sinks:

```text
TraceContext -> EventEmitter -> SQLite sink
                         \-> optional OTel sink
```

Rules:

- `run_id`, `goal_id`, `token_id`, and `node_id` are domain attributes, not OTel trace IDs.
- Nested workflows keep one trace and receive distinct run/span identity.
- Resume continues the trace with a new continuation span; transient span objects are not persisted.
- OTel export is bounded, asynchronous, best-effort, and never blocks canonical execution indefinitely.
- exporter failure, queue drop, or shutdown timeout must not fail `EventEmitter.emit()`.

Try the focused tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/runtime/test_telemetry_otel.py -m ci -q
.venv\Scripts\python.exe -m pytest tests/integration/test_telemetry_otel_collector.py -m ci_full -q
```

The collector test validates the adapter boundary. It does not make OTel an
event store, CDC stream, replay source, or recovery authority.

## Level 2: ADR-018, Separate Canonical State From Semantic Readiness

Two-stage mode is a projection handoff:

```text
canonical revision
      -> Stage 1: metadata/reference visible, semantic invisible
      -> Stage 2: semantic visible, Stage 1 no longer query-visible
```

For the same canonical revision:

> Stage 1 and Stage 2 MUST NEVER be simultaneously query-visible.

This is serving exclusivity. Physical cleanup may require reconciliation after
a crash, but readiness gates must prevent dual visibility during that window.

Before Stage 2 promotion:

- ID lookup, payload retrieval, graph traversal, and last/next references work.
- vector, HNSW, FTS, and hybrid semantic search exclude the entity.

After promotion:

- semantic search uses only the current Stage 2 projection;
- Stage 1 is removed or made ineligible and is cleaned up idempotently.

Deletion is canonical state, not a permanent Stage 1 tombstone:

```text
canonical delete -> remove/invalidate Stage 2 -> remove Stage 1
```

An old embedding job must not resurrect a deleted or superseded revision.
Workers compare the captured revision/fingerprint with current canonical state.

## Level 3: Index Jobs, Batch Efficiency Without Batch Transactions

One queue claim may contain mixed jobs:

```text
claim
  -> compatible node_embedding groups
       -> provider calls, max batch_size each
  -> non-embedding jobs
       -> existing handler, one job at a time
```

Embedding groups are processed first. Group compatibility includes model,
version, provider, namespace/tenant, and provider limits. A smaller available
group runs immediately; the worker does not wait to fill it.

Critical distinction:

```text
batch = provider/throughput boundary
batch != transaction boundary
```

Each member has independent promotion, ACK, retry, DLQ, and reconciliation.
PostgreSQL may use a local UOW for one member; this does not make the whole
provider batch one SQL transaction. Chroma and other cross-store arrangements
expect partial completion and eventual convergence.

Run the deterministic worker ladder:

```powershell
.venv\Scripts\python.exe -m pytest tests/outbox/test_phase5_worker_backpressure_unit_fake.py -m ci -q
.venv\Scripts\python.exe -m pytest tests/outbox/test_phase5_worker_negative_eventual_consistency.py -m ci_full -q
```

Inspect these assertions:

- mixed embedding/non-embedding claims are partitioned;
- non-embedding work uses the individual handler;
- adapter failure becomes per-job retry/failure;
- partial batch outcomes ACK successful members independently;
- ACK failure leaves work recoverable through lease/retry.

## Level 4: ADR-019, Goal As Ordinary Cyclic Workflow

Goal Mode adds business meaning, not a second engine:

```text
Observe -> Decide -> Act -> Check
   ^                       |
   |                       +-> satisfied -> Done
   +------ not satisfied --+
```

The meanings stay distinct:

- Observe gathers evidence into ordinary workflow state.
- Decide selects the next action.
- Act invokes one ordinary workflow, static or dynamic.
- Check answers only whether the objective is satisfied.

`WorkflowRuntime` still owns nodes, edges, predicates, tokens, fanout, joins,
nested invoke-and-await, checkpoints, and resume. Goal is therefore best
understood as an annotated cyclic workflow pattern for UI, validation,
provenance, and tooling. The annotation must not select a new scheduler,
token model, resolver family, or recovery engine.

For v1, Act means invoke-and-await. Durable background dispatch is a separate
future capability.

## Level 5: Cross-ADR Execution

Combine the boundaries in this order:

1. A goal-shaped workflow executes through ordinary `WorkflowRuntime`.
2. Its Act invokes a normal child workflow.
3. The child writes canonical graph state.
4. Two-stage arrangements expose temporary Stage 1 metadata/reference access.
5. `index_jobs` later promote compatible embeddings in bounded batches.
6. OTel observes run, step, checkpoint, and worker lifecycle events.

Resulting authority rule:

```text
Goal workflow success
  != OTel export success
  != Stage 2 readiness
```

Business logic may explicitly wait for semantic readiness, but that is an
ordinary workflow policy. It must not be hidden inside the OTel sink or index
worker.

## Level 6: Failure Drills

For each scenario, identify the durable recovery authority before considering
the repair:

| Failure | Expected behavior |
| --- | --- |
| OTel exporter down | workflow continues; telemetry is dropped/retried within bounded policy |
| Stage 1 exists, process stops | job/scanner rebuilds Stage 2; Stage 1 is cleaned after promotion |
| Stage 2 written, cleanup not finished | reconciliation removes Stage 1 visibility/rows |
| delete during embedding | stale worker cannot recreate semantic visibility |
| v1 job finishes after v2 | revision gate rejects v1 promotion |
| one batch member fails | successful members remain complete; failed member retries |
| goal controller run resumes | checkpoint and child run identity prevent blind duplicate invocation where supported |

Do not solve any of these by adding a second event store, a Goal scheduler, a
permanent Stage 1 tombstone table, or a transactional guarantee from OTel.

## Implementation Status

The ADRs describe target contracts, not proof that every backend already meets
them. Check backend capability before enabling two-stage mode. In particular,
Chroma cross-store recovery, PostgreSQL-specific staged projection behavior,
and Rust/native parity require capability-specific tests rather than inferred
support from nullable embeddings or a shared API name.

## Next Reading

- [08 Storage Backends and Parity](./08_storage_backends_and_parity.md)
- [09 Indexing Pipeline](./09_indexing_pipeline.md)
- [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md)
- [18 Nested Workflow Invocation](./18_nested_workflow_invocation.md)
- [26 Recovery and Durable Operational State](./26_recovery_and_durable_operational_state.md)
