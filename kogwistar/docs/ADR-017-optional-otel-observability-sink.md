# ADR-017: Optional OpenTelemetry Observability Sink

**Status:** Proposed
**Date:** 2026-08-30
**Owner:** Maintainers

## Context

Runtime telemetry currently flows through `TraceContext`, `EventEmitter`, and
one `SQLiteEventSink`. `EventEmitter` emits a structured event dictionary; the
SQLite sink owns a bounded background queue and durable trace writes. Runtime
already emits workflow lifecycle, step-attempt, checkpoint, routing, join, and
token events. Current `TraceContext` defaults derive pseudo IDs from run/token/
step strings; they are not W3C trace identifiers and must not be exported as
native OTel IDs.

Kogwistar needs optional OpenTelemetry (OTel) export without making an
observability vendor, exporter, or network path authoritative for workflow,
knowledge, queue, recovery, or replay state.

## Decision

OTel is an optional, best-effort observability projection behind the existing
runtime telemetry path. Kogwistar owns event truth; OTel observes Kogwistar.

`TraceContext` is Kogwistar's SDK-independent W3C-compatible runtime trace
carrier. It now provides opt-in `new_root()`/`child_span()` constructors and
validation for lowercase-hex identifiers:

```text
trace_id        32 hex characters (128-bit)
span_id         16 hex characters (64-bit)
parent_span_id  16 hex characters when present
```

These are trace identifiers, distinct from domain identifiers:

```text
trace:  trace_id, span_id, parent_span_id
domain: goal_id, run_id, token_id, node_id, step_seq,
        conversation_id, turn_node_id
```

`run_id` never becomes `trace_id`. In the current adapter, the OTel SDK owns
native span-context ID generation; validated Kogwistar trace fields are emitted
as `kogwistar.*` attributes and are not injected as native OTel IDs. Core runtime
remains independent of the OTel SDK/package.

The first phase projects workflow lifecycle events only:

```text
workflow_run_started
step_attempt_started
step_attempt_completed
workflow_run_completed
workflow_run_failed
workflow_run_cancelled
workflow_run_suspended
checkpoint_saved
```

No canonical event-store, outbox, CDC, replay, queue acknowledgement, or
recovery authority changes in this ADR.

## Architecture

`EventEmitter` keeps its existing event-dictionary format and event names. The
smallest required extension is a structural sink contract:

```text
emit(event_dict) -> None
flush(timeout?) -> bool
close(timeout?) -> None
```

`flush()` is best-effort and bounded; `True` means work queued before its
barrier was processed, while `False` means timeout/failure and has no effect on
runtime truth. SQLite implements a bounded commit barrier and OTel implements a
bounded queue drain. `close()` may call it internally. A sink lacking buffered
work may implement it as a no-op. This is lifecycle management for a small sink
contract, not a generic event-bus API.

`EventEmitter` must accept either one sink or a small `FanoutEventSink`. The
fan-out object iterates configured sinks and isolates each sink failure:

```text
EventEmitter
  -> FanoutEventSink
       -> SQLiteEventSink
       -> OpenTelemetryEventSink (optional)
```

This is not a generic event bus: it neither routes domain events nor owns
durability, ordering, retries, subscriptions, or canonical state.

`OpenTelemetryEventSink` lives in a separate optional module such as
`kogwistar.runtime.telemetry_otel`. Core `TraceContext` imports no OTel package.

### Trace propagation

The runtime must carry one validated `TraceContext` through `run`, scheduler,
`StepContext`, nested invocation, checkpoint metadata, ordinary resume,
suspended-client resume, and the Rust runtime-authority/remote-resume boundary.
The constructors/validator are implemented, but this remains a required
runtime integration seam: current runtime still creates fresh TraceContext
values at many emission sites and does not yet propagate one root context.

```text
top-level workflow -> new trace_id + root span_id
nested workflow    -> same trace_id + child span_id + parent span_id
Goal action run    -> Goal trace_id + new workflow-run span_id
resume             -> same trace_id + new continuation span_id,
                      parent_span_id = prior execution span when known
```

Minimum persisted trace state for resume is `trace_id` plus the prior
`run_execution_span_id`. `resume_from_latest_checkpoint()` can then create a
new continuation span whose parent is that prior execution span. Persisting a
transient current step span is unnecessary; `parent_span_id` is otherwise
derivable from the run execution record. Checkpoint metadata must contain this
minimum because it is the standalone input to ordinary resume.

Future HTTP/MCP transports may propagate a W3C `traceparent`; they do not alter
Goal/run domain semantics.

## Invariants

- OTel is never canonical knowledge-plane, control-plane, or workflow truth.
- An OTel import, enqueue, export, shutdown, or exporter failure never fails
  `EventEmitter.emit()` or a canonical workflow/node write.
- OTel queue pressure drops telemetry according to explicit policy; it never
  blocks a producer indefinitely.
- Existing Kogwistar event dictionaries and event names remain stable.
- `TraceContext` trace IDs are W3C-valid when the opt-in propagation seam is
  used. The OTel adapter never translates domain IDs into trace IDs; Phase 1
  records Kogwistar trace fields as attributes while the SDK creates native IDs.
- `event_id`, `goal_id`, `run_id`, `token_id`, `step_seq`, `node_id`, `attempt`,
  `conversation_id`, and `turn_node_id` remain OTel attributes.

## Span Lifecycle

Top-level workflow run is an OTel root execution span. A step attempt is a
child span of its workflow execution span. Phase 1 preserves Kogwistar trace
continuity as attributes; because the SDK owns native ID generation, separate
nested or Goal-invoked run spans are not promised to share one native OTel
trace. A later SDK integration may establish that mapping without changing
Kogwistar authority.
Predicate, routing, join, token, and checkpoint data are span events or
attributes unless a later ADR proves a separate span useful.

| Kogwistar event | OTel action |
| --- | --- |
| `workflow_run_started` | start root span if no propagated parent; otherwise child execution span |
| `step_attempt_started` | start child span |
| `step_attempt_completed` | end child span; record status/duration |
| `checkpoint_saved` | add event/attributes to current run span |
| `workflow_run_completed` | end root span with success |
| `workflow_run_failed` | record exception/status; end root span with error |
| `workflow_run_cancelled` | end root span with cancelled status/attribute |
| `workflow_run_suspended` | end current execution span with suspended attribute |
| resumed execution | begin a new continuation span; link to prior run execution when known |

A suspended span must not rely on an in-memory span object surviving restart.
Persisted W3C trace identity lets resume create a new continuation span in the
same trace. `run_id` remains a domain attribute and does not become trace
identity.

## Persistence, Failure, and Recovery

The OTel sink owns a bounded queue, background exporter worker, batch/flush
policy, and bounded shutdown timeout. Queue-full behavior is explicit:

```text
queue full -> increment local drop counter / log sampled warning -> drop event
```

The sink catches exporter and SDK exceptions internally. `FanoutEventSink` also
catches a child-sink exception so a defective optional sink cannot affect SQLite
telemetry or runtime execution. Shutdown is best-effort; timeout returns control
to the caller and may drop unflushed OTel events.

OTel exporter objects and span handles are process-local. On restart, runtime
loads persisted W3C trace identity for Kogwistar correlation and emits fresh
SDK-owned OTel spans; OTel still does not become authority for runtime or
canonical state. Preserving supplied runtime IDs as native OTel IDs would
require a separately tested SDK integration and is deferred; Phase 1 does not
promise that equivalence.

## Interaction with Existing Runtime

`WorkflowRuntime` already emits the first-phase lifecycle events through
`EventEmitter`. `TraceContext` remains a cheap correlation carrier. The adapter
consumes emitted dictionaries after runtime has formed them. Runtime will extend
workflow-run/checkpoint metadata with minimal trace continuation fields, but
this does not alter resolver execution, checkpoint authority, or replay state
semantics.

## Cross-ADR Interaction

ADR-018 may emit ordinary runtime/index-worker telemetry in a later phase, but
its Stage 1 node commit and Stage 2 readiness are never gated on OTel. ADR-019
uses one Kogwistar W3C correlation trace across Goal control and its workflow
runs when propagation is supplied, but Phase 1 OTel native spans need not
share that trace. Goal control-plane records remain authoritative when OTel is
unavailable, delayed, or dropped.

## Consequences

- SQLite telemetry and optional OTel telemetry coexist from the same emitted
  event dictionary.
- OTel requires only a small fan-out seam, not a new observability hierarchy.
- Suspended/resumed workflows receive finite, restart-safe execution spans.
- Observability can be incomplete under pressure by design.

## Alternatives

### Instrument `TraceContext` directly

Rejected. `TraceContext` owns vendor-neutral W3C trace identity, but must not
import OTel SDK calls, exporters, queues, or vendor policy.

### Replace SQLite telemetry with OTel

Rejected. SQLite remains an existing local telemetry sink; OTel is optional.

### Generic event bus / canonical OTel sink

Rejected. Canonical event distribution and authority are separate concerns.

## Non-Goals

- OTel as canonical event store, CDC mechanism, or replay source;
- RunRegistry, indexing, recovery, job-queue, and metrics/log bridges;
- synchronous network export;
- durable reconstruction of OTel spans after restart;
- changing runtime event names, or importing OTel SDK/exporter behavior into
  `TraceContext`.

## Future Extensions

- index-worker, recovery, queue, and Goal lifecycle telemetry;
- richer OTel logs/metrics mapping;
- configurable sampling and exporter batching;
- links between parent/child workflow execution spans.

## Acceptance Criteria

- Kogwistar imports and runs with OTel absent.
- SQLite and OTel sinks receive the same event dictionary when both enabled.
- workflow, step, completion, failure, cancellation, suspension, and resume
  lifecycle mappings are tested.
- OTel exporter failure and queue saturation do not fail `EventEmitter.emit()`.
- queue drop and bounded shutdown behavior are tested.
