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

`WorkflowRuntime(..., otel_enabled=True)` is the explicit runtime opt-in. The
default is `False`; no environment variable, installed package, or exporter
configuration silently enables OTel. When enabled, the runtime lazily creates
the optional sink and composes it with the configured SQLite sink. A caller
that supplies an existing `EventEmitter` must compose its sinks before creating
that emitter; the runtime rejects the ambiguous combination rather than
rewiring a shared emitter.

### Trace propagation

The synchronous and asynchronous Python workflow runtimes propagate one
validated `TraceContext` through workflow execution, `StepContext`, nested
invoke-and-await calls, run/checkpoint metadata, and ordinary checkpoint
resume. Lifecycle events derive child contexts from the run context rather
than creating unrelated trace identities. The Rust runtime-authority path and
remote resume boundary are not thereby guaranteed to propagate this context;
each boundary requires explicit parity evidence before being claimed.

```text
top-level workflow -> new trace_id + root span_id
nested workflow    -> same trace_id + child span_id + parent span_id
Goal action run    -> Goal trace_id + new workflow-run span_id
resume             -> same trace_id + persisted run_execution_span_id;
                      subsequent lifecycle events derive child span IDs
```

Minimum persisted Kogwistar correlation state for resume is `trace_id` plus the
prior `run_execution_span_id`; `parent_span_id` is persisted when available.
`resume_from_latest_checkpoint()` reconstructs the run-level correlation context
with that trace ID and execution span ID; newly emitted lifecycle events derive
their own child contexts from it. No transient span object is persisted. This
preserves Kogwistar correlation, but does not by itself restore an OpenTelemetry
SDK span context or prior parent relationship after the previous execution span
has ended.

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
child span of its workflow execution span. For nested runs in the same live
process, the adapter resolves a known in-process parent span and creates a
native child span. Kogwistar W3C fields remain attributes; the SDK generates
native IDs. After the prior execution span ends (including suspension) or the
process restarts, the adapter cannot recover the prior SDK span object or force
its IDs into a new SDK span, so resumed execution starts a new native OTel trace
unless an external OTel context is explicitly propagated.
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
| resumed execution after prior span ended/restart | begin a new SDK-owned trace/span; retain persisted Kogwistar trace/span identifiers as attributes; no OTel span link is currently emitted |

Test scope: `test_otel_resume_uses_same_trace_and_new_continuation_span` feeds
synthetic continuation events directly to the sink. Runtime checkpoint-resume
tests verify persisted Kogwistar IDs, but do not prove end-to-end OTel continuity
or span linking after resume.

A suspended span must not rely on an in-memory span object surviving restart.
Persisted Kogwistar W3C correlation identity lets resume continue the same
Kogwistar trace, but the current adapter does not restore native OTel identity
or emit a link to the prior native span after suspension/restart. `run_id`
remains a domain attribute and does not become trace identity.

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

An explicitly opted-in runtime exposes bounded `close(timeout)` shutdown for
the OTel sink it created. Caller-owned and process-shared sinks remain
caller-owned; closing one runtime must not close a sibling runtime's shared
SQLite sink.

## Interaction with Existing Runtime

`WorkflowRuntime` already emits the first-phase lifecycle events through
`EventEmitter`. `TraceContext` remains a cheap correlation carrier. The adapter
consumes emitted dictionaries after runtime has formed them. Run/checkpoint
metadata carries the minimum Kogwistar trace continuation fields; this does not
alter resolver execution, checkpoint authority, or replay state semantics.

## Cross-ADR Interaction

ADR-018 may emit ordinary runtime/index-worker telemetry in a later phase, but
its Stage 1 node commit and Stage 2 readiness are never gated on OTel. ADR-019
uses one Kogwistar W3C correlation trace across Goal control and nested runs
when propagation is supplied. Same-process nested OTel spans can reflect that
parent-child structure; resumed runs after the prior span ends do not currently
continue the same native OTel trace. Goal control-plane records remain
authoritative when OTel is unavailable, delayed, or dropped.

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
