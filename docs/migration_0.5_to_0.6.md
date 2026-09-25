# Kogwistar 0.5 to 0.6 Migration

Kogwistar 0.6 keeps the graph, workflow, provenance, and storage contracts
backward compatible while adding substrate capabilities needed by downstream
applications. Upgrade the core before updating downstream dependency pins.

## New Substrate Capabilities

### Named-projection batch CAS

Named projection stores can compare and swap a batch of projection entries as
one metadata-store operation. A rejected expected revision leaves the complete
batch unchanged. Use this for coordinated projection publication; do not treat
it as a replacement for graph transactions or source-of-truth events.

### Explicit telemetry

Runtime telemetry is opt-in. Applications that need traces must enable the
runtime telemetry configuration explicitly and close the runtime deterministically
when the application exits. Core does not enable exporters, choose an exporter,
or decide product-level sampling policy on behalf of an application.

### Authority propagation

Workflow runtime calls accept a trusted authority context and carry it into
nested runs. Mutable workflow state and model output cannot grant authority.
Callers must provide the authenticated context at the application boundary and
must keep tenant, project, capability, and budget restrictions bounded.

### Lane-claim filters

Durable queue claims can be restricted by lane and other job dimensions before a
worker claims a job. Workers should use filters matching their assigned work;
unrelated workers must not claim and later discard jobs. Queue ownership and
lease rules remain authoritative.

## Downstream Responsibilities

Kogwistar provides generic graph, workflow, projection, authority-carrier, and
queue primitives only. LLM-Wiki remains responsible for:

- maintenance scheduling and request/background policy;
- memory capture, review, promotion, and proposal acceptance;
- source provenance and application-specific ACL decisions;
- mapping authenticated product claims to a bounded core authority context;
- product telemetry configuration and runtime lifecycle ownership.

The parser remains a stateless reusable parser and does not adopt the agent
harness. Storage adapters may use named-projection CAS only when they advertise
the corresponding metadata-store capability; vector storage contracts are not
required to implement it.

## Upgrade Checklist

1. Update the core pin to the released 0.6 commit or `v0.6.0` tag.
2. Run the complete CPython 3.12, 3.13, and 3.14 suites.
3. Run the PyPy 3.11 suite, Rust checks, PostgreSQL checks, and SQLite checks.
4. For applications using telemetry, enable it explicitly and close runtimes in
   application shutdown paths.
5. For workers, configure lane filters and verify unrelated jobs remain queued.
6. Update downstream pins only after the core release gate is green.

No migration of existing graph data is required. Existing named projections,
workflow records, provenance records, and queue entries remain readable.
