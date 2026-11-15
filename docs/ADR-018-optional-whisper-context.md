# ADR-018: Optional Whisper Context Before Model Calls

## Status

Accepted and implemented.

## Context

Agent model calls can benefit from concise reminders about relevant tools and
nearby knowledge. Requiring the model to issue discovery calls first adds
latency and consumes workflow steps. Injecting full catalog or graph records,
however, increases prompt size and risks bypassing existing read and ACL
boundaries.

## Decision

Add an opt-in, deterministic Whisper context builder at the model-step boundary.
For each enabled model call, it uses that call's prompt as the retrieval query
and adds a compact projection under the model context key whisper. The same
projection and its source references are recorded in that workflow step's
output for inspection. It does not alter canonical graph state.

The profile may declare a WhisperConfig; explicit model-step configuration
takes precedence. Whisper is disabled by default. Defaults when enabled are:

- lexical/BM25 catalog search, with at most three tool hints;
- one-hop knowledge retrieval, with at most three search seeds;
- at most ten unique knowledge items total;
- summaries no longer than 320 characters each;
- a 6,000-character serialized-result ceiling.

Depth zero is direct RAG. Depth one keeps search results first, then appends
neighbors in backend order, deduplicating by entity ID without reranking.
Configuration can lower limits but cannot exceed the stated ceilings.

Tool hints contain only ID, name, kind, and a short summary. Catalog ACL/scope
filtering remains in the existing CatalogStore/AgentReadTools path, followed
by filtering each descriptor's required capabilities against the runtime's
trusted effective-capability set. Only descriptors with a non-empty required
capability set fully present in that set are suggested. Whisper uses no model
reranker and never invokes tools.

Knowledge reads use the injected AgentReadTools and a fixed or per-call
ReadScope supplied by trusted composition, or scope values from the runtime's
trusted authority carrier. Mutable workflow state is not used to construct
identity or scope. The AgentReadTools boundary applies ACL, namespace, tenant,
project, and visibility checks before projection.

## Data Flow

```text
prompt + host configuration
        |
        v
optional Whisper builder
   |                |
   |                +--> AgentReadTools + current ReadScope
   |                     |-- ACL-scoped lexical/BM25 catalog search
   |                     `-- scoped knowledge search/one-hop expansion
   |                |
   `----------------+--> compact, bounded, source-referenced result
                              |                    |
                              v                    v
                    model context[whisper]   workflow step output
                              |
                              v
                     existing model provider
```

## Security And Authority Boundaries

- Whisper is advisory context, not authorization, execution, or graph truth.
- It cannot grant capabilities, invoke a tool, mutate graph state, or promote
  retrieved content to verified truth.
- Existing runtime authorization remains authoritative for every later action.
- Tool capability metadata filters recommendations; it is not an enforcement
  boundary. Tool execution still requires the normal host authorization path.
- The model sees only compact, ACL-visible records and source references, not
  ACL policy internals or raw exception details.
- Caller-provided state cannot widen scope or effective capabilities.
- Retrieval failures fail open for the model call: Whisper returns no records
  and a bounded status such as retrieval_unavailable or
  read_scope_unavailable. The raw exception stays host-side.

## Tradeoffs

The hook adds synchronous read latency before each model call. The opt-in
default, small result limits, one-hop ceiling, and serialized-character budget
bound that cost. Retrieval can still surface irrelevant or stale information;
the model must treat it as evidence to inspect rather than as a command or
authoritative assertion. Separate provider calls may repeat retrieval across
workflow steps; caching is intentionally deferred because cache keys must
include the full authorized scope and current retrieval state.

## Consequences

- Profile and per-call configuration are serializable and validated.
- Existing model calls are unchanged unless Whisper is enabled.
- The selected result is visible in execution state for debugging and review.
- No new graph entity, persistence subsystem, model reranker, or authorization
  mechanism is introduced.

## Verification

Tests cover profile/call configuration precedence, per-model-call invocation,
ACL/scope/capability filtering, no tool execution, direct and one-hop retrieval,
ordering and deduplication, hard limits, compact source-referenced projection,
payload truncation, failure-open behavior, and unchanged ACL/budget outputs.
