# ADR-016: Agent Capability and Documentation Surfaces

**Status:** Proposed  
**Date:** 2026-07-25  
**Owner:** Maintainers

## Decision

Kogwistar and its immediate consumers will expose capabilities under one
governing invariant:

> **One versioned capability has one domain owner and one transport-neutral,
> typed behavioral contract. Skills, documentation, MCP, HTTP, CLI, Python, and
> deterministic workflow adapters may change its affordance, but must not fork its
> semantics, authority, security, or lifecycle.**

Corollaries:

1. Each authoritative state transition has exactly one writer.
2. Dependencies point from product intent to task services to bounded primitives
   to authoritative core; never upward.
3. A high-intelligence harness may select and compose capabilities. A
   low-intelligence harness receives a deterministic task contract. Neither owns
   domain semantics.
4. Skills contain routing and operating guidance, not hidden business logic.
5. Documentation is progressively disclosed, addressable data linked to the same
   capability ID; it is not an unversioned prompt appendix.
6. Tests are executable contract evidence and example use cases. Documentation
   claims without linked tests are informative, not compatibility guarantees.
7. Every side-effecting capability is scoped, capability-gated, auditable, and
   explicit about idempotency, approval, budget, timeout, and recovery.
8. This ADR does not initiate, continue, or broaden Rust migration. Rust work is
   limited to fixing a real defect in an already-implemented core capability, under
   ADR-015's existing parity gates; it may not change this exposure contract.

Short form:

> **Same contract, different affordance; one authority, no semantic fork.**

## Context

The ecosystem contains:

- `kogwistar`: authoritative graph, evidence, runtime, conversation, and
  compatibility surfaces;
- `kogwistar-llm-wiki`: research and knowledge-work application;
- `kogwistar-sqllm-wiki`: structured-data and SQL knowledge application;
- `kogwistar-chat`: conversational application;
- `kogwistar-obsidian-sink`: vault projection and publication;
- `kg-doc-parser`: document parsing and grounded extraction.

These products must serve both smart coding harnesses such as Codex, Claude Code,
and OpenCode, and less autonomous deterministic workflows or junior/vibecode
developers. They also need compact
self-service access to design intent, tutorials, configuration, examples, tests,
and troubleshooting.

Exposing every internal function as a tool would leak implementation structure.
Exposing only large autonomous skills would make deterministic automation,
security review, testing, and recovery unreliable. Duplicating behavior for each
harness would create semantic drift.

## Implementation Scope

This decision governs public capability contracts across the listed repositories,
but implementation work under it begins in `kogwistar` only. It neither authorizes
nor requires Rust migration in any consumer repository. Consumer repositories
remain independent product owners; they adopt a published contract through their
own changes and release decisions.

The current Rust port is not a continuing workstream of this ADR. A Rust change is
permitted only to repair a real bug in an implemented core feature. It must stay
within core, preserve public contracts, and pass applicable Cargo and Python parity
tests. New feature migration and migration work in consumers are out of scope.

## Capability Model

Every public capability is classified on two independent axes:

- **software abstraction**: primitive or task/domain;
- **harness intelligence**: compositional or deterministic.

This yields four affordances over the same underlying contracts:

| Software level | Smart harness | Deterministic harness |
|---|---|---|
| Domain/task | intent-oriented skill selecting task contracts | one typed task command |
| Primitive | bounded, discoverable tools for composition | stable SDK/HTTP operation |

The classification changes discoverability and guidance, not behavior.

Each capability descriptor must declare:

```yaml
id: wiki.answer
version: v1
owner: kogwistar-llm-wiki
abstraction: task
effect: read
input_schema: ...
output_schema: ...
requires: [core.retrieve-context, core.resolve-evidence]
capabilities: [knowledge.read]
idempotency: optional
budget: supported
timeout: supported
approval: none
docs: doc://wiki.answer
contract_tests: [wiki.answer.grounded]
```

Capability IDs are stable semantic identifiers. Transport names and Python import
paths may map to them but are not the identifiers.

## Layer and Dependency Rules

The normative layers are:

1. **Authoritative core** — deterministic state, transactions, events, replay,
   evidence identity, and bounded primitives.
2. **Task services** — product-owned operations such as answer, ingest, parse,
   publish, sync, and run workflow.
3. **Adapters** — Python, HTTP, MCP, CLI, syscall, and deterministic workflow bindings.
4. **Skills** — tool selection, composition, stop conditions, and recovery
   guidance for a stated harness profile.
5. **Documentation index** — targeted lookup over intent, design, tutorial,
   configuration, examples, tests, and troubleshooting.

Allowed dependencies point downward. Sibling products integrate through public
capability contracts or versioned events, not each other's internals.

Core must not import product policy. Parsers must not own graph, conversation, or
publication policy. Sinks must not decide knowledge truth. Skills and
documentation must not bypass adapters to mutate storage.

## Product Ownership

| Domain | Owner | Representative tasks |
|---|---|---|
| Graph, evidence, runtime | `kogwistar` | retrieve context, resolve evidence, run/inspect/resume |
| Research wiki | `kogwistar-llm-wiki` | research, answer, refresh knowledge |
| SQL wiki | `kogwistar-sqllm-wiki` | inspect schema, execute bounded read, ground result |
| Conversation | `kogwistar-chat` | send turn, recall, answer with evidence |
| Document parsing | `kg-doc-parser` | parse, validate, emit grounded extraction |
| Vault projection | `kogwistar-obsidian-sink` | preview, publish, sync, rebuild |

Composite product tasks may orchestrate dependencies but do not assume their
authority. For example, `wiki.publish-research` owns orchestration while parsing,
graph truth, and vault writes remain owned by their respective capabilities.

## Harness Profiles

Smart-harness skills may:

- discover capabilities;
- look up concise documentation;
- choose among bounded tools;
- compose task services;
- inspect run state and request approval.

They must not infer private storage contracts or synthesize unsupported mutations.

Deterministic harness bindings must provide fixed schemas, bounded retries,
machine-readable errors, explicit idempotency, and observable run state. They must
not require an LLM to choose the safe execution sequence.

Profile differences belong in skill metadata and adapter policy. They must not
create `smart` and `dumb` implementations of domain behavior.

## Documentation Contract

Documentation lookup is a first-class read-only capability family:

```text
doc.index
doc.lookup
doc.example
doc.test-case
doc.config
doc.troubleshoot
doc.related
```

Lookup follows progressive disclosure:

1. `doc.index` returns capability IDs, one-line summaries, confidence, and
   references.
2. `doc.lookup` returns only requested sections.
3. Example, test, design, or configuration detail is expanded only on demand.

Every documented public capability should expose:

- intent and non-goals;
- when to use and when not to use;
- minimal valid example;
- input, output, effects, and permission requirements;
- configuration keys, types, defaults, scope, and restart/security impact;
- failure modes and recovery;
- related capability IDs;
- linked contract tests and source references.

Documentation and tests use stable IDs. File paths are locators, not identity.
Generated indexes must be reproducible from repository-owned metadata and must not
become an independent source of truth.

Skills should remain small. They query documentation by capability and section
instead of embedding whole design documents or test suites in every prompt.

## Operation Envelope

All task and side-effect adapters converge on a common conceptual envelope:

```text
capability_id, capability_version, request_id, idempotency_key,
subject, tenant/namespace, capabilities, input, budget, timeout,
approval_context -> status, result/error, evidence_refs, run_id, audit_ref
```

Fields may be omitted only where semantically inapplicable. Structured status and
error codes are stable; prose messages are not control flow.

Long-running work returns a `run_id` and supports inspect, event consumption,
cancel, and—where declared—resume. Evidence-producing work returns stable evidence
references, not only rendered text.

## Compatibility and Verification

A new or changed exposed capability is mergeable only when:

1. owner, version, schemas, effects, and capability requirements are declared;
2. adapters map to one task/primitive implementation;
3. contract tests cover success, bounded failure, authorization, and idempotency
   where applicable;
4. at least one minimal example is executable or derived from a passing test;
5. documentation index entries resolve and contain no broken capability IDs;
6. affected consumer compatibility suites pass;
7. a permitted native-core bug fix passes Cargo and Python parity gates; consumer
   suites run only when its published compatibility surface is affected.

CI should reject:

- duplicate owners for one capability/version;
- upward or undeclared cross-product dependencies;
- public tools without capability metadata;
- documentation pointing to absent tests or capabilities;
- different output invariants across MCP, HTTP, SDK, or workflow adapters;
- side effects lacking explicit authorization and audit behavior.

## Evolution

Backward-compatible additions retain the capability version. Breaking schema,
authority, lifecycle, security, or behavioral changes require a new version and a
declared migration period.

Adapters and skills may evolve faster than capability versions when behavior is
unchanged. Deprecated adapters must identify the replacement capability, not only
a replacement transport path.

## Consequences

Benefits:

- one semantic surface serves smart agents, fixed workflows, and developers;
- documentation lookup reduces prompt size and unsupported inference;
- product ownership and Rust migration boundaries remain explicit;
- tests become both compatibility evidence and trustworthy examples;
- adapters can evolve without duplicating business logic.

Costs:

- capability metadata and documentation indexes require validation tooling;
- existing MCP, REST, syscall, and internal imports must be cataloged and mapped;
- some broad tools and coarse capabilities must be decomposed or marked internal;
- cross-repository release gates become mandatory for exposed behavior.

## Initial Implementation Order

1. Define descriptor schema and stable capability IDs.
2. Inventory existing core and consumer surfaces without changing behavior.
3. Map current Python, REST, MCP, and syscall entries to descriptors.
4. Add documentation index and targeted read-only lookup.
5. Publish smart, deterministic-task, primitive, and operator skill profiles.
6. Add descriptor, documentation-link, adapter-parity, and consumer contract gates.

Repository locations and consumer roots are configuration. No machine-specific
absolute path is part of the architecture.
