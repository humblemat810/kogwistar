# ADR-020: Thin Agent Harness by Composition

**Status:** Proposed
**Date:** 2026-09-25
**Owner:** Maintainers

## Context

Kogwistar already provides the execution and persistence primitives needed for
agentic behavior:

- persisted workflow nodes, edges, predicates, cycles, fanout, and joins;
- `WorkflowRuntime`, workflow state, step execution, checkpoints, and resume;
- nested invoke-and-await through `WorkflowInvocationRequest`;
- runtime budgets, tracing, authorization, and audit surfaces;
- conversation graphs and cross-conversation memory retrieval;
- evidence-backed knowledge retrieval;
- wisdom proposals, evaluation status, lessons, and workflow revision lineage.

The project needs a convenient agent-facing composition layer with plan mode,
goal mode, subagent delegation, context compression, skills, plugins, model
selection, and optional external memory providers. Adding a second agent loop,
scheduler, checkpoint model, memory authority, or multi-agent runtime would
duplicate existing Kogwistar semantics.

External systems provide useful design lessons without becoming Kogwistar
authority:

- Pi demonstrates a small harness, extension hooks, session branching, and
  lossy context compaction while retaining complete history.
- Hermes distinguishes compact durable facts from reusable procedures and
  supports reviewed learning.
- DeepSeek Harness demonstrates typed plugin dependencies, deterministic
  lifecycle disposal, and profile composition.
- LangChain Deep Agents demonstrates middleware, isolated subagent context, and
  bounded delegation over an existing graph runtime.
- Claude Code demonstrates declarative subagent profiles with explicit model,
  tools, permissions, skills, effort, isolation, and turn limits.
- A2A demonstrates capability discovery and durable remote-task interaction at
  an agent-to-agent transport boundary.
- CrewAI distinguishes deterministic flows from local pockets of autonomous
  collaboration.

## Decision

Kogwistar will implement a thin agent harness as composition over existing
primitives.

```text
AgentProfile
  + workflow design or workflow ID
  + resolver and tool bindings
  + context policy
  + model provider
  + bounded lifecycle hooks
  + capability and budget envelope
        |
        v
ordinary WorkflowRuntime.run(...)
```

There is no new Agent runtime, Goal runtime, Plan runtime, Crew runtime,
subagent scheduler, token type, checkpoint type, or memory store.

Plan mode and goal mode are ordinary workflow designs. Subagents are ordinary
nested workflow invocations. Agent-specific names and metadata communicate
policy and intent to people, validators, prompts, and user interfaces; they do
not alter runtime scheduling semantics.

Implementation work is tracked in
`PLAN-agent-harness-implementation.md`.

Goal labels group numbered phases only:

```text
Goal A -> Phase 0-2
Goal B -> Phase 3-5
Goal C -> Phase 6-8
```

The numbered phase checklist in the implementation plan remains authoritative;
Goal labels do not create a second completion checklist.

### Goal A Phase 0-2 Core Slice

The first implementation slice is deliberately offline and provider-neutral:

- `kogwistar.agent.AgentProfile` declares requested capabilities without
  granting authority.
- `AgentHarness` and `AsyncAgentHarness` delegate to the existing runtime;
  neither owns a loop, checkpoint, scheduler, or memory store.
- `ProviderRegistry` rejects provider-qualified identity collisions and disposes
  only registrations it owns.
- `CatalogStore` applies ACL/scope filtering before deterministic exact, alias,
  prefix, partial, or lexical ranking. Revisions are fingerprint-gated; unload
  removes active provider projections while history remains inspectable.
- `SkillGraphArtifact` is bounded JSON-compatible graph data. Core parsing is
  deterministic and handles frontmatter, explicit capability/MCP references,
  package-relative scripts, and structured command templates.
- Effectful or unbound parsed steps remain non-invocable. Shell wrappers,
  traversal paths, and implicit execution are rejected before any provider call.

This slice does not yet claim REST/MCP read-only tool parity, hook lifecycle,
semantic ingestion, LLM-Wiki integration, or graph projection persistence.
Those remain plan items and do not alter the authority model.

The subsequent read slice now provides transport-neutral `ReadScope`, bounded
cursor pages, descriptor-first catalog/skill/MCP/capability reads, raw-versus-
graph skill paths, and adapters over existing execution/memory/knowledge/
wisdom service callables. REST and MCP transport registration remains a separate
compatibility-gated step; discovery must not become invocation or approval.

### Goal B Phase 3-4 implementation slice

The first workflow-composition slice is now implemented without adding an agent
runtime. `kogwistar.agent.workflows` builds ordinary
`WorkflowDesignArtifact` graphs for normal, plan, and goal profiles. The mode
and contract metadata are advisory: removing them leaves node operations,
edges, predicates, cycles, and terminal reachability unchanged.

`kogwistar.agent.bindings` supplies deterministic fake-model and fake-tool
bindings through `MappingStepResolver` and `RunSuccess`/`RunFailure`. Catalog
lookup calls the Goal B descriptor-first read facade. Static and dynamic action
helpers return the existing `WorkflowInvocationRequest`; dynamic identity is
validated before the result reaches `WorkflowRuntime`. Phase 5 delegation is
not included in this slice.

`kogwistar.agent.limits.AgentBudgetPolicy` seeds the existing
`StateBackedBudgetLedger`. Remaining steps, model calls, tokens, time, and cost
are bounded context hints only; call, token, and cost checks remain ledger
checks. A prompt or model payload cannot grant capability or raise a ceiling.
The runtime still owns checkpoints, run/trace identity, nested invocation,
ACL, provenance, and recovery.

Focused tests execute all three ordinary graphs with fake payloads, exercise a
two-iteration goal, stop a cyclic goal at a hard step ceiling, validate static
and dynamic action identity, and verify fail-closed tool capability checks.
No real LLM, network provider, or remote telemetry is required.

This slice now includes the minimal queued/steering control-point contract:
typed policy metadata, bounded target-run claims, checkpoint-before-ack
delivery, terminalization, ACL filtering, and ordered reconnectable reads. It
does not claim transport-level disconnect/cancellation behavior or automatic
run-specific inbox provisioning; those remain runtime/adapter follow-up work.
Context compression, subagent profile delegation, wisdom hooks, A2A/CrewAI
adapters, and LLM-Wiki integration remain deferred under their numbered
phases; Goal C covers the later Phase 6-8 compression, wisdom/skill
projection, and plugin work.

## Existing Primitive Map

| Agent concept | Existing Kogwistar primitive |
| --- | --- |
| Agent loop | `WorkflowDesignArtifact` executed by `WorkflowRuntime` |
| Observe, plan, decide, act, check | Ordinary workflow nodes and resolvers |
| Routing | Predicate workflow edges |
| Iteration | Cyclic workflow edges |
| Subagent | `WorkflowInvocationRequest` and child `WorkflowRun` |
| State | Ordinary workflow state |
| Durability | Workflow checkpoints and resume |
| Human approval | Suspension and resume |
| Limits | Runtime budget ledger and pre-dispatch checks |
| Execution history | Workflow runs, steps, events, lineage, and evidence |
| Conversation memory | Conversation graph and memory retrieval |
| Domain knowledge | Knowledge graph and evidence references |
| Procedural learning | Wisdom proposals, evaluations, and lessons |
| Correlation | `TraceContext`, distinct from domain identifiers |
| Authorization | Capability kernel plus engine ACL enforcement |

## Agent Profile

`AgentProfile` is a typed composition contract, not a runtime primitive and not
an authorization grant. A minimal profile describes:

```text
agent_id
workflow_id or workflow_design
model_profile
requested_tool_capabilities
context_policy
compression_policy
budget_policy
skill_providers
plugin configuration
hook configuration
```

The requested capabilities are a ceiling, not a grant. Effective capabilities
are the intersection of authenticated caller authority, delegation constraints,
profile requests, revocation, and engine ACL. Profile text, role names,
backstory, tool lists, and model output cannot grant authority.

Profiles may select a normal, plan, or goal workflow. Runtime behavior remains
fully determined by the selected workflow graph and existing runtime contracts.

## Plan and Goal Modes

Plan mode is a reusable ordinary workflow pattern, for example:

```text
Observe -> Gather -> Plan -> Approve -> Execute -> Check -> Done
```

Goal mode remains the ordinary cyclic workflow pattern decided by ADR-019:

```text
Observe -> Decide -> Act -> Check
              ^                |
              +-- incomplete --+
```

Mode metadata is advisory. It may support discovery, prompt assembly, UI, and
linting, but runtime correctness must not depend on a magic `mode`, node name,
or metadata value.

## Interactive Control, Steering, and Queues

Responses-style interaction is represented by existing conversation, run,
event, lane-message, cancellation, checkpoint, and resume primitives. A model
response is not a new execution authority and must not be conflated with a
workflow run.

```text
conversation turn/input
  -> root WorkflowRun (queued/running/terminal)
  -> ordered run events
  -> zero or more model calls and tool calls
  -> response/result nodes and evidence references
```

An input received while a run is active declares one of three policies:

```text
steer
  apply to the active run at its next safe interruption point

queue
  persist now, then start or continue work after the current lane item finishes

cancel_and_replace
  request cooperative cancellation, then start replacement work after the
  cancellation boundary is durable
```

The authoritative user input is persisted in the conversation graph. A durable
lane message carries its immutable input-node reference and fingerprint, plus
delivery state and correlation to the target run, turn, lane, and input
sequence. It is not a second editable copy of the input. Same-lane input is
FIFO. Different lanes may execute concurrently only where the selected runtime
and scheduler already support it.

Steering is forward-only. It does not rewrite streamed output, undo persisted
state, or reverse a tool or external side effect that has already started.

Steering intake is explicit workflow topology. The harness and runtime must not
inject steering at arbitrary implicit code points. A workflow that accepts
steering places ordinary control-point nodes where new input may be observed:

```text
... -> ControlPoint -> AssembleContext -> Decide/ModelCall -> ...
             |
             +-> no control input: continue unchanged
             +-> steer input: apply bounded state/context patch, then continue
             +-> queued next-turn input: leave queued for the next lane item
```

A control point is an ordinary workflow node bound to a generic authorized
lane-message claim resolver/capability. It selects and claims at most a bounded
number of messages whose target run/lane, accepted message kind, and capability
scope already match the control point, then returns ordinary workflow state
patches. A run-specific control inbox is preferred over claiming unrelated
messages and rejecting them afterward. Its metadata may declare accepted
message kinds, maximum claims, and queue policy. No new node class, token, or
scheduler is introduced.

Lane-message delivery is at least once. Claim and workflow checkpoint writes
are not assumed to share one transaction. For strict FIFO inboxes, the
checkpoint records a per-inbox applied sequence high-water mark before
acknowledgement. A bounded `message_id` set is needed only where a declared
policy permits ordering holes. Redelivery skips already-applied input, and
acknowledgement occurs only after the corresponding workflow state is durable.
This is idempotent application, not an exactly-once side-effect claim.

If a target run becomes terminal before a steering message reaches a control
point, the message enters an existing terminal status such as `cancelled` or
`failed`, with an explicit `target_terminal` reason. It is converted to queued
next-turn input only when the sender selected that fallback policy. Steering is
never silently retargeted to another run.

Workflow authors choose control-point placement. Common templates may place one
after a tool result, before a new model call, before an expensive action, or at
an iteration boundary. A long-running resolver is not preemptible merely
because the graph contains a later control point; finer cooperative control is
owned by that resolver.

Hard cancellation, budget exhaustion, and security denial remain runtime
guards checked at dispatch boundaries. They are not graph steering and cannot
be disabled by omitting a control-point node.

Client disconnection does not imply cancellation for background work. Ordered
run events use durable sequence numbers so a client can reconnect after its
last cursor. Cancellation remains idempotent and cooperative. Input-required
interaction uses existing suspension, checkpoint, and resume contracts rather
than an ad hoc chat callback.

An optional Responses-style HTTP or WebSocket adapter may map external
response and stream identifiers to Kogwistar run, lane, turn, and event-cursor
identities. Those external identifiers remain transport identifiers and never
replace Kogwistar domain or trace identity. Exact compatibility is claimed only
by an adapter with versioned schema and event-conformance tests.

## Subagents

A subagent is a child workflow run with an explicit invocation contract. The
parent may select a different model or provider for a bounded task, including:

- vision inspection when the parent model has no vision support;
- local-model processing;
- isolated search or repository exploration;
- MCP-backed specialist work;
- low-cost lookup that should not contaminate parent context.

Delegation must persist parent run, parent step, child run, invocation identity,
model profile, effective capabilities, budget allocation, and result/evidence
references through generic workflow lineage.

Child capabilities must be an equal or narrower set than the delegated parent
envelope unless separately approved. Child context is isolated by default and
returns a bounded structured result rather than its complete working context.

## Lifecycle Hooks

Kogwistar will add one small typed hook contract for composition-time and
model/tool lifecycle extension. This is an extension seam, not a second event
bus or scheduler.

Initial hook points may include:

```text
context.before_assemble
context.after_assemble
model.before_call
model.after_call
tool.before_call
tool.after_call
run.after_complete
```

Each hook declaration must specify deterministic ordering, timeout, failure
mode, effects, required capabilities, and cleanup behavior. Security and
authority checks remain non-pluggable and fail closed. Observational hooks may
be best effort.

Hooks may transform bounded context, request compression, append mode and
remaining-budget guidance, or submit a wisdom-distillation request. Hooks must
not mutate authoritative graph, workflow, memory, or wisdom state directly.
Effectful work is performed through an authorized capability or ordinary
workflow.

Hooks also must not choose workflow branches, claim steering messages, dispatch
child runs, replace checkpoint behavior, or hide authoritative side effects.
Those actions remain explicit workflow topology or protected runtime behavior.

## Context and Compression

Context assembly may combine:

- current conversation path;
- selected same-conversation summaries;
- explicitly enabled cross-conversation memory;
- knowledge evidence references;
- approved wisdom or projected skills;
- current workflow state and bounded execution evidence;
- current mode and estimated remaining step/token/cost budget.

Cross-conversation memory is opt-in and ACL-filtered. A profile cannot bypass
engine visibility rules.

Compression is a derived conversation projection. It creates a summary node
with provenance to the covered messages or graph region. It never deletes or
replaces authoritative conversation history.

Compression aggressiveness is independent from model reasoning effort and
memory scope. Initial policy names may be `low`, `medium`, `high`, and `ultra`,
but their thresholds and retained-context contracts must be explicit and
versioned.

The runtime enforces hard budgets before dispatch. Prompt text that reports
remaining steps or tokens is guidance only and is never the enforcement
mechanism.

## Memory, Knowledge, Wisdom, and Skills

These planes remain distinct:

```text
execution history  = what the system did
conversation memory = what a user or conversation said or established
knowledge graph    = evidence-backed domain claims and relationships
wisdom graph       = evaluated guidance on how to perform better
skill              = an on-demand projection of approved wisdom or authored guidance
```

The agent harness will expose bounded read adapters over existing stores rather
than introduce a new memory database:

```text
workflow.run_status
workflow.run_events
workflow.run_steps
workflow.run_lineage
workflow.run_evidence
conversation.memory_search
conversation.memory_get
knowledge.search
knowledge.expand
knowledge.get
wisdom.search
wisdom.get_proposal
wisdom.get_lesson
```

Wisdom search defaults to approved lessons. Pending, rejected, or deprecated
artifacts require explicit status selection and must not silently become
normative prompt instructions.

Hermes-style learning maps to a post-run wisdom-distillation request:

```text
execution and feedback
  -> distillation request
  -> ordinary wisdom workflow
  -> proposal
  -> evaluation
  -> approved lesson
  -> optional skill or workflow projection
```

The agent never writes an authoritative skill merely because a model claims to
have learned something.

### Progressive skill disclosure

Skills use progressive disclosure rather than loading every skill body into
every prompt. The bootstrap context contains only bounded descriptors:

```text
skill_id
version
one-line description
tags and scope
required capabilities
source and approval status
```

Skill and MCP discovery share a small read-only catalog surface:

```text
catalog.search(query, kinds, scope, group_id=None, limit=...)
catalog.browse(group_id=None, kinds=None, scope=None, limit=...)
catalog.get(entry_id)

skill.get(skill_id, section=None)
skill.resource_get(skill_id, resource_id)
mcp.describe(entry_id)
capability.describe(capability_id)
```

Search, hierarchy browsing, descriptor retrieval, full skill loading, linked
resource retrieval, and MCP schema activation have different token and
authorization costs. One `skill.search` call alone is therefore insufficient.
Full skill text is loaded only after selection. Examples, scripts, large
references, and MCP tool schemas are fetched separately and only when needed.

Catalog entries form a graph-native hierarchy without making filesystem paths
their identity. One entry may belong to several groups. A tree shown by a UI is
a browse projection, not the authoritative catalog shape:

```text
entry_id
provider_id
provider_local_id
group_ids or membership edges
kind                 # skill, mcp_tool, mcp_resource, capability, documentation
name and summary
tags and aliases
project/tenant scope
required capabilities
version and approval status
```

Exact IDs and aliases, hierarchical browsing, prefix/substring matching, and a
lexical full-text rank are mandatory. Semantic retrieval is optional and may
improve ranking only when an embedding projection is ready. Search must remain
useful when no embedding provider is configured, embedding is pending, or the
semantic index is unavailable.

The default ranking policy is conceptually:

```text
exact ID/name/alias
  > scoped hierarchy and prefix match
  > lexical BM25/full-text match
  > partial-text match
  > optional semantic blend when ready
```

An implementation may use SQLite FTS5 or another backend-specific lexical
index, but the catalog search contract is backend-neutral. Catalog descriptors
and their lexical index are not ADR-018 Stage 1. They are a rebuildable catalog
read model and lexical search projection. An optional semantic projection may
enrich ranking without replacing or hiding lexical discovery. This avoids
conflicting with ADR-018's exclusive Stage-1/Stage-2 handoff contract.

### Skill-MCP-capability graph

An installed skill may remain an opaque provider-owned package, or an optional
ingester may parse it into a Kogwistar-native skill-MCP-capability hypergraph.
The original package remains the source for its authored content and version;
the parsed graph is a disposable, revision-gated discovery projection.

Ingestion is additive. It must not remove or weaken the provider-native way to
load or use the skill. An authorized agent may choose either path:

```text
provider-native path
  -> discover raw skill descriptor
  -> load/use skill through its provider capability

graph-native path
  -> search parsed skill-MCP-capability graph
  -> retrieve bounded relevant steps, dependencies, and MCP/capability refs
  -> feed them to an ordinary agent workflow
  -> invoke each selected capability through normal authorization
```

Both paths identify the same provider-qualified skill version and preserve its
source fingerprint. The graph-native path permits finer progressive disclosure,
dependency traversal, and step-level provenance; it is not required merely to
use an installed skill.

Useful projected nodes include:

```text
skill descriptor
skill section or procedural step
capability descriptor
MCP tool, resource, or prompt descriptor
linked documentation or evidence reference
```

Useful ordinary edges or hyperedges include:

```text
skill --contains/orders--> procedural step
step --requires--> capability
step --may invoke--> MCP tool
step --reads--> MCP resource or documentation
entry --member of--> one or more catalog groups
projected entry --derived from--> package/server version and fingerprint
```

Progressive disclosure is then bounded graph retrieval:

```text
lexical/optional semantic candidate search
  -> bounded group and dependency traversal
  -> descriptor or selected section
  -> selected validated MCP schema/capability contract
  -> separately authorized invocation
```

Parsing a skill does not turn prose into executable authority. Graph nodes do
not grant capabilities, and inferred step order does not become a workflow
unless an explicit validated compiler produces a `WorkflowDesignArtifact`.
MCP server discovery remains the source for its current remote catalog; any
local graph snapshot is versioned, attributable, and rebuildable.

An agent may nevertheless *run with* the digested form: its ordinary workflow
looks up the relevant skill subgraph, records selected step and source
references, and invokes the referenced typed capabilities one by one. The
workflow, resolver contracts, budget guards, and ACL remain execution authority;
the skill graph supplies procedure and discovery context only.

### Skill ingestion providers and authority

Kogwistar adopts a hybrid ingestion design:

```text
mandatory core path
  deterministic structural parsing and validation
  raw/provider-native skill use always remains available

optional enhanced path
  kogwistar-llm-wiki or another provider performs semantic digestion,
  cross-link candidate generation, and maintenance
```

The enhanced provider is optional. Core importability, installed-skill use,
exact descriptor retrieval, deterministic structural parsing, and execution do
not depend on it. Kogwistar-LLM-Wiki may improve understanding but never becomes
skill source, approval authority, capability authority, or executor.

| Concern | Kogwistar core-only path | Optional LLM-Wiki path |
| --- | --- | --- |
| Dependency | local and mandatory | optional plugin/service |
| Structural extraction | manifests, frontmatter, schemas, files, explicit refs | may reuse the same normalized input |
| Semantic decomposition | deliberately bounded | richer step, prerequisite, fallback, and cross-link candidates |
| Latency/cost | low and deterministic | asynchronous and potentially model-backed |
| Failure behavior | raw skill remains usable | failure cannot hide raw skill or replace current valid projection |
| Authority | validates, materializes, authorizes, and executes | returns attributed candidate artifacts only |
| Maintenance | source fingerprint and explicit rebuild | optional reparse, deduplication, stale-link detection, and candidate repair |

The current generic Kogwistar document ingester builds a document/chunk/summary
hierarchy. It is not silently redefined as a procedural skill parser. Core adds
a small skill-specific deterministic parser over the shared graph primitives;
semantic enrichment remains a provider contract.

Providers exchange one typed, JSON-compatible `SkillGraphArtifact`. Its exact
model is implementation work, but the minimum contract is:

```text
provider_id
provider_local_id
skill_version
source_fingerprint
parser_id and parser_version
nodes: list
edges: list
warnings: list
unsupported_constructs: list
```

Every projected node or edge carries source span/reference, parser provenance,
and confidence where inference occurred. Lists, not tuple-only schemas, are
used at the provider boundary. Core validates IDs, graph bounds, source
fingerprint, allowed node/edge kinds, namespace scope, and capability
references before materialization.

For an LLM-Wiki-backed implementation, raw skill source may be registered as a
versioned skill-package source. The digested graph belongs in a dedicated
projection lane, conceptually:

```text
ws:<workspace>:g:projection:lane:skills
```

with metadata such as:

```text
projection_kind = skill_mcp_capability_graph
provider_id
skill_id
skill_version
source_fingerprint
parser_profile
projection_schema_version
project_id / tenant_id
```

This is a derived serving projection, not knowledge, wisdom, conversation
memory, or workflow truth. A caller does not supply an arbitrary target
namespace; a typed `skill_projection` purpose is mapped to the allowed
workspace/project lane by trusted adapter configuration.

Skill steps may describe several effect classes:

```text
instruction           # context/guideline only
capability_call       # typed Kogwistar capability
mcp_call              # selected validated MCP schema
script_call           # package-relative script reference
command_template      # executable plus structured argv template
nested_workflow       # ordinary WorkflowInvocationRequest
check                 # validation or completion criterion
```

Parsing does not make these steps executable. Instruction steps have no direct
effect. MCP calls require server identity, selected-schema validation, and an
authorized invocation adapter. Script calls require a package-relative path,
content fingerprint, argument schema, working-directory policy, and sandbox.
Command templates store executable plus argv as data; arbitrary shell strings,
implicit `shell=True`, inherited secrets, unrestricted environment, filesystem,
network, or process access are not accepted. Each effect declares required
capabilities, timeout, output bound, and risk metadata. An unbound or invalid
step remains searchable but non-invocable.

### Cross-links, memory, and learning

Explicit links from manifests, MCP schemas, capability IDs, and package files
may be materialized deterministically. Model-inferred prerequisites,
alternatives, similarities, and fallbacks are candidate edges carrying source,
confidence, parser version, and lifecycle status. They do not silently become
workflow routes or capability grants.

Execution and learning preserve plane boundaries:

```text
selected skill revision/step
  -> workflow step and capability/MCP invocation
  -> result, error, performance, and user feedback evidence
  -> conversation/project memory observation
  -> wisdom distillation proposal
  -> evaluation and approval/rejection
  -> optional new skill revision and rebuilt graph projection
```

Memory may link a run or feedback item to the skill revision and step used, but
does not edit the skill. LLM-Wiki memory lifecycle status is evidence-serving
state, not Kogwistar wisdom approval. Only the wisdom proposal/evaluation
lifecycle may recommend a generalized procedural revision. Approved learning
creates a new attributable version; it does not mutate prior source or graph
history in place.

Projection replacement is revision gated. A new parsed graph becomes current
only after validation against the current source fingerprint. Provider failure
leaves raw use and the prior valid projection available. A stale late result
cannot replace a newer revision. Uninstall retracts provider-owned catalog and
derived graph records but retains execution, feedback, and wisdom evidence that
refers to their immutable historical identities.

MCP has no authority-specific Kogwistar skill semantics. An MCP adapter may
provide catalog descriptors for remote tools, resources, and prompts. Once the
agent selects an entry, the harness may include only the selected validated MCP
schemas in the next model call. The complete remote MCP catalog need not occupy
every prompt. Skills similarly route the agent to already published typed
capabilities. A broad unvalidated `skill.run` or arbitrary
`capability.invoke` escape hatch is not adopted.

Skill support and MCP support are provider plugins over this catalog contract:

```text
SkillProviderPlugin    -> descriptors, skill content, linked resources
McpDiscoveryPlugin     -> descriptors and selected schemas
McpInvocationAdapter   -> separately authorized bounded remote calls
DocProviderPlugin      -> descriptors and progressively disclosed documentation
```

Plugins own discovery and retrieval from their source. Core owns descriptor
validation, stable identity, ACL filtering, lexical fallback requirements,
bounded result shape, selected-schema validation, and invocation authorization.
Thus the harness remains minimal and extensible without making security or
catalog truth dependent on plugin load order.

Catalog logical identity is the stable pair `provider_id + provider_local_id`.
An immutable catalog revision additionally carries `version` and source
fingerprint. Registration rejects conflicting ownership rather than
overwriting by load order. Registration returns a disposer; unloading retracts
only entries owned by that provider and invalidates their derived
lexical/semantic records. Indexes are rebuildable from active provider
descriptors or persisted source snapshots. Credentials and invocation secrets
never enter catalog descriptors.

### Project terminology and private knowledge

A company glossary, internal abbreviations, product names, and domain terms are
primarily project-scoped knowledge, not skills. A project plugin may package
their source material and ingestion configuration, but ingestion places the
validated content in the project or tenant knowledge namespace with evidence,
version, and ACL metadata.

```text
project plugin
  knowledge/glossary sources -> project knowledge graph
  skills                     -> procedures for using that knowledge
  policies                   -> project context and authorization defaults
  tools                      -> optional bounded capability adapters
```

Term, abbreviation, alias, definition, source, validity, and relationship data
belong in the knowledge graph. A project context source retrieves only relevant
terms by exact match, alias expansion, or bounded semantic search. It does not
inject the entire company glossary into every prompt.

Use a skill when the content is procedural, such as how to prepare an internal
release, interpret a project-specific report, or apply naming conventions. The
skill may reference glossary entity IDs but must not duplicate the glossary as
its own authority. User-specific shorthand belongs in conversation memory;
generalized improvements to a procedure belong in wisdom before becoming an
approved skill projection.

## Plugins and Optional Providers

The harness may define small provider protocols for models, tools, skills,
memory, and compression. Provider replacement must not fork domain semantics.

Plugins may contribute:

- model adapters;
- authorized tool adapters;
- context sources;
- skill projections;
- compression policies;
- observational hooks;
- optional user interfaces.

Plugins may not replace transaction authority, event truth, workflow
scheduling, checkpoint semantics, ACL enforcement, budget enforcement, or
wisdom approval.

`kogwistar-llm-wiki` may be installed as an optional memory, knowledge, or skill
semantic-ingestion plugin through a public capability or MCP adapter. Core must
not import it, and absence of the plugin must not disable the agent harness or
provider-native skill use.

## A2A Boundary

A2A is an optional transport adapter, not an internal subagent mechanism.

An A2A Agent Card should be generated from the versioned capability catalog.
An external A2A task may map to one root workflow run and any number of child
runs. The adapter persists external context/task identifiers and their mapping
to local runs, status events, and artifact/evidence references through a
host-injected durable mapping store; its in-memory map is only a process-local
cache and test fallback.

Identifier meanings remain separate:

```text
A2A task ID != workflow run ID
A2A context ID != trace ID
A2A skill != Kogwistar wisdom skill
A2A artifact != canonical Kogwistar truth
```

Messages expose task negotiation and results, not internal chain-of-thought or
self-monologue. Authentication advertised by A2A does not replace Kogwistar
authorization or engine ACL.

Push callbacks, when enabled, are an outbound projection only. The adapter
accepts callback URLs only from trusted HTTPS host allowlists, signs bounded
payloads through an injected authenticator, and submits them to existing
durable delivery infrastructure with a deterministic `task_id + event_seq`
delivery ID, bounded retry metadata, and delivery audit. It performs no
synchronous network I/O and cannot make callback failure alter workflow truth.
An adapter without durable enqueue and signing capabilities rejects push
registration rather than silently falling back to unauthenticated delivery.

The adapter belongs under a server or interoperability package, not in the
agent execution core.

## CrewAI Interoperability

CrewAI concepts may be imported or projected only where semantics are clear:

```text
Flow                -> workflow design
Task                -> workflow node
task dependency     -> workflow edge
role agent          -> agent profile or resolver binding
delegation          -> nested workflow invocation
manager decision    -> ordinary Decide node
guardrail           -> validator, predicate, or protected hook
human input         -> suspension and resume
structured output   -> typed resolver result
```

Kogwistar will not run CrewAI as a second authority inside
`WorkflowRuntime`. A future importer may translate supported static CrewAI
Flow or Crew definitions to `WorkflowDesignArtifact`; unsupported dynamic
semantics must fail explicitly rather than silently degrade.

## Persistence and Provenance

Agent execution persists through existing workflow, conversation, knowledge,
and wisdom graphs. Agent-specific profile values are recorded as run metadata
only where required for replay, audit, or interpretation.

At minimum, an agent run should make these relations inspectable:

```text
caller/conversation -> root workflow run
root step -> child workflow run
workflow step -> tool invocation/result
retrieval -> selected memory/knowledge/wisdom references
compression -> covered conversation region
distillation request -> wisdom proposal/evaluation
```

No new mandatory Agent, Goal, Plan, Crew, or Skill canonical node type is
introduced by this ADR.

## Invariants

- `WorkflowRuntime` remains the sole local workflow executor.
- Agent modes are graph patterns and profiles, not runtime types.
- Names and metadata are descriptive; validators and typed contracts carry
  enforceable semantics.
- Authorization and budgets are enforced below prompts and plugins.
- Child execution receives bounded capabilities, context, and budget.
- Context compression never destroys authoritative history.
- Steering is forward-only and is applied only at explicit safe points.
- Steering points are ordinary nodes in workflow topology, never hidden harness
  injection points.
- Same-lane queued input is ordered and durably correlated to its target run.
- Memory, knowledge, wisdom, execution history, and skills remain distinct.
- Project terminology is project-scoped knowledge; skills contain procedures.
- Skill content is progressively disclosed after bounded discovery.
- Skill and MCP providers are plugins over one searchable graph-native catalog.
- Installed skills may remain opaque or gain a rebuildable
  skill-MCP-capability hypergraph projection with source provenance.
- Provider-native skill use and graph-guided skill use remain available in
  parallel; ingestion never becomes a prerequisite for using the raw skill.
- Core owns `SkillGraphArtifact` validation and materialization; semantic
  providers return candidates and never gain execution or approval authority.
- Skill execution evidence may feed memory and wisdom, but only approved wisdom
  may propose a new attributable skill revision.
- Lexical and partial-text discovery works without embeddings; semantic search
  is optional enrichment.
- Lane-message steering is at-least-once and checkpoint-idempotent; it is not
  presented as a cross-store exactly-once transaction.
- Wisdom must be status-gated; approved wisdom is the default serving view.
- Plugins cannot become authority by registration order or prompt injection.
- External A2A and CrewAI concepts are adapters over Kogwistar semantics.

## Alternatives

### New Agent Runtime

Rejected. It duplicates workflow routing, state, checkpoints, resume, tracing,
budgets, and nested execution.

### Goal and Plan Runtime Types

Rejected. Their behavior is already expressible as ordinary workflow topology.

### Agent Object Owns Memory, Knowledge, and Wisdom

Rejected. It collapses distinct authority and lifecycle planes into prompt
state.

### Everything Is Replaceable by Plugins

Rejected. Model, tool, skill, and context providers may be pluggable;
transaction, event, ACL, budget, and workflow authority may not.

### CrewAI or LangGraph as Embedded Runtime

Rejected. Optional import/export adapters are acceptable; dual execution
authority is not.

## Consequences

Benefits:

- small agent implementation with high reuse of tested primitives;
- plan, goal, and subagent behavior remains graph-visible and replayable;
- external memory and model providers remain optional;
- learning passes through wisdom evaluation instead of silent self-mutation;
- tools, MCP, HTTP, A2A, and skills can share versioned capability contracts.

Costs:

- the capability descriptor must grow beyond its current minimal form;
- a small deterministic hook lifecycle must be specified and tested;
- existing internal retrieval services need bounded agent-facing adapters;
- cross-plane provenance and ACL tests are required;
- optional interoperability adapters require explicit semantic loss reporting.

## Non-Goals

- a second workflow engine or autonomous scheduler;
- unrestricted recursive delegation;
- background durable spawn beyond existing runtime contracts;
- exactly-once external side effects;
- exporting private reasoning through A2A;
- making `kogwistar-llm-wiki`, CrewAI, LangChain, or any model provider a core
  dependency;
- treating generated summaries, skills, or telemetry as canonical truth.

## Future Extensions

- durable background remote-agent dispatch and reconciliation;
- signed A2A Agent Cards generated from capability descriptors;
- import/export adapters for supported external workflow formats;
- approved-wisdom skill compiler and serving index;
- richer multi-agent resource allocation and fairness policies;
- reusable UI projections for plans, goals, delegation, and provenance.
