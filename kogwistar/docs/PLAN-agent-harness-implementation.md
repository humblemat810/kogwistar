# Thin Agent Harness Implementation Plan

**Status:** Proposed
**Date:** 2026-09-25
**Governing decision:** ADR-020

## Objective

Deliver a thin Kogwistar agent harness that composes existing workflow,
conversation, knowledge, wisdom, budget, tracing, and authorization primitives.
Do not add a second runtime or authority store.

## Scope Rules

- Preserve the public Python API.
- Follow the strangler migration rule; do not reorganize unrelated packages.
- Add no new scheduler, token, checkpoint, memory database, or canonical Agent
  node type.
- Keep all tests deterministic with fake model payloads unless explicitly
  marked as live integration tests.
- Enforce ACL and budgets below prompt and plugin layers.
- Keep optional providers import-safe when their dependencies are absent.
- Implement sync behavior first only where async parity is explicitly tracked;
  do not claim parity before equivalent tests pass.

## Goal A Tracking: Phase 0-2 Core Slice

Goal A implements the minimal offline foundation only. It does not claim all
Phase 0-2 transport tools or later workflow behavior.

- [x] Typed immutable `AgentProfile`; capability requests cannot broaden caller
      authority.
- [x] Thin sync/async harness delegation to the existing workflow runtime.
- [x] Provider-qualified registry with collision rejection and deterministic
      disposal.
- [x] ACL/scope-filtered catalog with stable revisions, fingerprints, group
      membership, provider unload, and retained source history.
- [x] Exact, alias, prefix, partial, and deterministic lexical discovery;
      semantic results require readiness.
- [x] JSON-compatible bounded `SkillGraphArtifact` with source provenance,
      warnings, unsupported constructs, and explicit graph edges.
- [x] Deterministic frontmatter/Markdown parser for capability, MCP, script,
      instruction, and command-template references.
- [x] Effectful skill steps remain unbound/non-invocable until separate
      validation and authorization; shell wrappers and path traversal rejected.
- [x] Fake-payload contract tests, including stale revision, ACL, unload,
      deterministic fingerprints, async delegation, and command safety.
- [ ] REST/MCP read-only tool adapters, hook lifecycle, semantic provider, and
      full Phase-2 execution/provenance adapters remain open.

Evidence:

```text
tests/agent/test_agent_phase0_2.py
kogwistar/agent/{profile,harness,providers,catalog,skills}.py
```

## Goal B Tracking: Phase 2 Read Slice

- [x] Add transport-neutral `ReadScope` and bounded cursor `ReadPage`.
- [x] Add catalog search/browse/get with ACL and tenant/project/security-scope
      filtering.
- [x] Add descriptor-first skill reads, explicit graph artifact reads, and raw
      provider-native skill/resource reads.
- [x] Add MCP descriptor and capability descriptor reads without invocation or
      approval mutation.
- [x] Add bounded execution status/events/steps/checkpoints/lineage/evidence
      adapters over existing service method seams.
- [x] Add conversation-scoped memory default and explicit cross-conversation
      opt-in.
- [x] Add approved-wisdom default filtering and explicit non-approved denial.
- [x] Treat missing wisdom status as non-approved under the default fail-closed
      serving policy.
- [x] Preserve lexical catalog access when semantic readiness is absent.
- [x] Add fake-source tests for cursor bounds, scope isolation, raw/graph
      lookup, discovery-only MCP, wisdom filtering, and no mutation.
- [ ] Add concrete REST/MCP transport registration only after the existing MCP
      SDK compatibility issue is resolved and endpoint ownership is reviewed.

Evidence:

```text
kogwistar/agent/read_tools.py
tests/agent/test_agent_read_tools.py
```

## Goal C Tracking: Phase 3 Minimal Workflow Composition

- [x] Build ordinary normal, plan, and goal `WorkflowDesignArtifact` templates.
- [x] Keep `wf_mode` and contract metadata advisory; prove metadata removal
      preserves graph topology and operation semantics.
- [x] Bind deterministic fake model/tool payloads through the existing resolver
      and typed run-result contracts.
- [x] Use Goal B descriptor-first catalog reads from workflow bindings.
- [x] Seed existing `StateBackedBudgetLedger` with step, model-call, token,
      time, and cost ceilings; lower caller limits win.
- [x] Publish bounded remaining-budget hints as state metadata only.
- [x] Preflight model call/capability checks; prompt text cannot override
      runtime authority.
- [x] Support static and dynamic ordinary nested invocation request helpers with
      identity validation.
- [x] Exercise normal/plan/goal runtime graphs, two-iteration goal success,
      hard step suspension, ACL denial, and deterministic model/tool behavior.
- [ ] Queue/steering input, compression, delegation, wisdom hooks, A2A,
      CrewAI, LLM-Wiki, and concrete REST/MCP registration remain deferred.

Evidence:

```text
kogwistar/agent/workflows.py
kogwistar/agent/limits.py
kogwistar/agent/bindings.py
tests/agent/test_agent_goal_c.py
```

## Phase 0: Baseline and Contract Inventory

Deliverables:

- [ ] Inventory workflow invocation, checkpoints, resume, budget, trace, ACL,
      conversation context, memory retrieval, knowledge retrieval, and wisdom
      serving seams.
- [ ] Map existing REST, MCP, Python, and workflow operations to stable
      capability IDs under ADR-016.
- [ ] Record current gaps for run steps, lineage, evidence, memory search, and
      lifecycle-aware wisdom search.
- [ ] Confirm current sync/async/native capability matrix without expanding the
      Rust migration scope.

Tests and gates:

- [ ] Existing CI baseline recorded.
- [ ] No behavior change in this phase.

Exit criterion:

- One reviewed capability and ownership map with no duplicate authority.

## Phase 1: Agent Composition Contracts

Target package:

```text
kogwistar/agent/
  __init__.py
  profile.py
  harness.py
  context.py
  hooks.py
```

Deliverables:

- [ ] Add immutable `AgentProfile` with model, workflow, capability, context,
      compression, budget, skill-provider, plugin, and hook configuration.
- [ ] Add a thin harness that assembles dependencies and invokes the existing
      `WorkflowRuntime`.
- [ ] Reject unknown profiles, unsupported providers, invalid workflows, and
      attempts to grant capabilities through profile configuration.
- [ ] Add deterministic hook registration order, timeout, failure mode, effect
      declaration, capability requirement, and disposal.
- [ ] Define a read-only discovery-provider plugin contract and normalized
      catalog descriptor shared by skill, MCP, and documentation providers.
- [ ] Add the minimal provider registry now: provider-qualified identity,
      duplicate-owner rejection, deterministic registration order, and
      deterministic disposal.
- [ ] Keep MCP discovery/schema loading separate from authorized MCP invocation,
      even when one installed plugin implements both contracts.
- [ ] Keep protected ACL, budget, transaction, event, and scheduler operations
      outside hook replacement.
- [ ] Forbid hooks from choosing workflow branches, claiming steering,
      dispatching child runs, replacing checkpoints, or hiding side effects.

Tests:

- [ ] Agent profile validation and stable serialization.
- [ ] Profile cannot broaden caller capabilities.
- [ ] Hook ordering, timeout, cleanup, fail-open observation, and fail-closed
      security behavior.
- [ ] Provider registration cannot bypass descriptor validation, ACL, or
      capability requirements.
- [ ] Conflicting provider identities fail instead of depending on load order.
- [ ] Provider unload retracts only provider-owned descriptors and derived
      indexes.
- [ ] Harness invokes ordinary `WorkflowRuntime` with no alternate loop.

Exit criterion:

- A fake-model ordinary workflow executes through the harness with identical
  runtime artifacts to direct `WorkflowRuntime.run()`.

## Phase 2: Read-Only Agent Tools

Target package:

```text
kogwistar/agent/tools/
  execution.py
  memory.py
  knowledge.py
  wisdom.py
```

Deliverables:

- [ ] Expose bounded run status, events, steps, lineage, checkpoints, replay
      preview, and evidence reads through shared capability contracts.
- [ ] Add MCP parity for workflow steps, lineage, and evidence where REST
      already exposes them.
- [ ] Expose deterministic memory candidate search separately from optional
      LLM selection or pinning.
- [ ] Default memory scope to the current conversation; require explicit
      cross-conversation opt-in.
- [ ] Expose knowledge search, graph expansion, and entity retrieval with
      evidence references.
- [ ] Expose wisdom search with approved lessons as the default serving view.
- [ ] Require explicit status selection for pending, rejected, and deprecated
      wisdom artifacts.
- [ ] Add bounded `catalog.search`, `catalog.browse`, and `catalog.get`
      operations over skill, MCP, capability, and documentation descriptors.
- [ ] Add `skill.get`, `skill.resource_get`, `mcp.describe`, and
      `capability.describe` detail operations.
- [ ] Support graph-native group membership, provider-qualified stable IDs,
      tags, aliases, project/tenant scope, required capabilities, version, and
      approval status; expose tree-shaped browsing only as a projection.
- [ ] Provide exact, alias, hierarchy, prefix, BM25/full-text, and partial-text
      discovery without requiring embeddings.
- [ ] Add optional semantic ranking only when its projection is ready; lexical
      discovery remains available while embeddings are absent, pending, failed,
      or unavailable.
- [ ] Implement built-in SkillProvider and MCP discovery plugins over the same
      Phase-1 registry and catalog descriptor contract.
- [ ] Add optional skill ingestion that preserves the source package while
      projecting descriptors, procedural steps, ordering, capability
      requirements, MCP references, documentation links, groups, and source
      fingerprints into a rebuildable skill-MCP-capability hypergraph.
- [ ] Define a JSON-compatible `SkillGraphArtifact` with provider-qualified
      source identity, skill version/fingerprint, parser identity/version,
      bounded node and edge lists, warnings, and unsupported constructs.
- [ ] Add a small deterministic core parser for manifests, frontmatter, explicit
      capability/MCP references, package files, script references, and command
      templates; do not repurpose the generic document-summary ingester.
- [ ] Define an optional semantic-ingestion provider contract that accepts
      bounded normalized skill source and returns candidate `SkillGraphArtifact`
      data without materializing or executing it.
- [ ] Validate source fingerprints, graph bounds, source references,
      project/tenant scope, allowed node/edge kinds, and capability bindings in
      core before projection materialization.
- [ ] Map typed `skill_projection` requests to a dedicated project/workspace
      projection lane; never accept arbitrary target namespaces from providers.
- [ ] Model `instruction`, `capability_call`, `mcp_call`, `script_call`,
      `command_template`, `nested_workflow`, and `check` step kinds.
- [ ] Keep unbound, unsupported, or invalid effectful steps searchable but
      non-invocable.
- [ ] Require package-relative fingerprinted scripts, structured argv command
      templates, bounded environment/cwd/output/time, sandbox policy, and
      explicit filesystem/network/process capabilities. Do not default to shell
      string execution.
- [ ] Permit an installed skill to remain opaque when no parser exists; exact
      descriptor retrieval and provider-owned content loading must still work.
- [ ] Preserve provider-native skill loading/invocation after ingestion; expose
      graph-guided lookup as an additional path, never a replacement.
- [ ] Let an ordinary agent workflow retrieve a bounded skill subgraph, record
      selected step/source references, and invoke referenced typed capabilities
      through existing ACL and budget enforcement.
- [ ] Activate only selected validated MCP tool schemas for a model call rather
      than placing the entire remote tool catalog in every prompt.
- [ ] Return descriptors first; load full skill content and linked resources
      only after explicit selection.
- [ ] Do not add a broad unvalidated `skill.run` or `capability.invoke` escape
      hatch.
- [ ] Return bounded results, stable references, visibility scope, projection
      readiness, truncation, and cursor metadata.

Tests:

- [ ] RO/RW role and namespace enforcement for every transport.
- [ ] Engine ACL filtering for memory, knowledge, and wisdom results.
- [ ] Cross-conversation memory disabled by default.
- [ ] Pending or rejected wisdom never enters default agent context.
- [ ] Skill search does not leak inaccessible project or tenant skills.
- [ ] Full skill bodies and resources are absent from bootstrap context.
- [ ] Skill and MCP descriptors can be searched and browsed as a hierarchy.
- [ ] One descriptor may belong to multiple groups without duplicate identity.
- [ ] Raw and graph-ingested forms resolve to the same provider-qualified skill
      version and preserve source provenance.
- [ ] One deterministic fake skill works through both provider-native and
      graph-guided paths without changing effective capabilities.
- [ ] Graph-guided execution persists selected step, source fingerprint, MCP
      schema/capability reference, and resulting workflow evidence.
- [ ] Parsed graph content cannot grant capability or become executable workflow
      without separate validation and authorization.
- [ ] Core-only deterministic ingestion works with no optional provider or
      embedding dependency.
- [ ] Malformed, oversized, cross-tenant, wrong-fingerprint, and unsupported
      provider artifacts fail before materialization.
- [ ] Semantic-provider failure leaves raw skill use and the previous valid
      graph projection available.
- [ ] A stale v1 parse finishing after v2 cannot replace the v2 projection.
- [ ] Script path traversal, arbitrary shell strings, implicit environment
      inheritance, and undeclared process/network/filesystem access are denied.
- [ ] Search returns useful deterministic results with embedding disabled.
- [ ] Pending or failed semantic indexing does not hide lexical catalog entries.
- [ ] MCP provider failure does not break local skill or documentation search.
- [ ] MCP, REST, Python, and workflow adapters preserve output invariants.

Exit criterion:

- An agent can inspect execution, memory, knowledge, and approved wisdom without
  direct backend access or hidden mutation.

## Phase 3: Minimal Agent Workflow (Goal C baseline)

Deliverables:

- [x] Add minimal ordinary workflow templates, including
      `Observe -> Decide -> Act -> Check -> Done`.
- [x] Use existing resolver registration and typed result models.
- [x] Publish current mode and bounded remaining-budget estimates through
      ordinary workflow state metadata.
- [x] Enforce available hard step, model-call, token, time, and cost limits
      through the runtime budget path and binding preflight.
- [x] Persist state patches and result evidence through existing run/step
      artifacts when the ordinary runtime executes the graph.
- [ ] Define typed active-run input policies: `steer`, `queue`, and
      `cancel_and_replace`.
- [ ] Persist authoritative user input in the conversation graph and delivery
      state in existing lane-message infrastructure.
- [ ] Add an ordinary reusable workflow control-point resolver that atomically
      selects and claims bounded authorized active-run lane messages matching
      target run/lane and accepted message kind, then returns normal state
      patches.
- [ ] Prefer a run-specific control inbox so unrelated or wrong-target messages
      are not claimed and rejected after the fact.
- [ ] Treat delivery as at least once: persist a per-inbox applied sequence
      high-water mark in checkpointed workflow state, add bounded `message_id`
      deduplication only for policies permitting sequence holes, skip
      redelivery, and acknowledge only after state is durable.
- [ ] Store authoritative input once in the conversation graph; lane messages
      carry immutable input-node references/fingerprints and delivery metadata.
- [ ] Terminalize steering aimed at a completed run using an existing terminal
      status plus `target_terminal` reason, or convert it to next-turn input
      only under an explicit sender fallback.
- [ ] Put control-point nodes explicitly in workflow templates; do not poll or
      inject steering from hidden harness/runtime code.
- [ ] Let control-point metadata declare accepted message kinds, maximum claims,
      and queue policy without introducing a new node class.
- [ ] Place template control points before model calls, after tool results,
      before expensive actions, or at iteration boundaries as appropriate.
- [ ] Keep same-lane control input FIFO and preserve explicit input sequence and
      target-run correlation.
- [ ] Support reconnectable ordered event reads from a durable cursor.
- [ ] Treat client disconnect separately from cancellation for background work.

Tests:

- [x] Deterministic happy path with fake model and fake tools.
- [ ] Tool failure and retry path.
- [x] Budget exhaustion before model dispatch and hard step suspension.
- [x] Prompt-provided remaining budget cannot override runtime enforcement.
- [ ] Steering changes future work but cannot rewrite emitted output or undo a
      completed external effect.
- [ ] A workflow lacking a control-point node does not consume steering input.
- [ ] Control-point placement changes only where steering may be observed; it
      does not weaken hard cancellation, budget, or security guards.
- [ ] Crash after claim but before checkpoint redelivers and applies once.
- [ ] Crash after checkpoint but before acknowledgement redelivers and is
      skipped through checkpointed sequence/message deduplication.
- [ ] Wrong-run or unsupported message kinds are not claimed from the control
      inbox.
- [ ] Run termination before its next control point does not strand or silently
      retarget steering.
- [ ] Queued input waits behind the active same-lane item.
- [ ] Duplicate cancellation and duplicate control-message delivery are
      idempotent.
- [ ] Unauthorized steering or cross-tenant queue insertion is rejected.
- [ ] Disconnect/reconnect resumes event reading without rerunning completed
      steps.
- [ ] Crash/checkpoint/resume path where currently supported.

Exit criterion:

- One end-to-end fake-payload agent run is inspectable and replayable using
  existing workflow diagnostics.

## Phase 4: Plan and Goal Profiles (initial graph slice covered by Goal C)

Deliverables:

- [x] Add a plan workflow template with explicit plan artifact, approval, act,
      and check steps.
- [x] Reuse ADR-019 cyclic goal workflow pattern without runtime branching on
      mode metadata.
- [x] Treat mode metadata as advisory for discovery, prompt assembly, linting,
      and UI only.
- [ ] Add mode-specific validators as design lint, not scheduler behavior.

Tests:

- [ ] Plan approve and reject paths.
- [x] Goal satisfied/repeated paths are covered by deterministic runtime tests.
- [x] Goal stops on budget exhaustion.
- [x] Same workflow graph executes when advisory mode metadata is removed.

Exit criterion:

- Normal, plan, and goal behavior differ only by workflow graph/profile, not
  runtime implementation.

## Phase 5: Subagent Delegation

Deliverables:

- [ ] Express delegation through `WorkflowInvocationRequest` only.
- [ ] Add profile selection for alternate model/provider, including vision,
      local-model, MCP specialist, and low-cost lookup examples.
- [ ] Default child context to isolated and return a bounded structured result.
- [ ] Allocate explicit child budget and capability subset.
- [ ] Persist generic parent-step to child-run lineage and invocation identity.
- [ ] Keep v1 invoke-and-await; do not imply durable background spawn.

Tests:

- [ ] Child uses a different fake model profile.
- [ ] Vision-capable child returns a structured artifact reference.
- [ ] Child cannot use a capability absent from its delegated subset.
- [ ] Child output does not leak its complete context into the parent.
- [ ] Retry/recovery reuses invocation identity without blind duplicate work.

Exit criterion:

- A parent run delegates one bounded task and preserves inspectable lineage,
  capability, budget, and evidence contracts.

## Phase 6: Context Compression

Deliverables:

- [ ] Define versioned `low`, `medium`, `high`, and `ultra` compression policy
      contracts independently from model reasoning effort.
- [ ] Add threshold hooks that request compression without mutating history.
- [ ] Execute compression as an ordinary audited workflow.
- [ ] Persist summary nodes and provenance edges to covered conversation nodes.
- [ ] Retain full original conversation graph.
- [ ] Support same-conversation context first and opt-in cross-conversation
      memory through existing ACL filters.

Tests:

- [ ] Each policy selects the documented retained-context budget.
- [ ] Original messages remain retrievable after compression.
- [ ] Summary provenance covers the expected source nodes.
- [ ] Branches remain distinct.
- [ ] Unauthorized cross-conversation content is never summarized or injected.

Exit criterion:

- Long conversation context can be reduced without loss of authoritative
  history or ACL scope.

## Phase 7: Wisdom Distillation and Skill Projection

Deliverables:

- [ ] Add a best-effort post-run hook that submits a distillation request.
- [ ] Run distillation through the existing wisdom workflow and proposal
      lifecycle.
- [ ] Separate proposal generation, evaluation, approval, rejection, and
      deprecation.
- [ ] Add an optional compiler from approved wisdom to an on-demand skill
      projection.
- [ ] Keep the generated skill linked to source lessons, evidence, version, and
      approval status.
- [ ] Link selected skill revision/step to run, capability/MCP invocation,
      result, performance, error, and user-feedback evidence.
- [ ] Permit memory to record observations about skill use without mutating the
      skill or treating memory review as wisdom approval.
- [ ] Route generalized procedure improvements through proposal, evaluation,
      and approval before creating a new attributable skill revision.

Tests:

- [ ] Successful run may produce no proposal when nothing generalizes.
- [ ] User correction can produce a pending proposal.
- [ ] Rejected proposal does not become a skill.
- [ ] Approved lesson becomes a versioned skill projection.
- [ ] Distillation failure cannot fail or rewrite the source run.
- [ ] Reviewed memory alone cannot revise a skill or approve a candidate edge.
- [ ] New approved revision leaves prior skill/source/projection lineage
      inspectable.

Exit criterion:

- Learning is evidence-backed, reviewed, reversible, and never silent
  self-mutation.

## Phase 8: Plugins and Optional LLM-Wiki Provider

Deliverables:

- [ ] Define minimal model, tool, skill, memory, and compressor provider
      protocols only where multiple implementations exist.
- [ ] Extend the Phase-1 registry only where external package discovery,
      installation, or hot reload requires it; do not create a second registry.
- [ ] Package local filesystem/repository skills as one SkillProvider plugin,
      not hard-coded core discovery behavior.
- [ ] Package MCP discovery, selected schema loading, and bounded invocation as
      separate contracts that one MCP plugin may implement together.
- [ ] Add an optional `kogwistar-llm-wiki` MCP or public-capability adapter for
      memory/knowledge retrieval.
- [ ] Add an optional LLM-Wiki semantic skill-ingestion adapter using the same
      `SkillGraphArtifact` contract; keep core-only deterministic ingestion as
      the mandatory fallback.
- [ ] Map LLM-Wiki skill digestion to an isolated projection lane such as
      `ws:<workspace>:g:projection:lane:skills`, with provider, skill version,
      source fingerprint, parser profile, schema version, and project/tenant
      metadata.
- [ ] Use LLM-Wiki for semantic decomposition, cross-link candidates,
      deduplication, reparse, and stale-link maintenance only; do not delegate
      validation, capability binding, approval, or execution authority.
- [ ] Define a project-plugin package shape that may contribute glossary source
      material, project skills, policies, and bounded tool adapters.
- [ ] Ingest company terms, abbreviations, aliases, definitions, sources, and
      validity metadata into the project/tenant knowledge namespace rather than
      copying them into skill bodies.
- [ ] Add a bounded project glossary context source using exact match, alias
      expansion, and optional semantic retrieval.
- [ ] Preserve core importability and behavior when optional dependencies are
      absent.
- [ ] Prevent plugins from replacing authority checks or registering hidden
      side effects without declared capabilities.

Tests:

- [ ] Missing optional plugin leaves core operational.
- [ ] Plugin load/unload cleans resources.
- [ ] Plugin reload cannot leave stale catalog graph, lexical, or semantic
      entries from the prior source fingerprint.
- [ ] Plugin failure is isolated according to declared failure mode.
- [ ] LLM-Wiki adapter receives bounded authorized requests and returns stable
      references.
- [ ] LLM-Wiki absence or outage does not disable raw skills, deterministic
      descriptors, or a prior valid digested projection.
- [ ] LLM-Wiki inferred edges retain confidence/source/parser provenance and
      remain candidates until accepted by the relevant policy.
- [ ] Plugin uninstall removes provider-owned active catalog/projection records
      without deleting historical run, feedback, memory, or wisdom evidence.
- [ ] Project glossary retrieval honors tenant/project ACL and injects only
      query-relevant terms.
- [ ] Project procedure skills may reference glossary entity IDs without
      becoming glossary authority.

Exit criterion:

- Optional providers can extend the harness without reverse dependency or
  semantic fork.

## Phase 9: Optional A2A Adapter

Suggested package:

```text
kogwistar/server/a2a/
```

Deliverables:

- [ ] Extend capability descriptors enough to generate a versioned Agent Card.
- [ ] Map A2A message/task lifecycle to root workflow submission, inspection,
      cancellation, suspension, and result artifacts.
- [ ] Persist external context/task identifiers separately from run and trace
      identity.
- [ ] Support polling first, then streaming using existing ordered run events.
- [ ] Defer push callbacks until durable retry, authentication, SSRF protection,
      and delivery audit are specified.
- [ ] Export results and evidence references, never private reasoning traces.

Tests:

- [ ] Agent Card schema and security declaration.
- [ ] Submit, inspect, stream, cancel, and input-required/resume mapping.
- [ ] Identifier non-conflation tests.
- [ ] Authorization and tenant isolation.
- [ ] Reconnect and duplicate-request idempotency.

Exit criterion:

- One external A2A client can execute and inspect a bounded Kogwistar workflow
  without becoming workflow authority.

## Phase 10: Optional CrewAI Importer

Suggested package:

```text
kogwistar/interop/crewai/
```

Deliverables:

- [ ] Translate supported static Flow/task/dependency structures into
      `WorkflowDesignArtifact`.
- [ ] Translate role agents to agent profiles or resolver bindings.
- [ ] Translate delegation to ordinary nested invocation.
- [ ] Emit explicit diagnostics for dynamic manager behavior, callbacks, memory,
      or guardrails that cannot be represented safely.
- [ ] Never execute CrewAI as a second embedded authority.

Tests:

- [ ] Deterministic supported import fixture.
- [ ] Round-trip semantic comparison for the supported subset.
- [ ] Explicit rejection of unsupported or lossy constructs.

Exit criterion:

- Supported CrewAI descriptions become ordinary inspectable Kogwistar workflow
  designs; unsupported semantics never silently degrade.

## Cross-Cutting Verification

- [ ] Unit tests carry appropriate `unit`, `ci`, and domain markers.
- [ ] Deterministic end-to-end tests use fake model/tool payloads.
- [ ] Optional live-provider tests are separately marked and excluded from CI.
- [ ] No test enables third-party remote telemetry by default.
- [ ] Resource-owning fixtures close engines, workers, exporters, and providers.
- [ ] Sync/async parity is claimed only after equivalent lifecycle tests pass.
- [ ] Python and applicable native parity tests pass after any permitted native
      bug fix.
- [ ] Documentation links every exposed capability to contract tests.

## MVP Boundary

The first useful release ends after Phase 5:

```text
typed AgentProfile
+ thin harness
+ execution/memory/knowledge/wisdom read tools
+ searchable graph-native skill/MCP catalog with lexical fallback and optional
  skill-MCP-capability hypergraph ingestion
+ one minimal agent workflow
+ explicit workflow control-point steering, queued input, cancellation, and
  reconnectable events
+ plan and goal profiles
+ invoke-and-await subagent delegation
```

Compression, wisdom skill projection, optional LLM-Wiki integration, A2A, and
CrewAI import remain additive phases. They must not delay validation of the
thin composition architecture.

## Final Acceptance

The implementation is complete only when all statements below are true:

- [ ] Removing advisory agent/plan/goal labels does not change runtime
      correctness.
- [ ] Every run remains an ordinary workflow run with ordinary steps,
      checkpoints, trace, budget, ACL, and provenance.
- [ ] No plugin, profile, prompt, or role can grant itself authority.
- [ ] Full conversation history survives compression.
- [ ] Steering and queued input are durable, ordered, authorized, and applied
      only at documented safe points.
- [ ] Cross-conversation memory is explicit and ACL-filtered.
- [ ] Default wisdom retrieval serves approved lessons only.
- [ ] Skills are progressively disclosed and never replace project knowledge
      authority.
- [ ] Raw skill packages remain attributable source artifacts; parsed skill
      graphs remain rebuildable discovery projections and never grant authority.
- [ ] Skill ingestion is additive: provider-native use remains functional while
      ordinary workflows may also use bounded graph-digested procedures.
- [ ] Company terminology is evidence-backed project knowledge with ACL scope.
- [ ] Subagents are bounded child runs, not hidden threads or schedulers.
- [ ] Optional integrations are absent-safe and cannot become core truth.
- [ ] Agent behavior is explainable through existing graph and workflow
      diagnostics.
