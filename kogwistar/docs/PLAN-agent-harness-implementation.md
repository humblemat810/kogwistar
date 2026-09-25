# Thin Agent Harness Implementation Plan

**Status:** Implemented - verification complete
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

## Goal-to-Phase Mapping

Goal names group the numbered phases; they are not a second checklist:

```text
Goal A -> Phase 0-2: inventory, composition contracts, read-only tools
Goal B -> Phase 3-5: workflow control, plan/goal profiles, subagent delegation
Goal C -> Phase 6-8: context compression, wisdom/skill projection, plugins
```

The numbered `Phase 0` through `Phase 10` sections below are the canonical
implementation and acceptance checklist. Do not duplicate or infer phase
completion from a Goal label. Each completed item must retain its own evidence;
unchecked items remain open even when another item in the same Goal is done.

Phase status snapshot (2026-09-25):

| Scope | Status | Evidence |
| --- | --- | --- |
| Phase 0-2 | Complete | contract, discovery, read-tool, and ingestion tests |
| Phase 3-5 | Complete | control, budget, plan/goal, and delegation tests |
| Phase 6-8 | Complete | compression, wisdom, skill, plugin, and provider tests |
| Phase 9 | Complete | A2A polling/stream/push contract tests |
| Phase 10 | Complete | deterministic CrewAI import/diagnostic tests |
| Cross-Cutting | Complete | marker, fake-payload, ACL, cleanup, parity, docs gates |
| Final Acceptance | Complete | all acceptance assertions and full CI |

This snapshot records checklist completion, not a claim that live LLM,
transport-specific cancellation, or a native Rust agent runtime exists. Those
remain explicit non-goals or separately marked integration work.

Current evidence:

```text
Phase 0-2 implemented contract slice:
  kogwistar/agent/{profile,harness,providers,catalog,skills,read_tools}.py
  tests/agent/test_agent_phase0_2.py
  tests/agent/test_agent_read_tools.py

Phase 3-4 implemented contract slice:
  kogwistar/agent/{workflows,limits,bindings}.py
  tests/agent/test_agent_goal_c.py

Current control-input slice (Phase 3 implementation):
  kogwistar/agent/control.py
  kogwistar/messaging/service.py
  kogwistar/engine_core/{meta_lane_messages,engine_sqlite,engine_postgres_meta}.py
  rust/crates/kogwistar-{store,store-memory,store-sqlite,store-postgres,python}/
  tests/agent/test_agent_goal_c.py
  tests/agent/test_agent_read_tools.py
  tests/core/test_lane_message_meta_store_contract.py
  tests/core/test_lane_messaging.py

The control-input slice now has Python in-memory/SQLite/PostgreSQL behavior,
filtered Rust store/bridge operations, ACL preflight, explicit template
control-point placement, provenance coordinates, and deterministic fake crash
window/read-cursor tests. The persisted WorkflowRuntime checkpoint crash
window is covered by
`test_control_point_checkpoint_resume_ack_is_durable_with_real_runtime`.
Run-specific inbox provisioning and client-disconnect separation are covered
by deterministic control/A2A tests. The adapter contract deliberately keeps
disconnect observation separate from explicit runtime cancellation; durable
transport-specific cancellation remains outside this harness slice.

Phase 5-10 now have contract implementations and deterministic tests:
  Phase 5 delegation, Phase 6 compression, Phase 7 distillation/projection,
  Phase 8 local/MCP/LLM-Wiki provider seams, Phase 9 A2A polling adapter, and
  Phase 10 static CrewAI importer. Remaining non-checklist items below are
  intentional; they require transport wiring, live backend evidence, or
  broader integration.

Verification record (2026-09-25):
  `tests/agent` contract runs use deterministic fake payloads. The MCP golden
  contract is regenerated from the live registries (60 tools, three surfaces).
  `tests/conftest.py` forces `ANONYMIZED_TELEMETRY=FALSE` and makes PostHog
  network methods no-op mocks. Engine-pair fixtures close both engines;
  provider and hook registries dispose owned resources. Focused agent/MCP
  suite passes 96 tests. Full `pytest -m ci -q` passes 896 tests, skips 8,
  and deselects 1148 tests; tests use deterministic/fake model payloads
  unless their existing marker explicitly exercises a local backend container.
  Full-repository Ruff and `cargo fmt --all -- --check` pass. Rust
  store/bridge package tests pass (46 passed, 1 ignored). Live LLM remains
  intentionally unclaimed; PostgreSQL/Chroma container coverage is present
  only where the CI marker provisions it.
  A2A push delivery additionally requires the queue to confirm `durable=True`
  before the adapter reports success.
  Rust store/bridge parity passes the relevant package tests. No native agent
  lifecycle authority is introduced; shared lane/store bridge parity is the
  applicable native coverage, while a Rust agent runtime remains out of scope.
```

## Phase 0: Baseline and Contract Inventory

Deliverables:

- [x] Inventory workflow invocation, checkpoints, resume, budget, trace, ACL,
      conversation context, memory retrieval, knowledge retrieval, and wisdom
      serving seams.
- [x] Map existing REST, MCP, Python, and workflow operations to stable
      capability IDs under ADR-016.
- [x] Record current gaps for run steps, lineage, evidence, memory search, and
      lifecycle-aware wisdom search.
- [x] Confirm current sync/async/native capability matrix without expanding the
      Rust migration scope.

Tests and gates:

- [x] Existing CI baseline recorded.
- [x] No behavior change in this phase.

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

- [x] Add immutable `AgentProfile` with model, workflow, capability, context,
      compression, budget, skill-provider, plugin, and hook configuration.
- [x] Add a thin harness that assembles dependencies and invokes the existing
      `WorkflowRuntime`.
- [x] Reject unknown workflow/provider/hook selections and attempts to grant
      capabilities through profile configuration; workflow design validation
      remains owned by the existing runtime.
- [x] Add deterministic hook registration order, timeout, failure mode, effect
      declaration, capability requirement, and disposal.
- [x] Define a read-only discovery-provider plugin contract and normalized
      catalog descriptor shared by skill, MCP, and documentation providers.
- [x] Add the minimal provider registry now: provider-qualified identity,
      duplicate-owner rejection, deterministic registration order, and
      deterministic disposal.
- [x] Keep MCP discovery/schema loading separate from authorized MCP invocation,
      even when one installed plugin implements both contracts.
- [x] Keep protected ACL, budget, transaction, event, and scheduler operations
      outside hook replacement.
- [x] Forbid hooks from choosing workflow branches, claiming steering,
      dispatching child runs, replacing checkpoints, or hiding side effects.

Tests:

- [x] Agent profile validation and stable serialization.
- [x] Profile cannot broaden caller capabilities.
- [x] Hook ordering, timeout, cleanup, fail-open observation, and fail-closed
      security behavior.
- [x] Provider registration cannot bypass descriptor validation, ACL, or
      capability requirements.
- [x] Conflicting provider identities fail instead of depending on load order.
- [x] Provider unload retracts only provider-owned descriptors and derived
      indexes.
- [x] Harness invokes ordinary `WorkflowRuntime` with no alternate loop.

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

- [x] Expose bounded run status, events, steps, lineage, checkpoints, replay
      preview, and evidence reads through shared capability contracts.
- [x] Add MCP parity for workflow steps, lineage, and evidence where REST
      already exposes them.
- [x] Expose deterministic memory candidate search separately from optional
      LLM selection or pinning.
- [x] Default memory scope to the current conversation; require explicit
      cross-conversation opt-in.
- [x] Expose knowledge search, graph expansion, and entity retrieval with
      evidence references.
- [x] Expose wisdom search with approved lessons as the default serving view.
- [x] Require explicit status selection for pending, rejected, and deprecated
      wisdom artifacts.
- [x] Add bounded `catalog.search`, `catalog.browse`, and `catalog.get`
      operations over skill, MCP, capability, and documentation descriptors.
- [x] Add `skill.get`, `skill.resource_get`, `mcp.describe`, and
      `capability.describe` detail operations.
- [x] Support graph-native group membership, provider-qualified stable IDs,
      tags, aliases, project/tenant scope, required capabilities, version, and
      approval status; expose tree-shaped browsing only as a projection.
- [x] Provide exact, alias, hierarchy, prefix, BM25/full-text, and partial-text
      discovery without requiring embeddings.
- [x] Add optional semantic ranking only when its projection is ready; lexical
      discovery remains available while embeddings are absent, pending, failed,
      or unavailable.
- [x] Implement built-in SkillProvider and MCP discovery plugins over the same
      Phase-1 registry and catalog descriptor contract.
- [x] Add optional skill ingestion that preserves the source package while
      projecting descriptors, procedural steps, ordering, capability
      requirements, MCP references, documentation links, groups, and source
      fingerprints into a rebuildable skill-MCP-capability hypergraph.
- [x] Define a JSON-compatible `SkillGraphArtifact` with provider-qualified
      source identity, skill version/fingerprint, parser identity/version,
      bounded node and edge lists, warnings, and unsupported constructs.
- [x] Add a small deterministic core parser for manifests, frontmatter, explicit
      capability/MCP references, package files, script references, and command
      templates; do not repurpose the generic document-summary ingester.
- [x] Define an optional semantic-ingestion provider contract that accepts
      bounded normalized skill source and returns candidate `SkillGraphArtifact`
      data without materializing or executing it.
- [x] Validate source fingerprints, graph bounds, source references,
      project/tenant scope, allowed node/edge kinds, and capability bindings in
      core before projection materialization.
- [x] Map typed `skill_projection` requests to a dedicated project/workspace
      projection lane; never accept arbitrary target namespaces from providers.
- [x] Model `instruction`, `capability_call`, `mcp_call`, `script_call`,
      `command_template`, `nested_workflow`, and `check` step kinds.
- [x] Keep unbound, unsupported, or invalid effectful steps searchable but
      non-invocable.
- [x] Require package-relative fingerprinted scripts, structured argv command
      templates, bounded environment/cwd/output/time, sandbox policy, and
      explicit filesystem/network/process capabilities. Do not default to shell
      string execution.
- [x] Permit an installed skill to remain opaque when no parser exists; exact
      descriptor retrieval and provider-owned content loading must still work.
- [x] Preserve provider-native skill loading/invocation after ingestion; expose
      graph-guided lookup as an additional path, never a replacement.
- [x] Let an ordinary agent workflow retrieve a bounded skill subgraph, record
      selected step/source references, and invoke referenced typed capabilities
      through existing ACL and budget enforcement.
- [x] Activate only selected validated MCP tool schemas for a model call rather
      than placing the entire remote tool catalog in every prompt.
- [x] Return descriptors first; load full skill content and linked resources
      only after explicit selection.
- [x] Do not add a broad unvalidated `skill.run` or `capability.invoke` escape
      hatch.
- [x] Return bounded results, stable references, visibility scope, projection
      readiness, truncation, and cursor metadata.

Tests:

- [x] RO/RW role and namespace enforcement for every transport.
- [x] Engine ACL filtering for memory, knowledge, and wisdom results.
- [x] Cross-conversation memory disabled by default.
- [x] Pending or rejected wisdom never enters default agent context.
- [x] Skill search does not leak inaccessible project or tenant skills.
- [x] Full skill bodies and resources are absent from bootstrap context.
- [x] Skill and MCP descriptors can be searched and browsed as a hierarchy.
- [x] One descriptor may belong to multiple groups without duplicate identity.
- [x] Raw and graph-ingested forms resolve to the same provider-qualified skill
      version and preserve source provenance.
- [x] One deterministic fake skill works through both provider-native and
      graph-guided paths without changing effective capabilities.
- [x] Graph-guided execution persists selected step, source fingerprint, MCP
      schema/capability reference, and resulting workflow evidence.
- [x] Parsed graph content cannot grant capability or become executable workflow
      without separate validation and authorization.
- [x] Core-only deterministic ingestion works with no optional provider or
      embedding dependency.
- [x] Malformed, oversized, cross-tenant, wrong-fingerprint, and unsupported
      provider artifacts fail before materialization.
- [x] Semantic-provider failure leaves raw skill use and the previous valid
      graph projection available.
- [x] A stale v1 parse finishing after v2 cannot replace the v2 projection.
- [x] Script path traversal, arbitrary shell strings, implicit environment
      inheritance, and undeclared process/network/filesystem access are denied.
- [x] Search returns useful deterministic results with embedding disabled.
- [x] Pending or failed semantic indexing does not hide lexical catalog entries.
- [x] MCP provider failure does not break local skill or documentation search.
- [x] MCP, REST, Python, and workflow adapters preserve output invariants.

Exit criterion:

- An agent can inspect execution, memory, knowledge, and approved wisdom without
  direct backend access or hidden mutation.

## Phase 3: Minimal Agent Workflow

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
- [x] Define typed active-run input policies: `steer`, `queue`, and
      `cancel_and_replace`.
- [x] Persist authoritative user input in the conversation graph and delivery
      state in existing lane-message infrastructure.
- [x] Add an ordinary reusable workflow control-point resolver that atomically
      selects and claims bounded authorized active-run lane messages matching
      target run/lane and accepted message kind, then returns normal state
      patches.
- [x] Prefer a run-specific control inbox so unrelated or wrong-target messages
      are not claimed and rejected after the fact.
- [x] Treat delivery as at least once: persist a per-inbox applied sequence
      high-water mark in checkpointed workflow state, add bounded `message_id`
      deduplication only for policies permitting sequence holes, skip
      redelivery, and acknowledge only after state is durable.
- [x] Store authoritative input once in the conversation graph; lane messages
      carry immutable input-node references/fingerprints and delivery metadata.
- [x] Terminalize steering aimed at a completed run using an existing terminal
      status plus `target_terminal` reason, or convert it to next-turn input
      only under an explicit sender fallback.
- [x] Put control-point nodes explicitly in workflow templates; do not poll or
      inject steering from hidden harness/runtime code.
- [x] Let control-point metadata declare accepted message kinds, maximum claims,
      and queue policy without introducing a new node class.
- [x] Place template control points before model calls, after tool results,
      before expensive actions, or at iteration boundaries as appropriate.
- [x] Keep same-lane control input FIFO and preserve explicit input sequence and
      target-run correlation.
- [x] Support reconnectable ordered event reads from a durable cursor.
- [x] Treat client disconnect separately from cancellation for background work.

Tests:

- [x] Deterministic happy path with fake model and fake tools.
- [x] Tool failure and retry path.
- [x] Budget exhaustion before model dispatch and hard step suspension.
- [x] Prompt-provided remaining budget cannot override runtime enforcement.
- [x] Steering changes future work but cannot rewrite emitted output or undo a
      completed external effect.
- [x] A workflow lacking a control-point node does not consume steering input.
- [x] Control-point placement changes only where steering may be observed; it
      does not weaken hard cancellation, budget, or security guards.
- [x] Crash after claim but before checkpoint redelivers and applies once.
- [x] Crash after checkpoint but before acknowledgement redelivers and is
      skipped through checkpointed sequence/message deduplication.
- [x] Wrong-run or unsupported message kinds are not claimed from the control
      inbox.
- [x] Run termination before its next control point does not strand or silently
      retarget steering.
- [x] Queued input waits behind the active same-lane item.
- [x] Duplicate cancellation and duplicate control-message delivery are
      idempotent.
- [x] Unauthorized steering or cross-tenant queue insertion is rejected.
- [x] Disconnect/reconnect resumes event reading without rerunning completed
      steps.
- [x] Crash/checkpoint/resume path where currently supported, including the
      persisted WorkflowRuntime path when the backing engine supports it.

Exit criterion:

- One end-to-end fake-payload agent run is inspectable and replayable using
  existing workflow diagnostics.

## Phase 4: Plan and Goal Profiles

Deliverables:

- [x] Add a plan workflow template with explicit plan artifact, approval, act,
      and check steps.
- [x] Reuse ADR-019 cyclic goal workflow pattern without runtime branching on
      mode metadata.
- [x] Treat mode metadata as advisory for discovery, prompt assembly, linting,
      and UI only.
- [x] Add mode-specific validators as design lint, not scheduler behavior.

Tests:

- [x] Plan approve and reject paths.
- [x] Goal satisfied/repeated paths are covered by deterministic runtime tests.
- [x] Goal stops on budget exhaustion.
- [x] Same workflow graph executes when advisory mode metadata is removed.

Exit criterion:

- Normal, plan, and goal behavior differ only by workflow graph/profile, not
  runtime implementation.

## Phase 5: Subagent Delegation

Deliverables:

- [x] Express delegation through `WorkflowInvocationRequest` only.
- [x] Add profile selection for alternate model/provider, including vision,
      local-model, MCP specialist, and low-cost lookup examples.
- [x] Default child context to isolated and return a bounded structured result.
- [x] Allocate explicit child budget and capability subset.
- [x] Persist generic parent-step to child-run lineage and invocation identity.
- [x] Keep v1 invoke-and-await; do not imply durable background spawn.

Tests:

- [x] Child uses a different fake model profile.
- [x] Vision-capable child returns a structured artifact reference.
- [x] Child cannot use a capability absent from its delegated subset.
- [x] Child output does not leak its complete context into the parent.
- [x] Retry/recovery reuses invocation identity without blind duplicate work.

Exit criterion:

- A parent run delegates one bounded task and preserves inspectable lineage,
  capability, budget, and evidence contracts.

## Phase 6: Context Compression

Deliverables:

- [x] Define versioned `low`, `medium`, `high`, and `ultra` compression policy
      contracts independently from model reasoning effort.
- [x] Add threshold hooks that request compression without mutating history.
- [x] Execute compression as an ordinary audited workflow.
- [x] Persist summary nodes and provenance edges to covered conversation nodes.
- [x] Retain full original conversation graph.
- [x] Support same-conversation context first and opt-in cross-conversation
      memory through existing ACL filters.

Tests:

- [x] Each policy selects the documented retained-context budget.
- [x] Original messages remain retrievable after compression.
- [x] Summary provenance covers the expected source nodes.
- [x] Branches remain distinct.
- [x] Unauthorized cross-conversation content is never summarized or injected.

Exit criterion:

- Long conversation context can be reduced without loss of authoritative
  history or ACL scope.

## Phase 7: Wisdom Distillation and Skill Projection

Deliverables:

- [x] Add a best-effort post-run hook that submits a distillation request.
- [x] Run distillation through the existing wisdom workflow and proposal
      lifecycle.
- [x] Separate proposal generation, evaluation, approval, rejection, and
      deprecation.
- [x] Add an optional compiler from approved wisdom to an on-demand skill
      projection.
- [x] Keep the generated skill linked to source lessons, evidence, version, and
      approval status.
- [x] Link selected skill revision/step to run, capability/MCP invocation,
      result, performance, error, and user-feedback evidence.
- [x] Permit memory to record observations about skill use without mutating the
      skill or treating memory review as wisdom approval.
- [x] Route generalized procedure improvements through proposal, evaluation,
      and approval before creating a new attributable skill revision.

Tests:

- [x] Successful run may produce no proposal when nothing generalizes.
- [x] User correction can produce a pending proposal.
- [x] Rejected proposal does not become a skill.
- [x] Approved lesson becomes a versioned skill projection.
- [x] Distillation failure cannot fail or rewrite the source run.
- [x] Reviewed memory alone cannot revise a skill or approve a candidate edge.
- [x] New approved revision leaves prior skill/source/projection lineage
      inspectable.

Exit criterion:

- Learning is evidence-backed, reviewed, reversible, and never silent
  self-mutation.

## Phase 8: Plugins and Optional LLM-Wiki Provider

Deliverables:

- [x] Define minimal model, tool, skill, memory, and compressor provider
      protocols only where multiple implementations exist.
- [x] Extend the Phase-1 registry only where external package discovery,
      installation, or hot reload requires it; do not create a second registry.
- [x] Package local filesystem/repository skills as one SkillProvider plugin,
      not hard-coded core discovery behavior.
- [x] Package MCP discovery, selected schema loading, and bounded invocation as
      separate contracts that one MCP plugin may implement together.
- [x] Add an optional `kogwistar-llm-wiki` MCP or public-capability adapter for
      memory/knowledge retrieval.
- [x] Add an optional LLM-Wiki semantic skill-ingestion adapter using the same
      `SkillGraphArtifact` contract; keep core-only deterministic ingestion as
      the mandatory fallback.
- [x] Map LLM-Wiki skill digestion to an isolated projection lane such as
      `ws:<workspace>:g:projection:lane:skills`, with provider, skill version,
      source fingerprint, parser profile, schema version, and project/tenant
      metadata.
- [x] Use LLM-Wiki for semantic decomposition, cross-link candidates,
      deduplication, reparse, and stale-link maintenance only; do not delegate
      validation, capability binding, approval, or execution authority.
- [x] Define a project-plugin package shape that may contribute glossary source
      material, project skills, policies, and bounded tool adapters.
- [x] Ingest company terms, abbreviations, aliases, definitions, sources, and
      validity metadata into the project/tenant knowledge namespace rather than
      copying them into skill bodies.
- [x] Add a bounded project glossary context source using exact match, alias
      expansion, and optional semantic retrieval.
- [x] Preserve core importability and behavior when optional dependencies are
      absent.
- [x] Prevent plugins from replacing authority checks or registering hidden
      side effects without declared capabilities.

Tests:

- [x] Missing optional plugin leaves core operational.
- [x] Plugin load/unload cleans resources.
- [x] Plugin reload cannot leave stale catalog graph, lexical, or semantic
      entries from the prior source fingerprint.
- [x] Plugin failure is isolated according to declared failure mode.
- [x] LLM-Wiki adapter receives bounded authorized requests and returns stable
      references.
- [x] LLM-Wiki absence or outage does not disable raw skills, deterministic
      descriptors, or a prior valid digested projection.
- [x] LLM-Wiki inferred edges retain confidence/source/parser provenance and
      remain candidates until accepted by the relevant policy.
- [x] Plugin uninstall removes provider-owned active catalog/projection records
      without deleting historical run, feedback, memory, or wisdom evidence.
- [x] Project glossary retrieval honors tenant/project ACL and injects only
      query-relevant terms.
- [x] Project procedure skills may reference glossary entity IDs without
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

- [x] Extend capability descriptors enough to generate a versioned Agent Card.
- [x] Map A2A message/task lifecycle to root workflow submission, inspection,
      cancellation, suspension, and result artifacts.
- [x] Persist external context/task identifiers separately from run and trace
      identity.
- [x] Support polling first, then streaming using existing ordered run events.
- [x] Push callbacks use an injected durable delivery queue with bounded retry,
      callback authentication, SSRF-safe host allowlisting, idempotent delivery
      IDs, and delivery audit; the adapter performs no network I/O.
- [x] Export results and evidence references, never private reasoning traces.

Tests:

- [x] Agent Card schema and security declaration.
- [x] Submit, inspect, stream, cancel, and input-required/resume mapping.
- [x] Identifier non-conflation tests.
- [x] Authorization and tenant isolation.
- [x] Reconnect and duplicate-request idempotency.

Exit criterion:

- One external A2A client can execute and inspect a bounded Kogwistar workflow
  without becoming workflow authority.

## Phase 10: Optional CrewAI Importer

Suggested package:

```text
kogwistar/interop/crewai/
```

Deliverables:

- [x] Translate supported static Flow/task/dependency structures into
      `WorkflowDesignArtifact`.
- [x] Translate role agents to agent profiles or resolver bindings.
- [x] Translate delegation to ordinary nested invocation.
- [x] Emit explicit diagnostics for dynamic manager behavior, callbacks, memory,
      or guardrails that cannot be represented safely.
- [x] Never execute CrewAI as a second embedded authority.

Tests:

- [x] Deterministic supported import fixture.
- [x] Round-trip semantic comparison for the supported subset.
- [x] Explicit rejection of unsupported or lossy constructs.

Exit criterion:

- Supported CrewAI descriptions become ordinary inspectable Kogwistar workflow
  designs; unsupported semantics never silently degrade.

## Cross-Cutting Verification

- [x] Unit tests carry appropriate `unit`, `ci`, and domain markers.
- [x] Deterministic end-to-end tests use fake model/tool payloads.
- [x] Optional live-provider tests are separately marked and excluded from CI.
- [x] No test enables third-party remote telemetry by default.
- [x] Resource-owning fixtures close engines, workers, exporters, and providers.
- [x] Sync/async parity is claimed only after equivalent agent lifecycle
      contract tests pass, in addition to the existing runtime bijection suite.
- [x] Python and applicable native parity tests pass after any permitted native
      bug fix; Rust coverage applies to the shared lane/store bridge because no
      Rust agent execution authority exists.
- [x] Documentation links every exposed capability to contract tests.

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

- [x] Removing advisory agent/plan/goal labels does not change runtime
      correctness.
- [x] Every run remains an ordinary workflow run with ordinary steps,
      checkpoints, trace, budget, ACL, and provenance.
- [x] No plugin, profile, prompt, or role can grant itself authority.
- [x] Full conversation history survives compression.
- [x] Steering and queued input are durable, ordered, authorized, and applied
      only at documented safe points.
- [x] Cross-conversation memory is explicit and ACL-filtered.
- [x] Default wisdom retrieval serves approved lessons only.
- [x] Skills are progressively disclosed and never replace project knowledge
      authority.
- [x] Raw skill packages remain attributable source artifacts; parsed skill
      graphs remain rebuildable discovery projections and never grant authority.
- [x] Skill ingestion is additive: provider-native use remains functional while
      ordinary workflows may also use bounded graph-digested procedures.
- [x] Company terminology is evidence-backed project knowledge with ACL scope.
- [x] Subagents are bounded child runs, not hidden threads or schedulers.
- [x] Optional integrations are absent-safe and cannot become core truth.
- [x] Agent behavior is explainable through existing graph and workflow
      diagnostics.
