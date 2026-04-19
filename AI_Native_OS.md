# Executable Plan: Kogwistar → Production-Ready + AI-Native Operating System

> Two-track plan. Track A (Polish) can ship incrementally in weeks.
> Track B (AI OS) is a multi-month architectural build, ordered by dependency.
> Each item has: goal, concrete deliverables, acceptance criteria, and estimated effort.

---

## Track A — Polish to Production-Ready

*Goal: make the ecosystem adoptable by developers who have not read every design doc.*

---

### A1 — Publish kogwistar to PyPI

**Why first:** Every other repo bootstraps from it. PyPI presence eliminates the GitHub-install dependency in all downstream packages.

**Deliverables:**
- pyproject.toml version pinned to 1.0.0 or 0.x stable
- GitHub Actions workflow: release.yml that runs on tag push, builds wheel, publishes to PyPI via Trusted Publisher (OIDC, no stored secrets)
- CHANGELOG.md with first entry capturing current state
- Matching releases for kogwistar-obsidian-sink and (once API stable) kg-doc-parser

**Acceptance criteria:**
- pip install kogwistar works on a clean Python 3.11 venv
- pip install kogwistar-obsidian-sink pulls the correct version of kogwistar as a dep
- kogwistar-llm-wiki's pyproject.toml can declare kogwistar>=1.0.0 as a real dependency

**Effort:** ~1 day per repo (3 repos total: ~3 days)

---

### A2 — Cloistar: Close the Phase 5 Test Debt

**Why:** The governance layer is functionally complete. Tests are the documentation of that claim.

**Deliverables:**
- tests/unit/test_governance_plugin.py — plugin hook unit test: before_tool_call returns allow/block/requireApproval correctly
- tests/integration/test_bridge_endpoints.py — bridge endpoint tests for /policy/before-tool-call, /events/after-tool-call, /approval/resolution
- tests/e2e/test_governance_e2e.py — one smoke test: OpenClaw hook → bridge → decision → back

**Acceptance criteria:**
- pytest tests/ passes on a clean local install
- allow, block, requireApproval are each covered by at least one test case
- Approval resume flow has at least one test that goes full cycle

**Effort:** ~2–3 days

---

### A3 — Unified Bootstrap + One-Command Setup

**Why:** Three repos have three different bootstrap scripts. Someone starting fresh must run three commands in the right order.

**Deliverables:**
- scripts/setup-workspace.sh at the root of a proposed kogwistar-workspace meta-repo (or at the top of any of the main repos) that:
  1. Checks Python version (>=3.11)
  2. Creates .venv
  3. Calls each repo's bootstrap-dev.sh in dependency order
  4. Runs smoke tests to verify the install
- A Makefile or justfile with targets: setup, test, daemon-start, daemon-stop
- Windows PowerShell equivalent: scripts/setup-workspace.ps1

**Acceptance criteria:**
- A developer with only git and Python installed can go from zero to a running llm-wiki daemon maintenance in one command
- Works on macOS, Linux, Windows (Git Bash / PowerShell)

**Effort:** ~1–2 days

---

### A4 — Init System / Supervisor for Daemons

**Why:** The two daemons are started manually. On a restart they are gone.

**Deliverables:**
- scripts/kogwistar-llm-wiki.service — systemd unit file for the daemon pair (Linux)
- scripts/cogwistar-llm-wiki.plist — launchd plist (macOS)
- scripts/start-daemons.ps1 — PowerShell script that starts both daemons as background jobs with restart policy (Windows)
- A KernelSupervisor class in daemon.py that wraps both daemons, monitors their health via threading.Event heartbeats, and restarts them if they stop unexpectedly

**Acceptance criteria:**
- On Linux: systemctl start kogwistar-llm-wiki starts both daemons; systemctl status shows health
- On any OS: if MaintenanceDaemon crashes, it restarts within 5 seconds
- Graceful shutdown drains in-flight work before exiting

**Effort:** ~2 days

---

### A5 — Metadata Key Stabilisation Across the API

**Why:** The metadata.X vs bare X inconsistency in where clause queries causes bugs (already fixed in kogwistar-llm-wiki but may exist elsewhere).

**Deliverables:**
- Audit of all get_nodes(where={...}) calls across all repos — grep for "metadata.X" patterns
- A WhereClause typed helper or linter rule that enforces bare keys
- Release notes documenting the canon: bare keys, not metadata.X

**Acceptance criteria:**
- grep -r '"metadata\.' src/ tests/ returns zero results in all repos
- New tests use the canonical form

**Effort:** ~1 day

---

### A6 — kg-doc-parser: Complete the Refactor

**Why:** The README explicitly says "being extracted and consolidated into the main kogwistar repository — treat as staging area."

**Deliverables:**
- Decision: keep as standalone CLI or merge into kogwistar[ingestion] optional extra
- If standalone: stabilise the public API (run_ocr_source_workflow, run_layerwise_source_workflow), write a stable __init__.py export
- If merged: PR into kogwistar with the workflow-ingest code under kogwistar/ingest/
- Remove "work in progress" disclaimer from README once the above is done

**Effort:** ~3–5 days depending on decision

---

## Track B — AI-Native Operating System

*Build order is determined by dependency. Each layer depends on the one above it.*

---

### B1 — Token / Resource Budget Accounting

**Dependencies:** None (standalone concern)

**What:** Every workflow run has a declared budget. Every LLM step debits from it. The run is interrupted (with a RunSuspended outcome) if the budget is exhausted.

**Deliverables:**
- kogwistar/runtime/budget.py
  - BudgetLedger — per-run token counter, persisted as a graph node in the workflow engine
  - BudgetExhaustedError — raised when a step tries to call LLM with zero budget
- Integration into WorkflowRuntime.run():
  - Accept token_budget: int | None in run()
  - Pass BudgetLedger via _deps
  - Step resolvers call ctx.deps.budget.debit(tokens_used) after each LLM call
- kogwistar/runtime/cost_ledger.py — workspace-level cost accumulator
  - Writes a budget_event node per debit (append-only)
  - Queryable: engine.read.get_nodes(where={"artifact_kind": "budget_event", "workspace_id": ws})
- Governance integration: the cloistar bridge checks BudgetLedger.remaining before returning allow — returns block with reason "budget_exhausted" if empty

**Acceptance criteria:**
- WorkflowRuntime.run(workflow_id=..., token_budget=1000) raises or suspends when 1000 tokens are consumed
- budget_event nodes are queryable after a run
- A test proves a budget-exhausted run does not proceed

**Effort:** ~3 days

---

### B2 — Graph-Native Message Bus on CDC Oplog

**Dependencies:** B1 (budget debits need a bus to propagate alerts)

**What:** Turn the CDC oplog into a real message bus that any agent or daemon can publish to and subscribe from, without going through HTTP.

**Deliverables:**
- kogwistar/bus/message_bus.py
  - MessageBus — thin wrapper around the CDC oplog
  - publish(topic: str, payload: dict, workspace_id: str) — writes a bus_message node to the oplog
  - subscribe(topic: str, since_seq: int, workspace_id: str) — yields new bus_message nodes via a polling generator
  - Topics are just strings (e.g., "distillation.completed", "projection.requested", "budget.alert")
- Dead-letter namespace: messages that fail delivery N times are moved to conv:dead_letter
- Integration: MaintenanceDaemon and ProjectionDaemon publish daemon.heartbeat messages every poll cycle
- Integration: ProjectionWorker subscribes to promotion.completed instead of polling the queue

**Acceptance criteria:**
- Publishing a message then subscribing with since_seq=publish_seq returns the message
- Heartbeat messages appear in the graph at expected intervals
- Dead-letter test: force a consume error N times → verify node appears in dead_letter namespace

**Effort:** ~4 days

---

### B3 — Agent Identity Nodes

**Dependencies:** B2 (agents announce themselves via the bus)

**What:** Every running agent has a persistent identity node in the graph. Its history, capabilities, and trust level are queryable.

**Deliverables:**
- kogwistar/agents/identity.py
  - AgentNode — a first-class graph node type (artifact_kind: "agent")
  - Fields: agent_id, display_name, capabilities: list[str], trust_level: float, first_seen_ts, last_seen_ts
  - AgentIdentityManager — creates/updates agent nodes, uses stable_id("agent", agent_id) for stable addressing
- kogwistar/agents/session.py
  - AgentSession — a context manager that creates an identity node on enter, updates last_seen_ts on exit
  - Publishes agent.started and agent.stopped bus messages
- Integration: MaintenanceDaemon and ProjectionDaemon both create AgentSession on startup
- Trust evolution (initial): trust level increases 0.01 per successful run, decreases 0.05 per failed run

**Acceptance criteria:**
- After llm-wiki daemon maintenance runs, an agent node for MaintenanceDaemon exists in the graph
- Agent node is queryable by agent_id
- Trust level changes are reflected in the node after runs

**Effort:** ~3 days

---

### B4 — Capability Registry

**Dependencies:** B3 (capabilities are attached to agent identity)

**What:** A graph-native registry where available tools, workflow designs, and agent capabilities are advertised as nodes and discoverable via query.

**Deliverables:**
- kogwistar/agents/capability_registry.py
  - CapabilityNode — graph node: artifact_kind: "capability", fields: capability_id, version, description, input_schema, output_schema, provider_agent_id
  - CapabilityRegistry — wraps engine read/write
  - register(capability: CapabilityNode) — adds/updates capability in the graph
  - query(filter: dict) → list[CapabilityNode] — returns matching capabilities
  - revoke(capability_id) — tombstones the capability node
- Integration: on startup, each daemon registers its capabilities (e.g., MaintenanceDaemon.register_capabilities() registers "distillation", "history_wisdom")
- Integration: governance (cloistar) checks the registry before granting a tool call — if the capability is revoked, the decision is block

**Acceptance criteria:**
- CapabilityRegistry.query({"provider_agent_id": "maintenance-daemon"}) returns the registered capabilities
- Revoking a capability makes it invisible to subsequent queries
- A governance test: revoked capability → block decision

**Effort:** ~3 days

---

### B5 — Real-Time Perception / Environment Sensors

**Dependencies:** B2 (sensors publish to the bus)

**What:** The system reacts to its environment without being asked. Documents that appear in a watched folder are automatically ingested.

**Deliverables:**
- kogwistar_llm_wiki/sensors/filesystem_watcher.py
  - FilesystemWatcherDaemon — wraps watchdog (cross-platform file event library)
  - On file_created or file_modified in watched directories: publishes "ingest.trigger" bus message
  - Debounces rapid changes (1-second window)
- kogwistar_llm_wiki/sensors/webhook_receiver.py
  - Minimal FastAPI app that receives POST to /event and publishes to the bus
  - Can receive events from: GitHub webhooks, email relay, calendar notifications
- kogwistar_llm_wiki/daemon.py — add FilesystemWatcherDaemon to the daemon roster
- scripts/kogwistar-llm-wiki.service (from A4) includes the watcher daemon

**Acceptance criteria:**
- Dropping a .md file into the watched folder triggers ingestion within 5 seconds (no manual CLI call)
- The ingestion event appears in the graph with correct provenance
- Rapid file writes (10 files in 1 second) produce 10 ingestion jobs, not 100

**Effort:** ~3 days

---

### B6 — Workflow Revision Proposer + Approval Gate

**Dependencies:** B3 (proposer needs agent identity), B4 (revision is a capability), B1 (proposals consume budget)

**What:** The system reads execution_wisdom nodes and proposes concrete mutations to workflow designs, gated by human approval.

**Deliverables:**
- kogwistar_llm_wiki/worker.py — add _step_propose_workflow_revision:
  - Reads execution_wisdom nodes with failure patterns
  - For each pattern: generates a WorkflowRevisionProposal node (append-only)
  - Proposal contains: target_workflow_id, proposed_change (structured dict: add step, increase retry, change threshold), evidence_run_ids, confidence
  - Does NOT apply the change — only writes the proposal
- kogwistar/runtime/revision_engine.py
  - WorkflowRevisionEngine — reads proposals, applies them to WorkflowDesignArtifact nodes after approval
  - apply_revision(proposal_id) — validates, tombstones old design node, writes revised design node
  - reject_revision(proposal_id) — tombstones the proposal
- cloistar integration: WorkflowRevisionProposal nodes trigger requireApproval in the governance layer before apply_revision is called
- Replay validator (v1): re-runs the last 10 failed executions against the proposed new design in memory and reports whether the change would have improved outcomes

**Acceptance criteria:**
- After N failures of distill step, a WorkflowRevisionProposal appears in the graph
- No revision is applied without a workflow_approval node being present first
- Applied revision produces a new versioned WorkflowDesignArtifact with backlink to old version
- Replay validator reports: "N/10 past failures would have been prevented"

**Effort:** ~5–7 days

---

### B7 — Kernel Scheduler

**Dependencies:** B1 (budget), B2 (bus), B3 (agent identity), B4 (capability registry)

**What:** A coordinator that knows what work is pending, what agents are available, and schedules runs by priority and resource budget.

**Deliverables:**
- kogwistar/kernel/scheduler.py
  - KernelScheduler — reads pending jobs from the graph + bus
  - Priority queue: "critical" > "high" > "normal" > "background"
  - Budget-aware: refuses to schedule a run if the workspace's remaining token budget is < minimum for that job type
  - Concurrency control: max N simultaneous runs per workspace (configurable)
  - Dispatches runs to registered AgentExecutor instances via the bus
- kogwistar/kernel/executor.py
  - AgentExecutor — receives dispatch messages, runs WorkflowRuntime.run(), publishes run.completed / run.failed bus messages
- Integration: MaintenanceDaemon and ProjectionDaemon become passive executors, scheduled by KernelScheduler instead of self-polling
- llm-wiki kernel start CLI command — starts the scheduler as the system's main process

**Acceptance criteria:**
- Two pending jobs with different priorities: high-priority runs first
- Budget-exhausted workspace: scheduler defers all runs until budget is refilled
- Scheduler crash: in-flight runs are resumable from checkpoint (existing kogwistar resume_run mechanism)

**Effort:** ~5–7 days

---

### B8 — Real-Time Learning Feedback Loop

**Dependencies:** B6 (revision proposer), B7 (scheduler triggers distillation after every run)

**What:** Online learning — user corrections and run outcomes feed back into the wisdom layer immediately, not batch.

**Deliverables:**
- kogwistar/learning/online_feedback.py
  - FeedbackEvent — a graph node: artifact_kind: "feedback", fields: polarity (+1/-1), target_node_id, feedback_note, submitted_by_agent_id
  - FeedbackCollector.submit(target_id, polarity, note) — writes feedback node
- Integration: kogwistar-chat adds a 👍/👎 button next to every assistant response → calls FeedbackCollector.submit
- Integration: MaintenanceDaemon runs _step_distill immediately after any feedback event is published to the bus (instead of waiting for the next poll cycle)
- Embedding drift detector (v1): every 100 new KG nodes, re-query the 10 most-referenced wisdom nodes and check if their nearest neighbours have shifted significantly. If drift detected, publish "embedding.drift.detected" bus message.

**Acceptance criteria:**
- Submitting a 👎 on a chat response creates a feedback node and triggers immediate distillation
- Positive feedback increases the confidence score of the cited knowledge node
- Drift detection runs automatically and logs drift events to the graph

**Effort:** ~4 days

---

### B9 — Unified Resource Namespace (Virtual Filesystem)

**Dependencies:** B3 (agents need addressable identity), B4 (capabilities need stable addresses)

**What:** Every addressable resource in the ecosystem gets a URI that can be resolved to a graph node or filesystem path.

**Deliverables:**
- URI scheme: kogwistar://{workspace_id}/{namespace}/{kind}/{id}
  - Examples:
    - kogwistar://demo/kg/entity/abc123
    - kogwistar://demo/workflow/run/run-xyz
    - kogwistar://demo/agent/maintenance-daemon
    - kogwistar://demo/file/obsidian/Concepts/AcmeCorp.md
- kogwistar/namespace/resolver.py
  - URIResolver.resolve(uri: str) → GraphNode | FilesystemPath | WorkflowRun
  - URIResolver.list(prefix: str) → list[str] — enumerates all URIs under a prefix
- MCP tool: kg_resolve_uri(uri) and kg_list_resources(prefix) — exposes the namespace to any MCP-connected agent
- Integration: wisdom nodes and execution_wisdom nodes add self_uri to their metadata on creation

**Acceptance criteria:**
- URIResolver.resolve("kogwistar://demo/kg/entity/abc123") returns the correct Node
- URIResolver.list("kogwistar://demo/kg/") returns all KG entities
- MCP kg_list_resources works from an agent that has never seen the codebase

**Effort:** ~3 days

---

## Summary Timeline

Week 1-2    A1 PyPI publish + A5 metadata key fix + A2 cloistar tests
Week 3      A3 unified bootstrap + A4 init/supervisor
Week 4-5    A6 kg-doc-parser decision + B1 token budget
Week 6-7    B2 message bus + B3 agent identity
Week 8      B4 capability registry + B5 filesystem watcher
Week 9-10   B6 workflow revision proposer
Week 11-12  B7 kernel scheduler
Week 13     B8 real-time learning
Week 14     B9 unified resource namespace


---

## Definition of Done — AI-Native OS

The system qualifies as a genuine AI-native OS when:

- [ ] Any document dropped in a watched folder is automatically ingested, promoted, and projected without human input (Gaps B5 + A4)
- [ ] The system's own workflow designs are revised based on failure patterns, gated by governance approval (Gap B6)
- [ ] Multiple independent agents can coordinate through the graph without shared HTTP session state (Gap B2 + B3)
- [ ] Every LLM API call is metered against a declared budget and the budget is enforceable (Gap B1)
- [ ] Any resource (node, file, run, agent) has a stable URI and is discoverable via a common query surface (Gap B9)
- [ ] A developer types one command and a fully operational system starts, supervises itself, and restarts crashed components (Gaps A3 + A4 + B7)
- [ ] User feedback on an assistant response triggers immediate re-distillation without any CLI command (Gap B8)