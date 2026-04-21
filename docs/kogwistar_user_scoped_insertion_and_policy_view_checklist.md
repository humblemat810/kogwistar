# Kogwistar Core - User-Scoped Insertion and ACL-Graph View Checklist

Status: In progress. Phase 1 complete; Phase 2 complete; later phases pending.
Audience: Local coding agents and human maintainers
Companion: `docs/kogwistar_user_scoped_insertion_and_policy_view_ard.md`

---

## 0. Purpose

This checklist is execution companion for user-scoped insertion and ACL-graph view ARD.

Use it to track slice progress, keep design boundaries honest, and prevent partial visibility work from shipping as if complete.

---

## 1. Glossary

- `principal`: active subject whose rights are being evaluated. May be user, agent, service account, workflow actor, or any authenticated or authorized identity.
- `truth graph`: any domain graph that stores domain facts, such as knowledge, conversation, or workflow.
- `ACL graph`: versioned overlay graph that stores visibility, sharing, and access facts for any truth graph.
- `ACL record`: one versioned ACL fact set for one target truth node.
- `effective visibility`: result of joining principal context with ACL truth.
- `ACL node`: a versioned overlay object that refers to one truth target at a chosen grain such as document, grounding, span, node, edge, or artifact.
- `target_item_id`: usage-level subkey inside a grain, used to pin one specific grounding usage, span usage, edge usage, or artifact usage rather than a generic shared bucket.
- `ACLNode`: one persisted ACL state node for one target usage.
- `ACLEdge`: persisted ACL relationship edge, such as superseding an older ACL state.

ACL node types are not all the same:

- `document ACL`: gates a source document or document-like input before parsing or ingestion.
- `grounding ACL`: gates one node-specific or edge-specific grounding usage before its spans can support visible truth.
- `span ACL`: gates a specific evidence span or excerpt.
- `span ACL` is usage-level: same span text may appear under multiple nodes, and each node usage may carry its own ACL record.
- `node ACL`: gates a truth node produced from one or more sources.
- `edge ACL`: gates a relationship or provenance edge.
- `artifact ACL`: gates a derived summary, memory, digest, or other computed object.

Public knowledge fast path:

- user-facing node read must not treat missing ACL as allow
- if ACL record exists and says public, navigation may skip expensive group/share joins for that checked level
- if a source family wants default public behavior, it should materialize explicit public ACL records or define a family policy that is resolved before user-facing read
- if any required ACL record is missing for a non-public or unresolved family, deny or return unresolved instead of exposing
- explicit-public entities should remain cheaply navigable through a short-circuit decision path

Compact mental model:

```text
one target usage -> one ACLNode
one ACL version link -> one ACLEdge
```

Do not collapse node ACL plus grounding ACL plus all span ACL into one giant ACL blob.

---

## 1. Must Stay True

- knowledge truth and ACL truth stay separate
- ACL facts live in ACL graph or ACL event truth, not only serving projections
- ACL graph overlays knowledge, conversation, workflow, and any truth graph
- ACL graph must be engine-native and persisted through graph engine write/read paths, not only held in process memory
- ACL state is versioned; new ACL nodes or edges supersede old state instead of mutating truth inline
- execution namespace, security scope, and visibility policy remain distinct
- ACL nodes may sit at document, grounding, span, node, edge, or artifact grain
- explicit-public knowledge should keep a short-circuit path after ACL/family-policy resolution
- user-facing reads filter first, answer later
- hidden nodes and edges do not influence ranking, counts, confidence, or summaries
- derived artifacts inherit safe ACL from source set
- background jobs that can emit user-visible output carry principal and visibility context
- effective visibility projections are disposable and rebuildable

---

## 2. First Slice Goal

Ship safe minimum contract for:

- user-scoped insertion facts
- effective visibility filtering on read paths
- strict derived-artifact ACL inheritance
- background propagation of principal and visibility context

---

## 3. Execution Phases

### Phase 1 - Authoritative insertion facts

- [x] define ACL grain mapping for document, grounding, span, node, edge, and artifact
- [x] define usage-level `target_item_id` for grounding/span ACL and node/edge reuse
- [x] audit insertion entrypoints that create visibility-relevant nodes and edges
- [x] implement persisted ACL node schema and ACL edge/version-link schema
- [x] ensure insertion captures `created_by`
- [x] ensure insertion captures `owner_id`
- [x] ensure knowledge insert does not mutate ACL truth inline
- [x] ensure ACL insert captures `security_scope`
- [x] ensure ACL insert references target truth node
- [x] ensure ACL insert assigns explicit visibility or share policy fact
- [x] ensure ACL mutation creates superseding ACL state instead of mutating old truth inline
- [x] define default visibility mode for inserts that omit explicit policy, materialized as ACL truth rather than inferred from missing ACL
- [x] wire `record_acl(...)` to persisted engine graph writes instead of in-memory-only helper state
- [x] emit auditable ACL-related events on insert and policy mutation
- [x] implement graph-native lookup for `target_item_id -> truth target usages`

### Phase 1 Note

Phase 1 is complete.
Canonical ACL writes land in persisted ACL record nodes and supersession edges.
ACL grain mapping is defined in ARD glossary and design principles, and the `target_item_id` usage model is in active use.

Query/read rule:

```text
usable node span evidence
  = pass node ACL
  and pass relevant grounding ACL for that usage
  and pass that node-specific span-usage ACL
```

```text
read node content
  = pass node ACL
  and pass all underlying grounding ACLs for that node version
  and pass all underlying span-usage ACLs for that node version
  and deny/unresolved if any required ACL record is missing
```

Lookup rule:

```text
known truth_graph + entity_id
  -> lookup node ACL
  -> expand underlying grounding and span usages if node content read
  -> lookup grounding ACLs by (truth_graph, entity_id, target_item_id)
  -> lookup span-usage ACLs by (truth_graph, entity_id, target_item_id)
```

### Phase 2 - Read-path enforcement

- [x] define effective ACL context contract
- [x] identify all user-facing retrieval, traversal, and query entrypoints
- [x] route normal engine reads through ACL-aware subsystem when `acl_enabled=True`
- [x] route normal engine writes through ACL-aware subsystem when `acl_enabled=True`
- [x] route all answer-facing graph reads through ACL-filtered entrypoints
- [x] prevent raw graph traversal in answer and navigator flows
- [x] ensure ranking does not use hidden nodes or edges
- [x] ensure explanations do not cite hidden structure
- [x] pin navigator workflow order: interpret, security context, ACL context, anchors, traverse, rank, evidence, answer
- [x] add explicit-public ACL short-circuit path for cheap navigation after required ACL levels are resolved
- [x] add ACL-aware user-facing node read entrypoint that requires principal, security scope, node id, grounding usage ids, and span usage ids
- [x] keep raw graph read/write available as `raw_read` and `raw_write` for internal, operator, rebuild, or policy-resolution paths
- [x] define query/read contract that evidence use must pass node ACL, relevant grounding ACL, and node-specific span-usage ACL
- [x] define partial-recording rule: missing node, grounding, or span ACL cannot authorize user-facing node read

### Phase 2 Note

Phase 2 now has ACL-aware normal read/write when `GraphKnowledgeEngine(..., acl_enabled=True)`.
In that mode, normal `read` filters through persisted ACL truth and normal `write` materializes node, edge, grounding, and span ACL records.
Raw `raw_read` and `raw_write` remain available for internal, operator, rebuild, and policy-resolution paths.
Answer-facing candidate retrieval, evidence pack materialization, and KG projection now route through ACL-aware read surfaces.
Full navigator traversal enforcement now routes through ACL-aware reads on acl-enabled engines.
Legacy raw fallbacks remain only for non-ACL engines and internal repair paths.

Phase 4 now has a first envelope slice: lane messages carry an ACL context blob in truth-node metadata, preserving principal, scope, purpose, source graph, and visibility intent across background propagation.

### Phase 3 - Derived artifact safety

- [x] define minimum ACL record fields for derived artifacts: source ids, derivation type, ACL mode
- [x] implement strictest-source ACL inheritance rule across truth graphs
- [x] prevent unsafe reuse of artifact derived from mixed-visibility sources
- [x] test summary, digest, embedding, and path-style derived artifacts where applicable

### Phase 3 Note

Current code already has a working first-pass persisted ACL path and tests, but Phase 3 is still partial because full truth-graph traversal, retrieval path enforcement, and background propagation are not yet wired to persisted ACL graph truth everywhere.
Derived-artifact ACL data should live in ACL truth as versioned ACL records, not as authority inside truth-node metadata.
- [x] engine owns a first graph-native persisted ACL record path
- [x] engine exposes ACL record write and decide helper entrypoints

### Phase 4 - Background propagation

- [x] add principal, security scope, and ACL context to lane-message payload or envelope
- [x] ensure worker and service handlers preserve that context
- [x] ensure background-created user-visible artifacts inherit correct ACL
- [ ] keep global maintenance flows separate from user-visible derivation flows where needed

### Phase 5 - Projections, repair, and operability

- [ ] define effective ACL projection shape or index family
- [ ] ensure projection rebuild from authoritative truth
- [ ] add repair path for stale or dropped ACL projection rows
- [ ] expose operator-safe inspection surfaces for ACL and effective visibility state
- [ ] document operator-only versus user-facing ACL surfaces

---

## 4. Acceptance Checklist

### Insertion

- [ ] inserting as user A records creator, owner, and knowledge truth facts
- [x] inserting as user A records ACL facts in persisted ACL graph
- [ ] insert in scope A does not silently expose entity in scope B
- [x] ACL mutation is auditable

### Read and view

- [ ] user A sees visible entity set A
- [ ] user B sees visible entity set B
- [ ] hidden nodes do not appear in counts, topology, or path summaries
- [ ] filtered traversal never ranks hidden artifacts
- [ ] answer assembly exposes answer, evidence, paths, confidence, and uncertainty only from visible sources
- [x] node read can require node ACL plus all grounding and span-usage ACLs to pass
- [x] node-specific span usage can require node ACL, grounding ACL, and span-usage ACL to pass
- [x] partial ACL recording denies user-facing node read when node, grounding, or span ACL is missing
- [x] ACL-enabled engine normal read hides unauthorized nodes while raw read remains available for internal use

### Derived artifacts

- [ ] derived artifact from mixed-visible sources is not exposed to unauthorized user
- [ ] derived artifact records source linkage and ACL mode

### Background work

- [ ] lane message carries principal, scope, and ACL context
- [x] background-generated artifact inherits correct ACL semantics

### Rebuild and repair

- [ ] effective visibility projection can rebuild from authoritative truth
- [ ] ACL changes recompute effective visibility correctly
- [ ] repair path does not mutate authoritative history incorrectly

---

## 5. Docs Sync

- [x] create this checklist
- [x] keep ARD and checklist aligned if semantics move
- [ ] update related operator or visibility docs when new surfaces appear
- [ ] update status or roadmap docs once first slice stabilizes

---

## 6. Notes

- If implementation pressure pushes semantics into projection-only shortcuts, stop and fix design first.
- If implementation pressure pushes ACL semantics into in-memory-only helper objects, stop and rewire into graph-native persisted truth.
- If read filtering happens after traversal or ranking, treat slice as incomplete.
- If background jobs lose principal context, treat user-visible outputs as unsafe by default.

