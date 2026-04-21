# Hypergraph Navigator Implementation Checklist

Companion ARD: `docs/hypergraph_navigator_ard_v2.md`

Status: Draft checklist. Contract skeleton exists; production navigator wiring is not complete.

---

## Phase 0 - Contract Pinning

- [x] document navigator is ACL-visible, cross-graph, and not owner-only
- [x] document live pointer versus evidence snapshot semantics
- [x] document recursive source-closure requirements for derived artifacts
- [x] document revocation stale / bounded repair semantics
- [x] add fake-backed contract tests for core ACL invariants

Current contract tests:

- `tests/graph_navigator/test_hypergraph_navigator.py::test_traversal_requires_visible_edge_and_both_endpoints`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_pointer_visibility_does_not_grant_target_visibility`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_pointer_dereference_uses_current_acl_not_historic_visibility`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_node_cannot_answer_if_required_grounding_or_span_is_hidden`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_ranking_pool_must_be_acl_visible_only`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_derived_artifact_requires_recursive_source_visibility_by_default`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_sanitized_artifact_may_be_reused_without_full_source_visibility`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_sanitizer_flag_without_visible_proof_does_not_bypass_source_acl`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_stale_derived_artifact_is_not_user_facing_until_repaired`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_acl_change_marks_bounded_downstream_dependents_not_whole_graph`
- `tests/graph_navigator/test_hypergraph_navigator.py::test_conversation_graph_stores_pointer_or_snapshot_not_full_truth_copy`

---

## Phase 1 - Real ACL Integration Tests

- [ ] replace fake visibility oracle with real `ACLAwareReadSubsystem` test fixture
- [ ] test traversal denies edge when either endpoint is hidden in persisted ACL graph
- [ ] test pointer node visible but KG target hidden returns unresolved, not target content
- [ ] test pointer dereference re-checks current ACL after ACL supersession / tombstone
- [ ] test node answer eligibility requires node ACL, grounding ACL, and span-usage ACL
- [ ] test missing ACL at node, grounding, or span level denies user-facing read
- [ ] test ranking pool is built from ACL-aware reads, not raw reads plus post-filter
- [ ] test conversation graph stores pointer / evidence snapshot, not full KG node copy

Done means tests use real engine-backed ACL records, not only fake `visible_ids`.

---

## Phase 2 - Navigator Read Surface

- [ ] define `HypergraphNavigator` service/module boundary
- [ ] define `VisibilityContext` input shape: principal, groups/roles, security scope, graph family
- [ ] ensure navigator accepts only ACL-aware engine views for user-facing traversal
- [ ] keep raw read available only for internal repair/operator code paths
- [ ] implement pointer dereference helper that checks pointer ACL and target ACL
- [ ] implement hop expansion helper that checks edge ACL plus both endpoint ACLs
- [ ] implement answer candidate selection over visible nodes/edges only
- [ ] expose unresolved/denied target as structured uncertainty, not hidden detail

---

## Phase 3 - Evidence Snapshot Boundary

- [ ] define minimal evidence snapshot schema for navigator runs
- [ ] persist run anchor, pointer nodes, selected evidence snapshots, and answer artifact
- [ ] record source ids and source ACL / policy versions for selected evidence
- [ ] ensure snapshots are append-only and audit-oriented
- [ ] ensure replay display checks snapshot/artifact ACL before user-facing output
- [ ] test old pointer remains while target dereference can become denied

---

## Phase 4 - Derived Artifact ACL Envelope

- [ ] define folded ACL envelope fields for derived navigator artifacts
- [ ] store immediate source ids
- [ ] store recursive source closure digest
- [ ] store strictest source visibility or equivalent effective ACL mode
- [ ] store source ACL / policy versions when available
- [ ] store derivation type
- [ ] store sanitizer proof reference when sanitization is used
- [ ] deny artifact use when envelope is missing or unresolved
- [ ] deny promotion unless full source closure allows target visibility or sanitizer proof passes

---

## Phase 5 - Revocation and Bounded Repair

- [ ] maintain reverse provenance / dependency index from source usage to derived artifacts
- [ ] on ACL change, mark affected downstream closure stale / tainted
- [ ] do bounded stale propagation, not whole-graph recompute
- [ ] skip stale artifacts in answer, ranking, path scoring, and wisdom recommendations
- [ ] enqueue bounded rederive / repair jobs for affected artifacts
- [ ] produce new artifact versions instead of mutating old history
- [ ] test source span revoke stales derived KG node and downstream wisdom artifact
- [ ] test unrelated artifacts are not marked stale

---

## Phase 6 - Cross-Graph Navigation

- [ ] support conversation -> KG pointer dereference with ACL checks on both graphs
- [ ] support KG -> derived artifact traversal with artifact ACL envelope checks
- [ ] support workflow/trace references without leaking hidden trace details
- [ ] test each hop independently enforces its graph family's ACL
- [ ] test entry graph visibility does not imply target graph visibility

---

## Phase 7 - Background and Lane Messaging

- [ ] navigator background messages carry principal / subject
- [ ] messages carry groups/roles when relevant
- [ ] messages carry security scope
- [ ] messages carry source ids / provenance anchors
- [ ] derived background outputs inherit strictest source ACL
- [ ] background maintenance jobs may read raw truth but cannot emit user-facing artifacts without ACL envelope
- [ ] test background summary from mixed sources is hidden from unauthorized principal

---

## Acceptance

- [ ] no user-facing navigator path uses raw traversal then post-filter
- [ ] hidden nodes, edges, spans, counts, and paths cannot influence answers
- [ ] pointers are live and never grants
- [ ] evidence snapshots are append-only derived artifacts with ACL
- [ ] derived artifacts require recursive source-closure safety
- [ ] revocation stales downstream derived artifacts through bounded dependency closure
- [ ] stale artifacts are excluded from answer/ranking/wisdom until repaired or revalidated
- [ ] cross-graph navigation checks ACL at every hop
- [ ] implementation tests pass on fake/in-memory backend
- [ ] smoke tests cover Chroma and PostgreSQL-backed ACL graph where available
