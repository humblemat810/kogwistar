# ARD — Hypergraph Knowledge Navigator Agent (Visibility-Aware, Runtime-Native)

Status: Draft v3 (Reviewed with Visibility, ACL, and revocation semantics)
Companion checklist: `docs/hypergraph_navigator_implementation_checklist.md`

---

## 1. Purpose

Design a **Hypergraph Knowledge Navigator Agent** that:

- Uses Kogwistar runtime semantics (workflow, lane messaging, service)
- Is **visibility-aware (Sharing & Access / ACL) by design**
- Supports multi-user shared knowledge safely
- Produces explainable, provenance-backed answers

---

## 2. Core Principle

> Navigation MUST operate on a visibility-filtered graph, not raw graph.

No post-filtering.

---

## 3. System Positioning

This is NOT:
- a simple RAG agent
- a stateless query layer
- an owner-only graph navigator

This IS:
- a workflow-native agent
- a graph traversal + reasoning system
- a visibility-scoped knowledge interpreter
- a cross-graph navigator over ACL-visible truth

---

## 4. Architecture Overview

### 4.1 Foreground (User Query Path)

Workflow-run based:

submit_turn_for_answer → wf.hypergraph_navigator

---

### 4.2 Background (Async Enrichment)

Via lane messaging + services:
- subgraph summarization
- alias expansion
- relationship enrichment
- recency indexing
- bounded re-derive / repair after ACL change

---

### 4.3 Service Layer

- graph_change trigger → refresh summaries
- message arrival → process enrichment
- schedule → rebuild indexes
- ACL / sharing change trigger → mark affected derived artifacts stale and enqueue bounded repair

---

## 5. Workflow Design

### wf.hypergraph_navigator.v1

1. InterpretTurn  
2. ResolveSecurityContext  
3. ResolveVisibility  
4. ExpandAnchors  
5. TraverseHypergraph  
6. Score & Rank  
7. AssembleEvidence  
8. Answer or Escalate  
9. Emit Run Summary  

---

## 6. Visibility / ACL Model (UX: "Sharing & Access")

Users see:

- Who can see this  
- Who can edit this  
- Who can manage sharing  

Hidden = non-existent

The navigator MUST operate on the set of graph objects visible to the principal in the current security context.

Visibility is not limited to ownership.

Allowed visibility may come from:
- owner
- explicit sharing
- group / team / role membership
- security scope
- public visibility

---

## 7. Navigator ACL Contract

### 7.1 ResolveVisibility Contract

ResolveVisibility MUST produce a visibility context for the run, including:

- principal
- groups / roles
- security scope
- allowed graph families
- ACL / policy version context when available

The navigator MUST use ACL-aware reads only.

No raw traversal followed by filtering.

---

### 7.2 Traversal Contract

A traversal hop is allowed only if:

- the edge itself is visible
- the source endpoint is visible
- the target endpoint is visible

No partially hidden topology may be exposed.

An invisible endpoint makes the hop unresolved / denied for the current principal.

---

### 7.3 Cross-Graph Contract

Cross-graph navigation is allowed and expected.

Typical paths may cross:
- conversation graph
- knowledge graph
- workflow graph
- derived artifacts / summaries

Each hop MUST independently satisfy the ACL rules of that graph family.

Visibility at the entry graph MUST NOT imply visibility of the target graph.

---

### 7.4 Pointer Dereference Contract

Pointers are not grants.

A pointer may be visible while its target is no longer visible.

Dereference is allowed only if:
- pointer is visible
- target is currently visible to the principal

If target visibility fails, the dereference result is denied / unresolved.

---

### 7.5 Node Read Contract

A node may be used for answer generation only if:

- node ACL passes
- required grounding / provenance references are visible
- required usage spans are visible

If any required hidden source would influence the answer, the node is not eligible for answer assembly for that principal.

---

### 7.6 Ranking and Counting Contract

Ranking, path scoring, counts, summaries, and evidence selection MUST be computed only over the ACL-visible set.

No hidden object may influence:
- ranking
- confidence
- path choice
- counts
- summary wording

---

## 8. Derived Artifact Rules

Derived artifacts inherit strictest source visibility.

This applies to recursive source closure, not only immediate parents.

A derived artifact should carry an ACL envelope / folded record sufficient to decide safe reuse, such as:
- immediate source ids
- source closure digest
- strictest source visibility
- source ACL / policy versions when available
- derivation type
- sanitization flag / proof reference when applicable

A derived artifact without sufficient source / ACL lineage is unresolved and must not be used for answering or promotion.

---

## 9. Live Pointer vs Snapshot Evidence

The system distinguishes:

### 9.1 Pointer / Reference

A pointer records that a conversation or run referenced another object.

A pointer is live.

Dereference always uses current ACL.

### 9.2 Evidence Snapshot

An evidence snapshot records what the agent actually used during a run.

A snapshot is append-only and audit-oriented.

The system should not copy the full KG into the conversation graph.

Instead, the conversation graph should store:
- run anchors
- pointers / references
- selected evidence snapshots
- answer artifacts / summaries

The KG remains truth; the conversation graph stores working traces and derived artifacts.

---

## 10. Revocation / ACL Change Semantics

ACL changes affect future reads immediately.

If a referenced source becomes hidden later:
- the pointer may remain
- dereference must re-check current ACL
- the target may become denied / unresolved

Derived artifacts whose recursive source closure includes the changed source become stale / tainted until revalidated or re-derived.

Default conservative policy:
- stale derived artifacts are not eligible for user-facing answer generation
- stale artifacts may remain for audit / replay administration
- repair is bounded to affected dependency closure, not whole-graph recompute

This requires reverse provenance / dependency indexing sufficient to find affected downstream artifacts.

---

## 11. Lane Messaging Contract

Must include subject + visibility context.

At minimum, lane messages for navigator background work should carry:
- principal / subject
- groups / roles if relevant
- security scope
- allowed graph family context when relevant
- ACL / policy version context when available
- source ids / provenance anchors when deriving new artifacts

Prefetching or caching may improve latency, but cache warmth is not authorization.

---

## 12. Answer Format

Structured:
- answer
- evidence
- paths
- confidence
- uncertainty

Evidence and paths must be assembled only from ACL-visible material.

---

## 13. Invariants

- no hidden influence  
- visibility-first traversal  
- safe derivation  
- rebuildable projections  
- cross-graph visibility checked hop-by-hop  
- pointers are live, evidence is snapshot  
- hidden topology is non-existent  
- stale derived artifacts are not user-facing until repaired or revalidated  

---

## 14. First Slice

- navigator workflow  
- visibility filter  
- one background service  
- pointer dereference guarded by current ACL  
- evidence snapshot emitted for used context  
- minimal reverse provenance index for bounded stale marking  

---

## 15. Key Insight

This is a **visibility-scoped hypergraph reasoning system**, not RAG with permissions.

It is also not an owner-only navigator.

It is a cross-graph, ACL-aware navigator over visible truth, with provenance-backed snapshots and revocation-safe derivation semantics.
