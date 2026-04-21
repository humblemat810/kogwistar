# ARD — Hypergraph Knowledge Navigator Agent (Visibility-Aware, Runtime-Native)

Status: Draft v2 (Reviewed with Visibility & ACL considerations)

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

This IS:
- a workflow-native agent
- a graph traversal + reasoning system
- a visibility-scoped knowledge interpreter

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

---

### 4.3 Service Layer

- graph_change trigger → refresh summaries
- message arrival → process enrichment
- schedule → rebuild indexes

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

---

## 7. Derived Artifact Rules

Derived artifacts inherit strictest source visibility.

---

## 8. Lane Messaging Contract

Must include subject + visibility context.

---

## 9. Answer Format

Structured:
- answer
- evidence
- paths
- confidence
- uncertainty

---

## 10. Invariants

- no hidden influence  
- visibility-first traversal  
- safe derivation  
- rebuildable projections  

---

## 11. First Slice

- navigator workflow  
- visibility filter  
- one background service  

---

## 12. Key Insight

This is a **visibility-scoped hypergraph reasoning system**, not RAG with permissions.
