# LangGraph interop capability matrix

Status: conservative, one-way interop.

| Capability | Status | Boundary |
|---|---|---|
| Kogwistar design to LangGraph visual graph | Supported | Visual mode is best-effort and does not become native runtime authority. |
| Kogwistar design to LangGraph semantics graph | Supported | Blob state, shared route selection, fanout/join modeling, and declared destinations are exported. |
| LangGraph compiled graph to Kogwistar | Explicitly unsupported | `from_langgraph()` raises `LangGraphImportUnsupportedError`; arbitrary callables and dynamic `Command`/`Send` targets are not lossless. |
| Round-trip equivalence | Not claimed | `WorkflowDesignArtifact` remains the canonical interchange contract. |
| Nested workflow execution | Supported natively | Invoke-and-await preserves result propagation, distinct child run IDs, cycle/depth guards, and persisted `wf_invoked` lineage. Durable background spawn remains out of scope. |
| Nested workflow export/import | Partial | Export preserves workflow node metadata and declared invocation metadata/design references; arbitrary compiled LangGraph subgraphs are not losslessly importable. Dynamic child materialization remains a general runtime concern, not an interop guarantee. |
| Run lineage API | Ancestor chain v1 | The current endpoint returns the requested run and persisted parent chain. It is not yet a complete descendant tree or a substitute for a workflow-run graph projection. |
| Child viewer navigation | Separate canvas v1 | Breadcrumbs, per-workflow layout isolation, unresolved-child errors, and ACL enforcement are supported by the external viewer contract. Expand-in-place is intentionally deferred; the built-in designer remains a single-graph surface. |

Runtime `edge_selected` telemetry is the selection authority. Viewers may show
all declared routes, but highlight a selected route only from the native run
event payload; fixture-only route selection is not a supported contract.

Interop is deliberately one-way at the semantic boundary. A Kogwistar graph
may be rendered/exported to LangGraph-compatible structures, but a compiled
LangGraph object is not treated as a recoverable workflow design because
callable closures, dynamic `Command`/`Send` destinations, and arbitrary nested
subgraphs do not expose a lossless portable design contract.
