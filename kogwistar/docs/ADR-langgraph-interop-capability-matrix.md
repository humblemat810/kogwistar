# LangGraph interop capability matrix

Status: conservative, one-way interop.

| Capability | Status | Boundary |
|---|---|---|
| Kogwistar design to LangGraph visual graph | Supported | Visual mode is best-effort and does not become native runtime authority. |
| Kogwistar design to LangGraph semantics graph | Supported | Blob state, shared route selection, fanout/join modeling, and declared destinations are exported. |
| LangGraph compiled graph to Kogwistar | Explicitly unsupported | `from_langgraph()` raises `LangGraphImportUnsupportedError`; arbitrary callables and dynamic `Command`/`Send` targets are not lossless. |
| Round-trip equivalence | Not claimed | `WorkflowDesignArtifact` remains the canonical interchange contract. |
| Nested workflow execution | Supported natively | Invoke-and-await preserves result propagation, distinct child run IDs, cycle/depth guards, and persisted `wf_invoked` lineage. |
| Child viewer navigation | Separate canvas v1 | Breadcrumbs, per-workflow layout isolation, unresolved-child errors, and ACL enforcement are supported. Expand-in-place is intentionally deferred. |

Runtime `edge_selected` telemetry is the selection authority. Viewers may show
all declared routes, but highlight a selected route only from the native run
event payload; fixture-only route selection is not a supported contract.
