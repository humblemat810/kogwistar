# ARD-0000: ARD Naming And Catalog

**Status:** Accepted  
**Date:** 2026-03-06  
**Owner:** Maintainers

## Naming Convention
Use these conventions for architecture records in this folder:

1. `ADR-<NNN>-<kebab-case-topic>.md` for Architecture Decision Records.
2. `ARD-<kebab-case-topic>.md` for Architecture Requirements/Design Records without numeric sequence.
3. `ARD-<NNNN>-<kebab-case-topic>.md` for globally numbered ARDs (zero-padded to 4 digits).
4. `ARD-p<phase>-<kebab-case-topic>.md` for phase-series ARDs (example: `ARD-p2d-...`).
5. Use lowercase kebab-case for topic segments. Avoid underscores and mixed casing.
6. Do not tombstone for minor status/content updates; edit in place and capture changes in a history section.

## Consolidation Performed (2026-03-05)
Renamed files to align with kebab-case convention:

- `ARD_Agentic_Answering_System.md` -> `ARD-agentic-answering-system.md`
- `ARD_postgresql_inclusive.md` -> `ARD-postgresql-inclusive.md`
- `ARD_Retrieval_Orchestration.md` -> `ARD-retrieval-orchestration.md`
- `ARD_WorkflowRuntime_Token_Nesting.md` -> `ARD-workflowruntime-token-nesting.md`
- `ARD-ContextSnapshot-and-PromptContext.md` -> `ARD-context-snapshot-and-prompt-context.md`

Metadata normalization:

- `ARD-0012-repo-restructure-core-workflow-conversation.md`: header ID corrected from `ARD-0001` to `ARD-0012`.
- `ARD-postgresql-inclusive.md`: status updated to Accepted (Implemented; code-derived) and history added.

## Consolidation Performed (2026-03-06)
Normalized numbering and phase-series naming:

- `ARD-006-conversation.md` -> `ARD-0006-conversation.md`
- `ARD-2D-conversation-workflow-v2-parity.md` -> `ARD-p2d-conversation-workflow-v2-parity.md`
- `ARD-2D-Appendix-A-state-retries-ordering.md` -> `ARD-p2d-appendix-a-state-retries-ordering.md`
- Updated corresponding document headers and cross-file appendix reference.

## Current ARD/ADR Catalog

### ADR
- `ADR-004-phase4-pgvector-authoritative.md`

### ARD (numbered, global)
- `ARD-0000-naming-and-catalog.md`
- `ARD-0006-conversation.md`
- `ARD-0012-repo-restructure-core-workflow-conversation.md`

### ARD (topic + phase series)
- `ARD-p2d-conversation-workflow-v2-parity.md`
- `ARD-p2d-appendix-a-state-retries-ordering.md`
- `ARD-agentic-answering-system.md`
- `ARD-context-snapshot-and-prompt-context.md`
- `ARD-postgresql-inclusive.md`
- `ARD-retrieval-orchestration.md`
- `ARD-workflowruntime-token-nesting.md`
