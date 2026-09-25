# Agent Capability Map

本表為 agent harness 之 transport-neutral ownership map；capability ID 為語義
identity，REST、MCP、Python、workflow adapter 不得各自實作一份 authority。

| Capability ID | Owner | Effect | Existing seam | Read contract/test |
| --- | --- | --- | --- | --- |
| `workflow.run.read` | `WorkflowRuntime`/chat service | read | run status/events/steps/checkpoints/lineage/evidence | `AgentReadTools.execution_*` |
| `conversation.memory.read` | conversation memory retriever | read | `memory.search` | `AgentReadTools.memory_search` |
| `knowledge.graph.read` | graph engine | read | `knowledge.search/get/expand` | `AgentReadTools` |
| `wisdom.lesson.read` | wisdom serving | read | approved-status search | `AgentReadTools.wisdom_search` |
| `catalog.read` | agent catalog projection | read | `catalog.search/browse/get` | `tests/agent/test_agent_read_tools.py` |
| `skill.read` | skill provider/projection | read | descriptor/raw/graph progressive disclosure | `skill.get`, `skill.resource_get` |
| `mcp.discovery.read` | MCP discovery provider | read | `mcp.describe` | `mcp_describe` |
| `capability.describe` | capability kernel | read | `capability.describe` | `capability_describe` |
| `workflow.run.write` | runtime service | write | ordinary `WorkflowRuntime.run/resume` | runtime contract tests |

## Authority Rules

- Profile, prompt, hook, catalog descriptor, and parsed skill graph never grant
  capability.
- Read adapters filter scope and ACL before pagination or ranking.
- Semantic ranking is optional; lexical/exact/alias/prefix/partial discovery
  remains valid while embeddings are unavailable.
- Skill source packages remain provider-owned. Parsed graphs are rebuildable
  projections with fingerprints and revision gates.
- MCP schema discovery is distinct from MCP invocation authorization.

## Phase Ownership

Phase 0 inventories this map. Phase 1 composes contracts and lifecycle hooks.
Phase 2 exposes bounded read operations and progressive skill/MCP discovery.
Later phases add control, delegation, compression, wisdom projection, optional
providers, A2A, and CrewAI import without changing these owners.
