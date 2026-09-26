| Capability                                          | Your System               | Datomic         | Temporal                       | LangGraph               | GraphRAG / KG tools |
| --------------------------------------------------- | ------------------------- | --------------- | ------------------------------ | ----------------------- | ------------------- |
| **Event sourcing backbone**                         | ✅ Core                    | ✅ Core          | ⚠️ (history of workflows only) | ❌                       | ❌                   |
| **Replay entire system state**                      | ⚠️ (partial but intended) | ✅               | ✅ (workflow only)              | ❌                       | ❌                   |
| **Unified graph substrate**                         | ✅ Hypergraph              | ⚠️ entity graph | ❌                              | ⚠️ execution graph only | ✅ KG only           |
| **Execution + knowledge unified**                   | ✅                         | ❌               | ❌                              | ⚠️ partial              | ❌                   |
| **Conversation as graph**                           | ✅                         | ❌               | ❌                              | ⚠️ ephemeral            | ❌                   |
| **Workflow trace as graph**                         | ✅                         | ❌               | ⚠️ history logs                | ⚠️                      | ❌                   |
| **Provenance as invariant**                         | ✅ strong                  | ❌               | ❌                              | ❌                       | ⚠️ weak citations   |
| **Context snapshot (LLM state)**                    | ✅                         | ❌               | ❌                              | ❌                       | ❌                   |
| **Multi-domain projections (conv/workflow/wisdom)** | ✅                         | ❌               | ❌                              | ❌                       | ❌                   |
| **Idempotent / CR-style mutation model**            | ⚠️ evolving               | ❌               | ❌                              | ❌                       | ❌                   |
| **CDC / event bus integration**                     | ✅                         | ⚠️              | ⚠️                             | ❌                       | ❌                   |
| **Wisdom / meta-learning layer**                    | ⚠️ planned                | ❌               | ❌                              | ❌                       | ❌                   |
