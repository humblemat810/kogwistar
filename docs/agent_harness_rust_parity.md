# Rust Agent Harness Parity

## Scope

`rust/crates/kogwistar-agent` is the first Rust-native composition surface for the
Python `kogwistar.agent` harness. It intentionally remains a thin layer:

- it validates an `AgentProfile` and fails closed before dispatch;
- it intersects requested capabilities with caller authority and revoked
  capabilities;
- it delegates execution to an injected Rust runtime seam;
- it does not add a second workflow engine, scheduler, checkpoint store, or
  memory authority.

The crate depends on the Kogwistar Rust runtime for canonical JSON and the
recorded-runtime contract version.

## Implemented Surface

| Python surface | Rust surface | Status |
| --- | --- | --- |
| `AgentProfile` | `kogwistar_agent::AgentProfile` | Partial |
| `AgentHarness` / `AsyncAgentHarness` | `AgentHarness` / `AsyncAgentHarness` | Partial |
| `ContextPolicy` / `ContextSnapshot` | `ContextPolicy` / `ContextSnapshot` | Partial |
| `AgentBudgetPolicy`, budget hints | `AgentBudgetPolicy`, `budget_hints` | Partial |
| `ControlPolicy` / `ControlPointSpec` | `ControlPolicy` / `ControlPointSpec` | Partial |
| delegation helpers | `DelegationSpec`, `DelegationContext`, helpers | Partial |
| `ProviderRegistry` | `ProviderRegistry` | Partial |
| `HookRegistry` | `HookRegistry` | Partial |

## Non-Goals

The current Rust crate does not claim parity with the full Python package. The
following remain separate work:

- durable catalog/skill projection stores;
- live model providers and streaming event contracts;
- durable event journals and control journals;
- workflow graph builders and their validation gates;
- live PostgreSQL/SQLite parity evidence for agent effects.

## Verification

From `rust/`:

```bash
cargo test -p kogwistar-agent --locked
cargo fmt --all -- --check
cargo clippy -p kogwistar-agent --all-targets --locked -- -D warnings
```

Focused tests cover capability attenuation, binding validation, authority
context construction, runtime delegation, budget seeding, context
fingerprints, control-point validation, delegated child state, bounded child
results, provider lifecycle, and hook capability gating.