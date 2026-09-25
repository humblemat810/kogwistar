# ACL Taint and Declassification

Kogwistar distinguishes **visibility ACL** from **derived-output taint**.
Visibility decides who may read a source. Taint decides the minimum ACL carried
by any artifact derived from semantic inputs.

## Strict default

`STRICT` is default and needs no model or network dependency. Every semantic
input contributes to one conservative high-water join:

```text
output ACL = join(user query, private context, retrieved sources,
                  tool output, conversation, memory, reasoning, metadata)
```

`public` source plus `private` query therefore produces `private` output.
Mixed private owners retain `private` and do not grant either owner automatic
access. Source identities and ACL descriptors are recorded without storing
prompt or document content in the audit record.

The source object is not changed. A public source may remain public while its
derived answer is private.

## LLM_GUARDED

`LLM_GUARDED` is **declassification**, not another taint definition.

Execution order is fixed:

```text
semantic inputs -> STRICT join -> optional guard proposal -> final ACL
```

The guard may propose a less restrictive ACL only after the strict result
exists. It cannot widen access, erase provenance, or make a private input
disappear. Missing guard, exception, timeout handled by the caller, malformed
response, low confidence, or uncertainty keeps the strict result.

Enabling this policy requires explicit acknowledgement. The audit contains:

- original strict ACL;
- proposed and final ACL;
- policy, model, and classifier version;
- reason, confidence, evidence, timestamp;
- input identities and joined ACL;
- declassification attempted/result;
- clean-room assertion.

Model payload fields named `acl`, `visibility`, or similar are not authority.
Only a host-supplied guard callback and runtime authority context can affect
the decision.

## Runtime boundary

`register_model_step` and `register_tool_step` emit ACL metadata alongside
their ordinary result. ACL descriptors from `StepContext.authority_context`
are trusted inputs. Mutable workflow state is provenance only and cannot
downgrade an input to public; untrusted state-side descriptors fail closed as
private.

Retrieval must filter by ACL before context construction. Any cache key for a
derived result must include principal/scope, source ACL join, policy, and
provenance fingerprint. Cross-agent handoff carries the derived ACL; a private
input taints the next agent unless an explicit guarded declassification is
audited.

## Persistence

Derived ACL audit may be stored in the existing ACL record's `derivation_audit`
property. It is provenance, not a replacement for canonical graph truth. No
new event store or alternate ACL authority is introduced.
