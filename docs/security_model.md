# Agent Security Model

## Overview

Kogwistar is a graph, workflow, provenance, and MCP/REST substrate for agent
systems. Its security boundary is not defined by whether code can be found,
installed, or invoked. The governing rule is:

> Code must not gain the authority of the main agent merely because the agent
> can discover, install, or invoke it.

The main agent process is a trusted computing base (TCB). Code outside that
process must use a separate runtime identity, explicitly scoped credentials,
and an enforced protocol or operating-system boundary. MCP is a protocol
boundary; it is not automatically a sandbox.

This document defines the intended security architecture. Statements marked
REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION are requirements for
integrators and future implementation work, not claims that the current
repository already enforces them.

## Trust Boundary

~~~mermaid
flowchart LR
    U[User identity] --> M[Main agent<br/>agent:main]
    M --> P[Policy and delegation<br/>attenuate capabilities]
    P --> T[Trusted in-process tools<br/>TCB]
    P --> R[Authenticated RPC / MCP]
    R --> X[External tool or MCP service<br/>separate principal]
    P --> D[Delegated agent runtime<br/>agent:child]
    X --> E[Enforced OS / container / VM policy]
    D --> E
    T --> G[Graph, workflow, and provenance APIs]
    X --> G2[Only explicitly granted resources]
    D --> G2
    G --> V[Receipts and provenance]
    G2 --> V
~~~

An MCP server running as the same user, with the same environment, HOME,
filesystem, network, and credentials is not an isolated security boundary.

## Definitions

- **TCB:** Code whose compromise can directly exercise the main agent's
  process authority, credentials, memory, or trusted in-process APIs.
- **Built-in tool:** A bounded tool shipped in the same reviewed, versioned
  release as the main agent and intentionally included in the TCB.
- **External tool:** User-, marketplace-, downloaded-, or separately deployed
  executable code that is not automatically part of the TCB.
- **MCP service:** A protocol endpoint that exposes tools or resources. MCP
  provides message semantics; isolation requires separate runtime and policy
  controls.
- **Agent:** A component with an independent decision loop, persistent state,
  multi-step behavior, tool choice, delegation, or capability acquisition.
- **Principal:** An attributable identity such as user:id, agent:id, tool:id,
  or service:id.
- **Capability:** An authority to perform an operation on a resource. A
  manifest declaration is not an authorization grant.
- **Delegation:** A principal granting a bounded subset of its authority to
  another principal for a specified task, resource, and lifetime.
- **Provenance:** The immutable chain recording initiator, delegators,
  executor, authorization decision, resource, operation, credential class,
  and returned result.

## Trusted Computing Base

Code executing directly inside the main agent process belongs to the TCB. A
built-in tool should normally be:

- shipped with the agent;
- installed from the same trusted release;
- version-pinned and integrity-checked, preferably signature or hash pinned;
- code-reviewed under the agent's security review process; and
- immutable or tightly controlled at runtime.

“Preinstalled” is not sufficient evidence of trust. A compromised dependency
inside a trusted release remains a TCB risk and requires dependency review,
integrity verification, and least-privilege credentials.

The following is unsafe as a normal extension mechanism:

~~~text
agent decides a package is useful
  -> pip/npm install package
  -> import package into the agent process
  -> package inherits agent identity and credentials
~~~

Installation provenance only says where code came from. It does not establish
that the code was deliberately reviewed as part of the TCB, nor does it reduce
the authority available after import.

REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: the production launcher
must prevent arbitrary runtime package installation into the main agent
environment, or require an explicit trusted-release/restart workflow.

## External Tools And MCP Services

Externally supplied code should execute behind a boundary such as a dedicated
process, container, sandbox, VM, delegated agent runtime, or MCP service. The
boundary must be paired with a distinct principal and constrained credentials.

~~~text
agent:main
  -- authenticated RPC/MCP, scoped delegation -->
tool:github-reader
  credential: repository-read-only
  filesystem: none
  network: GitHub API only
~~~

A malicious or defective write attempt must fail at authorization or
infrastructure enforcement, not depend on the service voluntarily honoring a
read-only description.

REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: each production MCP
service has its own OS/runtime identity, environment allowlist, filesystem
root, network egress policy, resource limits, credential set, and auditable
request identity. Separate process alone does not prove any of these.

## Declarations, Authorization, And Enforcement

These are separate concepts:

1. **Declaration:** the tool says what it intends or needs, for example
   repository.read.
2. **Authorization:** policy decides whether the principal may request that
   operation on that resource in this context.
3. **Enforcement:** credentials, filesystem ACLs, network controls, OS policy,
   and guarded APIs make unauthorized behavior fail.

The effective capability set is bounded by every applicable control:

~~~text
EffectiveCapability =
    DeclaredCapability
    intersect AgentPolicy
    intersect UserPolicy
    intersect DelegationPolicy
    intersect IAMCredentials
    intersect FilesystemPolicy
    intersect NetworkPolicy
    intersect OSOrSandboxPolicy
~~~

Declarations never expand authority. A tool claiming “read-only” may still be
malicious, buggy, or compromised. The final read-only property must come from
read-only credentials and enforced resource controls.

REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: capability checks are
performed outside untrusted executable code wherever possible, and a request
cannot authorize itself by returning a favorable manifest.

## Tool Versus Agent

A tool is a bounded capability endpoint. A component should be treated as an
agent when it can reason independently, retain state, choose among tools,
perform multi-step actions, delegate, acquire capabilities, modify memory, or
make decisions beyond one constrained operation.

~~~text
user:peter
  -> agent:main
      -> agent:security-review
          -> tool:repository-reader
~~~

The child agent must have its own principal and delegated capability set. It
must not silently inherit the parent identity merely because it was launched
by the parent.

## Identity And Delegation

Use explicit principal forms:

~~~text
user:<id>
agent:<id>
tool:<id>
service:<id>
~~~

Every privileged action must identify the principal that actually executed it,
not only the user who initiated the top-level request. Delegated authority is
normally narrower than or equal to the delegator's authority. Capability
attenuation should constrain at least:

- allowed operation: read, write, execute, administer;
- resource and namespace;
- workspace, repository, or graph scope;
- network destination;
- maximum data volume and runtime;
- expiration and single-use constraints; and
- whether further delegation is permitted.

Prefer short-lived scoped credentials or token exchange to forwarding a
parent's bearer token. Parent credentials must not be copied into child
environments by default. Revocation must invalidate delegated credentials or
make subsequent authorization fail.

This prevents confused-deputy attacks: a child cannot turn the parent's
identity into permission to access a resource that the child was not delegated
to use.

REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: the complete
delegation chain, attenuation decision, credential class, and revocation state
are persisted with each privileged operation.

## Provenance And Delegation Chains

The minimum provenance shape is:

~~~text
user:peter
  -> agent:main
      -> agent:security-review
          -> tool:git-reader
              -> repository:abc
                  operation:read
~~~

Receipts should answer:

- who initiated the operation;
- which principal delegated it;
- which principal executed it;
- which capability authorized it;
- which resource and namespace were touched;
- which credential class was used;
- whether the operation was read or write; and
- what output was returned upstream.

Do not collapse the chain to “the user performed the action.” Untrusted tool
output is also data with provenance and must not become an instruction merely
because it is returned to the parent agent.

## Trust Classes

### Class A: Core / TCB Tool

- runs in the agent process or a tightly coupled trusted runtime;
- ships with the reviewed release;
- is version- and integrity-pinned;
- may use the agent identity only where justified; and
- changes require normal release and security review.

### Class B: External Tool / MCP Service

- is outside the TCB;
- runs in an isolated or separately governed runtime;
- has a separate principal and scoped credentials;
- receives explicitly granted capabilities; and
- has no ambient access to parent secrets.

### Class C: Delegated Agent

- has an autonomous decision loop;
- has a separate agent identity;
- receives an explicit delegated capability set;
- has its own runtime and state boundary where appropriate; and
- preserves the complete delegation and provenance chain.

Trust class and execution mechanism are related but not identical. Placing
software behind MCP does not make it trustworthy, and a container does not
guarantee isolation without correct user, mount, network, credential, and
kernel configuration.

## Default Security Invariants

1. Main-process code belongs to the TCB.
2. Runtime-installed executable extensions do not automatically enter the TCB.
3. External executable components receive separate principals.
4. Capability manifests describe requested authority; they do not grant it.
5. Authorization and enforcement occur outside untrusted code where possible.
6. MCP is a protocol boundary; isolation is separately enforced.
7. Autonomous extensions normally become agent principals.
8. Delegation attenuates capabilities by default.
9. Parent credentials are not automatically passed to children.
10. Privileged operations retain end-to-end provenance.
11. Read-only claims require read-only credentials or infrastructure.
12. A compromised tool must not automatically compromise the main agent identity.
13. Workspace, namespace, ACL, and resource authorization precede ranking,
    prompting, or tool execution.
14. Untrusted output cannot directly create new authority or recursive
    delegation.

### Unbounded Projection Reads

PostgreSQL projection reads support an explicit `limit=None` for trusted repair
and rebuild workflows. This removes SQL `LIMIT`; it does not remove ACL,
tenant, namespace, resource-budget, or audit requirements. Public REST/MCP
surfaces must not pass an untrusted unlimited-read request through directly.
They must impose a finite limit or bounded cursor/page contract before invoking
the backend. The collection default remains bounded. The backend itself does
not enforce this service-boundary policy, so each REST/MCP adapter remains
responsible for doing so.

At the core engine boundary, `acl_enabled=True` requires the engine-owned ACL
policy plus guarded read/write protocol surfaces. `acl_enabled=False` does not
require those ACL protocols and preserves the raw backend contract. This is a
conditional engine contract, not a requirement for every storage backend to
reimplement ACL policy.

The backend also omits pgvector embeddings unless `"embeddings"` is explicitly
requested. This limits unnecessary data transfer and memory use, but is not an
authorization check: embedding access remains subject to the same graph and
scope policy as document and metadata access.

## Threat Scenarios And Controls

| Threat | Required containment |
| --- | --- |
| Tool claims read-only but writes | Enforce write denial with credentials, API policy, filesystem policy, and graph ACLs. |
| Supply-chain compromise | Release pinning, hashes/signatures, dependency review, isolated runtime, and least privilege. |
| MCP reads parent environment | Separate runtime identity and environment allowlist; do not mount parent HOME or secrets. |
| Tool escapes its working directory | OS/container sandbox, read-only mounts, explicit bind mounts, and path enforcement. |
| Unexpected network endpoint | Egress allowlist, DNS/proxy policy, and service-specific credentials. |
| Child forwards parent credentials | Token exchange with attenuation; prohibit ambient environment/token inheritance. |
| Confused deputy | Authorize the child principal and resource, not only the parent request. |
| Nested delegation privilege escalation | Maximum delegation depth, no authority widening, explicit re-delegation permission. |
| Tool updates itself after review | Immutable image/package environment and controlled upgrade workflow. |
| Compromised trusted dependency | Integrity checks, dependency pinning, review, and TCB minimization. |
| Prompt injection in tool output | Treat output as untrusted data; re-authorize every requested action. |

## Secure And Insecure Deployments

### Secure shape

~~~text
agent:main
  -> authenticated RPC
tool:github-reader
  OS user: tool-github-reader
  filesystem: empty/read-only temporary root
  network: api.github.com only
  credential: short-lived repository-read token
  writes: denied by token and runtime policy
  receipts: retained with delegation chain
~~~

### Insecure shape

~~~text
agent:main
  -> MCP over localhost
MCP server:
  same OS user
  same HOME and environment
  parent API tokens available
  unrestricted filesystem
  unrestricted network
  self-updatable package
~~~

The second shape has a protocol hop but no meaningful isolation boundary.

## Kogwistar Integration Status

Kogwistar already exposes graph ACL, namespace, authentication, workflow
capability, MCP, and provenance-related primitives. Their presence does not
by itself prove that every external tool deployment applies the model above.

The following must be verified per deployment and integration:

- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: external MCP
  processes receive separate OS identities and scoped secrets.
- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: MCP tool calls
  carry an authenticated principal and delegation chain end to end.
- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: capability
  declarations cannot bypass graph ACL, namespace, or REST/MCP authorization.
- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: child agents cannot
  recursively create maintenance or delegation authority without policy.
- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: filesystem and
  network restrictions are enforced outside the MCP server implementation.
- REQUIRED BY MODEL - NOT YET VERIFIED IN IMPLEMENTATION: receipts identify
  the actual executor and credential class rather than only the initiating user.

These are implementation and deployment gates, not assumptions to be silently
treated as guarantees.

## What This Model Does Not Guarantee

This model does not claim that:

- preinstalled code is automatically trustworthy;
- MCP automatically provides sandboxing;
- a container alone guarantees security;
- a capability manifest enforces behavior;
- a separate process automatically has a separate identity;
- read-only APIs prevent every side effect;
- provenance makes malicious behavior harmless; or
- authentication alone limits filesystem, network, or operating-system access.

Security depends on the enforcement controls at the actual runtime boundary.

## Recommended Implementation Requirements

1. Define a principal and credential class for every tool, service, and agent.
2. Require explicit, attenuated delegation for every cross-boundary call.
3. Carry principal, delegator, capability, resource, namespace, and operation
   through MCP, REST, workflow, and graph receipts.
4. Keep external tools outside the main process and deny ambient credentials.
5. Enforce filesystem, network, resource, and graph policy outside the tool.
6. Make trusted runtime dependencies reproducible, pinned, and integrity-checked.
7. Treat tool output as untrusted data at every parent-agent boundary.
8. Add negative tests proving read-only tools cannot write, cross-namespace
   calls fail, and delegation cannot widen authority.
9. Add deployment checks for identity separation, secret absence, mount policy,
   egress policy, and resource limits.
10. Record and review delegation chains for every privileged operation.

## Open Architectural Questions

- Which principal and token-exchange service issues short-lived child tokens?
- Is delegation depth bounded globally, per agent, or per operation?
- Which graph receipts are authoritative for security audit and retention?
- Which MCP transports require mutual authentication rather than bearer auth?
- What sandbox is mandatory for local desktop, Docker, Kubernetes, and VM
  deployments?
- Which capabilities may be delegated again, and how is that recorded?
- How are tool package hashes and signatures admitted into a trusted release?
- Which operator emergency controls can revoke all child credentials quickly?
- Which external tool outputs may be stored as graph evidence, and what
  redaction policy applies to them?
