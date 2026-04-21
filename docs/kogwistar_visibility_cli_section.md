## CLI / Operator Visibility Conventions

This section defines the **shared visibility plane** for Kogwistar.  
All slices MUST expose their observable state through this common model.

### Principles

- Visibility is a **projection**, never authoritative truth.
- All CLI / MCP / REST inspection must be **reconstructible from events + graph truth**.
- There is **one unified operator surface**, not per-slice custom tools.
- ACL-controlled visibility must come from persisted ACL truth, not from CLI-only or in-memory-only filters.
- All visibility must respect:
  - `security_scope`
  - `execution_namespace`
  - `storage_namespace`
  - capability filtering (unless explicitly elevated)

---

## Command Shape

All inspectable entities MUST follow a consistent 3-pattern interface:

1. **List / Summary**
2. **Detail / Status**
3. **Event / Audit**

Pattern:

```
kogwistar <noun> list
kogwistar <noun> get <id>
kogwistar <noun> events <id>
```

Optional:

```
kogwistar <noun> audit <id>
kogwistar <noun> pending
kogwistar <noun> queue
```

---

## Global Commands

```
kogwistar status
```

Returns system-wide summary:

- active / blocked workflow runs
- degraded services
- pending messages / inbox backlog
- recent failures / repair actions

---

## Slice-Aligned CLI Surface

### Process / Execution (Slice 1–2)

```
kogwistar ps
kogwistar ps <run_id>
kogwistar ps events <run_id>
```

Must expose:

- run_id
- workflow_id
- state
- blocked reason (if any)
- current node / step
- last checkpoint
- security scope

---

### Messaging / IPC (Slice 5)

```
kogwistar msg inbox <inbox_id>
kogwistar msg pending
kogwistar msg events <message_id>
```

Must expose:

- per-inbox ordering / latest seq
- pending / leased / failed messages
- lease state
- sender / recipient
- requeue / retry state

---

### Capability / Permissions (Slice 4)

```
kogwistar capability snapshot
kogwistar capability approvals
kogwistar capability audit
```

Must expose:

- subject / action / capability mapping
- approvals (subject-local)
- revocations
- audit decisions (allow / deny / require approval)

---

### Scheduler (Slice 7)

```
kogwistar scheduler queue
kogwistar scheduler blocked
kogwistar scheduler policy <run_id>
```

Must expose:

- scheduling decisions
- blocked reasons (budget, capability, dependency)
- queue ordering
- policy inputs (not just final outcome)

---

### Budget / Quota (Slice 8)

```
kogwistar budget status
kogwistar budget ledger <subject_or_run>
```

Must expose:

- budget consumption
- remaining quota
- debit/credit history
- enforcement events

---

### Tool / Device Execution (Slice 9)

```
kogwistar tool calls
kogwistar tool call <call_id>
```

Must expose:

- tool invocation history
- arguments (sanitized if needed)
- result / error
- latency / execution metadata

---

### Service / Daemon (Slice 10)

```
kogwistar service list
kogwistar service status <service_id>
kogwistar service events <service_id>
```

Must expose:

- service_id
- service_kind
- enabled / disabled
- health state (healthy / degraded / stopped)
- last heartbeat
- restart count
- last trigger source
- current child run id (if any)

---

### Recovery / Replay / Repair (Slice 12)

```
kogwistar repair pending
kogwistar repair history
kogwistar replay status
```

Must expose:

- pending repair actions
- completed repair actions
- before / after state linkage
- replay progress and checkpoints

---

## Permission Model for Visibility

By default:

- CLI output MUST respect capability filtering
- Cross-scope visibility MUST be denied unless explicitly allowed

Elevated operator mode MAY exist:

```
kogwistar --admin ...
```

But MUST be:

- explicitly gated
- auditable
- never the default path

---

## Projection Requirements

All CLI commands MUST be backed by:

- fast latest-state projections (e.g. `meta_sqlite`)
- rebuildable from authoritative truth

Deleting projection rows MUST NOT break CLI correctness.

---

## Anti-Patterns

The following are forbidden:

- visibility layer becoming authoritative truth
- per-slice custom CLI patterns
- bypassing capability or namespace checks in operator tools
- embedding hidden state only visible via logs
- operator-only state not derivable from graph + events

---

## Minimal Slice Contract

Each slice MUST define:

- at least one `list` or summary command
- at least one `get/status` command
- at least one `events/audit` command

If a slice cannot be inspected via these, it is considered **non-operable**.
