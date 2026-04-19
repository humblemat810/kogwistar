# Slice 4 Capability Kernel

```text
claims_ctx
  ├─ subject / role / ns
  ├─ explicit capabilities? ---- yes --> use claims
  └─ no explicit caps ----------> legacy role grants

service layer
  ├─ approve_action
  ├─ revoke_capability
  ├─ require_capability(action, required)
  └─ audit_log append on every decision

runtime submit
  ├─ capture effective caps at submit time
  ├─ inject caps + subject into RuntimeRunRequest / RuntimeResumeRequest
  └─ background thread reuse captured caps; no context drift

child process
  ├─ inherit parent caps
  ├─ approvals can add
  └─ revokes can remove
```

Rules:
- capability truth live in service/kernel layer
- backend stay dumb
- denial durable: audit_log + snapshot
- child process no auto-all access

Current gates:
- read_graph
- write_graph
- send_message
- spawn_process
- invoke_tool
- read_security_scope
- project_view
- approve_action
- workflow.design.inspect
- workflow.design.write
- workflow.run.read
- workflow.run.write
