# Slice 3 Scope / Partition Map

```text
claims
  ├─ storage_ns
  ├─ execution_ns
  ├─ security_scope
  ├─ tenant / workspace / project
  └─ agent_id

derived helpers
  ├─ get_storage_namespace()
  ├─ get_execution_namespace()
  ├─ get_security_scope()
  ├─ get_security_scope_parts()
  └─ describe_storage_security_mapping()

rules
  ├─ storage namespace = backend partition
  ├─ execution namespace = runtime lane
  ├─ security scope = visibility gate
  ├─ tenant/workspace/project = hierarchical scope parts
  └─ agent_id = private memory owner key

access
  ├─ private memory => owner_agent_id match
  ├─ shared memory => explicit agent/scope grant
  ├─ pinned ref / projection => same filter
  └─ backend keeps metadata only, no new truth
```
