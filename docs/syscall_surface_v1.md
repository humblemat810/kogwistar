# Syscall Surface v1

```text
client
  -> /api/syscall/v1
  -> /api/syscall/v1/{op}

op
  spawn_process
  terminate_process
  send_message
  receive_message
  mount_memory
  project_view
  invoke_tool
  checkpoint
  resume
  request_approval
```

Shape:
- request: `{"version":"v1","op":"...","args":{...}}`
- response: `{"version":"v1","op":"...","status":"ok|blocked|error","result":{...},"error":{...}}`

Notes:
- `spawn_process` -> workflow submit
- `terminate_process` -> cancel
- `send_message` -> turn answer
- `receive_message` -> transcript read
- `mount_memory` -> snapshot view
- `project_view` -> resource view
- `invoke_tool` -> tool-facing preview path
- `checkpoint` -> checkpoint read
- `resume` -> resume flow
- `request_approval` -> blocked contract placeholder
