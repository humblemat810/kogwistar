# Service Daemon Demo

This demo shows `Slice 10` as a managed service, without recursive read-path
ticks.

## What it proves

- A service definition can be declared.
- Heartbeat updates health.
- Autostart can wake child workflow once.
- An external trigger can spawn child run.
- Operator dashboard shows service plus child workflow.

## Shape

```text
service_definition
  -> service_event
  -> service projection
  -> process table / dashboard
```

## Run

```powershell
.\.venv\Scripts\python.exe -m kogwistar.demo.service_daemon_demo
.\.venv\Scripts\pytest.exe tests/core/test_service_daemon_demo.py -q
```

