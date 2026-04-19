# Operator Views Demo

This demo shows `Slice 11` operator surfaces.

## What it proves

- Process table shows runs and services.
- Operator inbox returns projected lane messages.
- Blocked runs and dashboard are readable in one view.
- Capability and resource snapshots stay projection-based.

## Run

```powershell
.\.venv\Scripts\python.exe -m kogwistar.demo.operator_views_demo
.\.venv\Scripts\pytest.exe tests/core/test_service_daemon_demo.py -q
```

