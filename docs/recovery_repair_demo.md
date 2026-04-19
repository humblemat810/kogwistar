# Recovery / Repair Demo

This demo shows `Slice 12` recovery utilities.

## What it proves

- A service projection can be rebuilt from authoritative service truth.
- Dead-letter snapshot is visible.
- Operator dashboard still works after repair actions.

## Run

```powershell
.\.venv\Scripts\python.exe -m kogwistar.demo.recovery_repair_demo
.\.venv\Scripts\pytest.exe tests/core/test_recovery_repair_demo.py -q
```

