# Semantic Dedup Matrix (落刀去重)

Goal: one canonical spec per semantic area. Tutorials/demos only “how to run”, no re-stating contracts.

## Canonical Semantics (Truth Docs)

- Slice overview / status: `docs/kogwistar_ai_os_slice_checklist.md`
- Slice 3 (Namespaces/Isolation):
  - `docs/slice3_scope_partition_map.md`
  - `docs/slice3_memory_visibility_diagram.md`
- Slice 4 (Capability kernel): `docs/slice4_capability_kernel.md`
- Slice 5 (Syscall surface): `docs/syscall_surface_v1.md`
- Slice 6 (Resume semantics): `tests/runtime/test_resume_wait_reasons.py` + runtime contracts (tests are truth for edge cases)
- Slice 7 (Scheduler): `docs/run_scheduler_flow.md`
- Slice 8 (Budgets/resource accounting): `docs/budget_accounting_flow.md`
- Slice 9 (Tool/device subsystem): `docs/tool_device_subsystem.md`
- Slice 10 (Service/daemon model): `docs/service_daemon_model.md`
- Slice 11 (Operator views/introspection): `docs/operator_views_introspection.md`
- Slice 12 (Recovery/repair utilities): `docs/recovery_repair_utilities.md`

## Tutorials / Demos (How-To Only)

- Slice 8 demos:
  - `docs/budget_rate_switch_demo.md`
  - `docs/budget_branch_pin_demo.md`
- Slice 10 demo: `docs/service_daemon_demo.md` + `kogwistar/demo/service_daemon_demo.py` (if present)
- Slice 11 demo: `docs/operator_views_demo.md`
- Slice 12 demo: `docs/recovery_repair_demo.md` + `kogwistar/demo/recovery_repair_demo.py`
- Scheduler priority demo: `docs/scheduler_priority_demo.md` + `kogwistar/demo/scheduler_priority_demo.py`

Rule: demo doc must link back to canonical doc, and only include:
- prerequisites
- exact commands
- expected output/assertions
- troubleshooting

## Interface Index (Names Must Match Code)

- MCP/CLI map (index): `docs/os_cli_map.md`

Rule: `docs/os_cli_map.md` must list only real tool names / endpoints that exist in code.

