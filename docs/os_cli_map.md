# Kogwistar OS CLI / MCP Map

This map lists real user-facing diagnostics / operator entrypoints. Names below must match code.

## Conversation

- `conversation.create`
- `conversation.get_transcript`
- `conversation.ask`
- `conversation.run_status`
- `conversation.cancel_run`

## Workflow Runs

- `workflow.run_submit`
- `workflow.run_status`
- `workflow.run_cancel`
- `workflow.run_events`
- `workflow.run_checkpoint_get`
- `workflow.run_replay`
- `workflow.run_resume_contract`
- `workflow.run_resume`

## Operator Views

- `workflow.process_table`
- `workflow.operator_inbox`
- `workflow.blocked_runs`
- `workflow.process_timeline`
- `workflow.operator_dashboard`

## Service / Daemon

- `workflow.service_declare`
- `workflow.service_list`
- `workflow.service_get`
- `workflow.service_enable`
- `workflow.service_disable`
- `workflow.service_heartbeat`
- `workflow.service_trigger`
- `workflow.service_events`

## Recovery / Repair

- `workflow.service_repair`
- `workflow.services_repair`
- `workflow.message_orphans_repair`
- `workflow.dead_letters`
- `workflow.dead_letter_replay`

## Capability Kernel

- `workflow.capabilities_snapshot`
- `workflow.capability_approve`
- `workflow.capability_revoke`

## Workflow Design

- `workflow.design_node_upsert`
- `workflow.design_edge_upsert`
- `workflow.design_node_delete`
- `workflow.design_edge_delete`
- `workflow.design_history`
- `workflow.design_undo`
- `workflow.design_redo`

## Notes

- Some old names in earlier drafts were conceptual only. Use only names above.
- This map is diagnostics / operator surface. It does not define backend truth.
- If a new surface is added, update both this map and the slice checklist.

