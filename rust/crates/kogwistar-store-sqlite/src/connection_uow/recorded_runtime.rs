use super::*;

impl SqliteUnitOfWork<'_> {
    pub fn recorded_runtime_steps(
        &self,
        run_id: &str,
        step_seqs: &[i64],
    ) -> SqliteStoreResult<Vec<ServerRunEvent>> {
        if step_seqs.is_empty() {
            return Ok(Vec::new());
        }
        let requested = step_seqs
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        Ok(recorded_runtime_events(self.transaction, run_id)?
            .into_iter()
            .filter(|event| {
                parse_runtime_event_payload(&event.payload_json)
                    .ok()
                    .flatten()
                    .is_some_and(|payload| requested.contains(&payload.reduced.state.last_step_seq))
            })
            .collect())
    }

    pub fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(self.transaction, transition, abort_after_writes)
    }

    pub fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            self.transaction,
            handoff,
            transition,
            abort_after_writes,
        )
    }

    pub fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> SqliteStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(self.transaction, handoff, effect)
    }
}
