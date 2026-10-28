use super::*;

impl SqliteUnitOfWork<'_> {
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
