use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn apply_recorded_runtime_transition(
        &mut self,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_runtime_transition(
        &mut self,
        handoff: RecordedWorkerHandoff,
        transition: RecordedRuntimeTransition,
        abort_after_writes: bool,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_runtime_transition(
            &self.transaction,
            &self.tables,
            handoff,
            transition,
            abort_after_writes,
        )
        .await
    }

    pub async fn apply_claimed_recorded_worker_effect(
        &mut self,
        handoff: RecordedWorkerHandoff,
        effect: RecordedWorkerSuccessEffect,
    ) -> PostgresStoreResult<RecordedTransitionResult> {
        apply_claimed_recorded_worker_effect(&self.transaction, &self.tables, handoff, effect).await
    }
}
