use super::*;

impl SqliteUnitOfWork<'_> {
    pub fn project_lane_message(&mut self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        project_lane_message(self.transaction, row)
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(self.transaction, message_id)
    }
    pub fn get_server_run(&self, run_id: &str) -> SqliteStoreResult<Option<ServerRun>> {
        server_run(self.transaction, run_id)
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(self.transaction, filter)
    }
    pub fn update_projected_lane_message_status(
        &mut self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_status(self.transaction, message_id, status, error_json)
    }
    pub fn update_projected_lane_message_links(
        &mut self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        update_projected_lane_message_links(
            self.transaction,
            message_id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
    }
    pub fn claim_projected_lane_messages_for_run(
        &mut self,
        namespace: &str,
        inbox_id: &str,
        run_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_for_run(
            self.transaction,
            namespace,
            inbox_id,
            run_id,
            claimed_by,
            limit,
            lease_seconds,
        )
    }
    pub fn update_projected_lane_message_payload(
        &mut self,
        message_id: &str,
        payload_json: String,
    ) -> SqliteStoreResult<()> {
        let changed = self.transaction.execute(
            "UPDATE projected_lane_messages SET payload_json=?1 WHERE message_id=?2",
            params![payload_json, message_id],
        )?;
        if changed != 1 {
            return Err(SqliteStoreError::RecordedRuntimeConflict(format!(
                "lane message {message_id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub fn clear_projected_lane_messages(&mut self, namespace: &str) -> SqliteStoreResult<u64> {
        clear_projected_lane_messages(self.transaction, namespace)
    }
    pub fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            self.transaction,
            namespace,
            inbox_id,
            claimed_by,
            limit,
            lease_seconds,
        )
    }
    pub fn claim_projected_lane_messages_filtered(
        &mut self,
        filter: LaneMessageClaimFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_filtered(self.transaction, &filter)
    }
    pub fn ack_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        ack_projected_lane_message(self.transaction, message_id, claimed_by)
    }
    pub fn requeue_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        requeue_projected_lane_message(
            self.transaction,
            message_id,
            claimed_by,
            error_json,
            delay_seconds,
        )
    }
    pub fn dead_letter_projected_lane_message(
        &mut self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        dead_letter_projected_lane_message(self.transaction, message_id, claimed_by, error_json)
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(self.transaction, namespace, inbox_id, limit)
    }
}
