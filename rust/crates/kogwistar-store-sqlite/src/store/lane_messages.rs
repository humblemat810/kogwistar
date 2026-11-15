use super::super::*;

impl SqliteStore {
    pub fn project_lane_message(&self, row: NewProjectedLaneMessage) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.project_lane_message(row))
    }
    pub fn get_projected_lane_message(
        &self,
        message_id: &str,
    ) -> SqliteStoreResult<Option<ProjectedLaneMessage>> {
        self.with_connection(|conn| projected_lane_message(conn, message_id))
    }
    pub fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.with_connection(|conn| list_projected_lane_messages(conn, &filter))
    }
    pub fn update_projected_lane_message_status(
        &self,
        message_id: &str,
        status: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_status(message_id, status, error_json)
        })
    }
    pub fn update_projected_lane_message_links(
        &self,
        message_id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.update_projected_lane_message_links(
                message_id,
                prev,
                next,
                inbox_tail,
                conversation_tail,
            )
        })
    }
    pub fn clear_projected_lane_messages(&self, namespace: &str) -> SqliteStoreResult<u64> {
        self.immediate_transaction(|uow| uow.clear_projected_lane_messages(namespace))
    }
    pub fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox_id: &str,
        claimed_by: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.immediate_transaction(|uow| {
            uow.claim_projected_lane_messages(namespace, inbox_id, claimed_by, limit, lease_seconds)
        })
    }
    pub fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> SqliteStoreResult<Vec<ProjectedLaneMessage>> {
        self.immediate_transaction(|uow| uow.claim_projected_lane_messages_filtered(filter))
    }
    pub fn ack_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| uow.ack_projected_lane_message(message_id, claimed_by))
    }
    pub fn requeue_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
        delay_seconds: i64,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.requeue_projected_lane_message(message_id, claimed_by, error_json, delay_seconds)
        })
    }
    pub fn dead_letter_projected_lane_message(
        &self,
        message_id: &str,
        claimed_by: &str,
        error_json: Option<String>,
    ) -> SqliteStoreResult<()> {
        self.immediate_transaction(|uow| {
            uow.dead_letter_projected_lane_message(message_id, claimed_by, error_json)
        })
    }
    pub fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> SqliteStoreResult<Vec<String>> {
        self.immediate_transaction(|uow| {
            uow.repair_orphaned_claimed_lane_messages(namespace, inbox_id, limit)
        })
    }
}
