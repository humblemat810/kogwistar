use super::super::*;

impl LaneMessageReadStore for SqliteStore {
    async fn projected_lane_message(
        &self,
        message_id: &str,
    ) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(message_id)
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for SqliteStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        SqliteStore::project_lane_message(self, row).map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_status(self, id, status, error)
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        SqliteStore::clear_projected_lane_messages(self, namespace).map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        SqliteStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        SqliteStore::claim_projected_lane_messages_filtered(self, filter).map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        SqliteStore::ack_projected_lane_message(self, id, owner).map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        SqliteStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        SqliteStore::dead_letter_projected_lane_message(self, id, owner, error).map_err(trait_error)
    }
}
