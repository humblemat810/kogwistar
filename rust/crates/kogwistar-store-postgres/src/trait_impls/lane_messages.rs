use super::super::*;

impl LaneMessageReadStore for PostgresStore {
    async fn projected_lane_message(&self, id: &str) -> StoreResult<Option<ProjectedLaneMessage>> {
        self.get_projected_lane_message(id)
            .await
            .map_err(trait_error)
    }
    async fn projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        self.list_projected_lane_messages(filter)
            .await
            .map_err(trait_error)
    }
}
impl LaneMessageWriteStore for PostgresStore {
    async fn project_lane_message(&self, row: NewProjectedLaneMessage) -> StoreResult<()> {
        PostgresStore::project_lane_message(self, row)
            .await
            .map_err(trait_error)
    }
    async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::update_projected_lane_message_status(self, id, status, error)
            .await
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
        PostgresStore::update_projected_lane_message_links(
            self,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
        .map_err(trait_error)
    }
    async fn clear_projected_lane_messages(&self, namespace: &str) -> StoreResult<u64> {
        PostgresStore::clear_projected_lane_messages(self, namespace)
            .await
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        PostgresStore::claim_projected_lane_messages(self, namespace, inbox, owner, limit, lease)
            .await
            .map_err(trait_error)
    }
    async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> StoreResult<Vec<ProjectedLaneMessage>> {
        PostgresStore::claim_projected_lane_messages_filtered(self, filter)
            .await
            .map_err(trait_error)
    }
    async fn ack_projected_lane_message(&self, id: &str, owner: &str) -> StoreResult<()> {
        PostgresStore::ack_projected_lane_message(self, id, owner)
            .await
            .map_err(trait_error)
    }
    async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> StoreResult<()> {
        PostgresStore::requeue_projected_lane_message(self, id, owner, error, delay)
            .await
            .map_err(trait_error)
    }
    async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> StoreResult<()> {
        PostgresStore::dead_letter_projected_lane_message(self, id, owner, error)
            .await
            .map_err(trait_error)
    }
}
