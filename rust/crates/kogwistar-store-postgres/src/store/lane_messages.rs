use super::super::*;

impl PostgresStore {
    pub async fn project_lane_message(
        &self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        self.transaction(move |uow| Box::pin(async move { uow.project_lane_message(row).await }))
            .await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&**self.client().await?, &self.tables, id).await
    }
    pub async fn list_projected_lane_messages(
        &self,
        filter: LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&**self.client().await?, &self.tables, &filter).await
    }
    pub async fn update_projected_lane_message_status(
        &self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let status = status.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_status(&id, &status, error)
                    .await
            })
        })
        .await
    }
    pub async fn update_projected_lane_message_links(
        &self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.update_projected_lane_message_links(
                    &id,
                    prev,
                    next,
                    inbox_tail,
                    conversation_tail,
                )
                .await
            })
        })
        .await
    }
    pub async fn clear_projected_lane_messages(&self, namespace: &str) -> PostgresStoreResult<u64> {
        let namespace = namespace.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.clear_projected_lane_messages(&namespace).await })
        })
        .await
    }
    pub async fn claim_projected_lane_messages(
        &self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        let namespace = namespace.to_owned();
        let inbox = inbox.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.claim_projected_lane_messages(&namespace, &inbox, &owner, limit, lease)
                    .await
            })
        })
        .await
    }
    pub async fn claim_projected_lane_messages_filtered(
        &self,
        filter: LaneMessageClaimFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        self.transaction(move |uow| {
            Box::pin(async move { uow.claim_projected_lane_messages_filtered(filter).await })
        })
        .await
    }
    pub async fn ack_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move { uow.ack_projected_lane_message(&id, &owner).await })
        })
        .await
    }
    pub async fn requeue_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.requeue_projected_lane_message(&id, &owner, error, delay)
                    .await
            })
        })
        .await
    }
    pub async fn dead_letter_projected_lane_message(
        &self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        let id = id.to_owned();
        let owner = owner.to_owned();
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.dead_letter_projected_lane_message(&id, &owner, error)
                    .await
            })
        })
        .await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        let namespace = namespace.to_owned();
        let inbox_id = inbox_id.map(str::to_owned);
        self.transaction(move |uow| {
            Box::pin(async move {
                uow.repair_orphaned_claimed_lane_messages(&namespace, inbox_id.as_deref(), limit)
                    .await
            })
        })
        .await
    }
}
