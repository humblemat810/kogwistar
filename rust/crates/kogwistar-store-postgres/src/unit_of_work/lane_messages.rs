use super::*;

impl PostgresUnitOfWork<'_> {
    pub async fn project_lane_message(
        &mut self,
        row: NewProjectedLaneMessage,
    ) -> PostgresStoreResult<()> {
        project_lane_message(&self.transaction, &self.tables, row).await
    }
    pub async fn get_projected_lane_message(
        &self,
        id: &str,
    ) -> PostgresStoreResult<Option<ProjectedLaneMessage>> {
        projected_lane_message(&self.transaction, &self.tables, id).await
    }

    pub async fn list_projected_lane_messages(
        &self,
        filter: &LaneMessageFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        list_projected_lane_messages(&self.transaction, &self.tables, filter).await
    }
    pub async fn update_projected_lane_message_status(
        &mut self,
        id: &str,
        status: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_status(&self.transaction, &self.tables, id, status, error)
            .await
    }
    pub async fn update_projected_lane_message_payload(
        &mut self,
        id: &str,
        payload_json: String,
    ) -> PostgresStoreResult<()> {
        let changed = self
            .transaction
            .execute(
                &format!(
                    "UPDATE {} SET payload_json=$1 WHERE message_id=$2",
                    self.tables.projected_lane_messages
                ),
                &[&payload_json, &id],
            )
            .await
            .map_err(backend)?;
        if changed != 1 {
            return Err(PostgresStoreError::RecordedRuntimeConflict(format!(
                "lane message {id:?} payload update matched {changed} rows"
            )));
        }
        Ok(())
    }
    pub async fn update_projected_lane_message_links(
        &mut self,
        id: &str,
        prev: Option<String>,
        next: Option<String>,
        inbox_tail: Option<String>,
        conversation_tail: Option<String>,
    ) -> PostgresStoreResult<()> {
        update_projected_lane_message_links(
            &self.transaction,
            &self.tables,
            id,
            prev,
            next,
            inbox_tail,
            conversation_tail,
        )
        .await
    }
    pub async fn clear_projected_lane_messages(
        &mut self,
        namespace: &str,
    ) -> PostgresStoreResult<u64> {
        clear_projected_lane_messages(&self.transaction, &self.tables, namespace).await
    }
    pub async fn claim_projected_lane_messages(
        &mut self,
        namespace: &str,
        inbox: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn claim_projected_lane_messages_filtered(
        &mut self,
        filter: LaneMessageClaimFilter,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_filtered(&self.transaction, &self.tables, &filter).await
    }
    pub async fn claim_projected_lane_messages_for_run(
        &mut self,
        namespace: &str,
        inbox: &str,
        run_id: &str,
        owner: &str,
        limit: usize,
        lease: i64,
    ) -> PostgresStoreResult<Vec<ProjectedLaneMessage>> {
        claim_projected_lane_messages_for_run(
            &self.transaction,
            &self.tables,
            namespace,
            inbox,
            run_id,
            owner,
            limit,
            lease,
        )
        .await
    }
    pub async fn ack_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
    ) -> PostgresStoreResult<()> {
        ack_projected_lane_message(&self.transaction, &self.tables, id, owner).await
    }
    pub async fn requeue_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
        delay: i64,
    ) -> PostgresStoreResult<()> {
        requeue_projected_lane_message(&self.transaction, &self.tables, id, owner, error, delay)
            .await
    }
    pub async fn dead_letter_projected_lane_message(
        &mut self,
        id: &str,
        owner: &str,
        error: Option<String>,
    ) -> PostgresStoreResult<()> {
        dead_letter_projected_lane_message(&self.transaction, &self.tables, id, owner, error).await
    }
    pub async fn repair_orphaned_claimed_lane_messages(
        &mut self,
        namespace: &str,
        inbox_id: Option<&str>,
        limit: usize,
    ) -> PostgresStoreResult<Vec<String>> {
        repair_orphaned_claimed_lane_messages(
            &self.transaction,
            &self.tables,
            namespace,
            inbox_id,
            limit,
        )
        .await
    }
}
