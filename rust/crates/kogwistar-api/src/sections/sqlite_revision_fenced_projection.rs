// SQLite remains canonical; this adapter fences eventual Chroma projection writes by revision.
// Python parity: kogwistar/engine_core/two_stage_async.py (durable projection/reconciliation boundary).
#[derive(Clone)]
struct RevisionFencedChromaProjectionStore {
    queue: SqliteStore,
    graph: kogwistar_store_chroma::ChromaGraphStore,
}

impl RevisionFencedChromaProjectionStore {
    fn current_revision(
        &self,
        namespace: &str,
        entity_kind: &str,
        entity_id: &str,
    ) -> Result<Option<kogwistar_engine::canonical_revision::CanonicalEntityRevision>, StoreError>
    {
        sqlite_canonical_entity_revision(&self.queue, namespace, entity_kind, entity_id)
    }

    fn enqueue_current_revision(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        let payload_json =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(revision);
        self.queue
            .enqueue_index_job(NewIndexJob {
                job_id: format!("revision-fence-{}", uuid::Uuid::new_v4()),
                namespace: revision.namespace.clone(),
                entity_kind: revision.entity_kind.clone(),
                entity_id: revision.entity_id.clone(),
                index_kind: "node_embedding".to_owned(),
                op: if revision.is_deleted() {
                    "DELETE".to_owned()
                } else {
                    "UPSERT".to_owned()
                },
                payload_json: Some(payload_json),
                max_retries: 10,
            })
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        Ok(())
    }

    fn clear_stage1_if_current(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        self.queue
            .clear_stage1_node_projection_if_current(
                &revision.namespace,
                &format!("{}:{}", revision.entity_kind, revision.entity_id),
                revision.revision,
            )
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        Ok(())
    }

    fn clear_stage1_through_deleted_revision(
        &self,
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<(), StoreError> {
        let key = format!("{}:{}", revision.entity_kind, revision.entity_id);
        let stage1 = self
            .queue
            .get_stage1_node_projection(&revision.namespace, &key)
            .map_err(|error| StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: error.to_string(),
            })?;
        if let Some(stage1) = stage1
            && stage1.last_authoritative_seq <= revision.revision
        {
            self.queue
                .clear_stage1_node_projection_if_current(
                    &revision.namespace,
                    &key,
                    stage1.last_authoritative_seq,
                )
                .map_err(|error| StoreError::Backend {
                    backend: "sqlite-chroma-projection".to_owned(),
                    message: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn canonical_fingerprint(
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
    ) -> Result<String, StoreError> {
        let payload =
            kogwistar_engine::canonical_revision::canonical_revision_job_payload(revision);
        serde_json::from_str::<Value>(&payload)
            .ok()
            .and_then(|value| {
                value
                    .get("source_fingerprint")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .ok_or_else(|| StoreError::InvalidEntityEventPayload {
                message: "canonical revision fingerprint is unavailable".to_owned(),
            })
    }

    fn restore_stage1_from_revision(
        revision: &kogwistar_engine::canonical_revision::CanonicalEntityRevision,
        fingerprint: &str,
        class_name: Option<&str>,
    ) -> Result<NamedProjectionWrite, StoreError> {
        kogwistar_store_sqlite::stage1_projection_from_canonical_event_with_class_name(
            &revision.namespace,
            &revision.entity_kind,
            &revision.entity_id,
            revision.revision,
            &revision.payload,
            fingerprint,
            class_name,
        )
    }

    fn graph_kind(entity_kind: &str) -> Result<kogwistar_store::GraphEntityKind, StoreError> {
        match entity_kind {
            "node" => Ok(kogwistar_store::GraphEntityKind::Node),
            "edge" => Ok(kogwistar_store::GraphEntityKind::Edge),
            _ => Err(StoreError::Backend {
                backend: "sqlite-chroma-projection".to_owned(),
                message: "projection entity kind must be 'node' or 'edge'".to_owned(),
            }),
        }
    }

    async fn upsert_with_revision_fence(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> Result<kogwistar_store::ProjectionWriteOutcome, StoreError> {
        use kogwistar_store::ProjectionWriteOutcome;

        let kind = Self::graph_kind(entity_kind)?;
        let Some(current) = self.current_revision(namespace, entity_kind, &record.id)? else {
            kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
                &self.graph,
                namespace,
                kind,
                &record.id,
            )
            .await?;
            return Ok(ProjectionWriteOutcome::Applied);
        };
        if current.is_deleted() {
            self.enqueue_current_revision(&current)?;
            return Ok(ProjectionWriteOutcome::Superseded);
        }
        let current_fingerprint = Self::canonical_fingerprint(&current)?;
        let record_revision = record
            .metadata
            .get("_kogwistar_source_revision")
            .and_then(Value::as_i64);
        let record_fingerprint = record
            .metadata
            .get("_kogwistar_source_fingerprint")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let stage2_ready = record
            .metadata
            .get("_kogwistar_stage2_ready")
            .and_then(Value::as_bool)
            == Some(true);
        if !stage2_ready
            || record_revision != Some(current.revision)
            || record_fingerprint.as_deref() != Some(current_fingerprint.as_str())
        {
            self.enqueue_current_revision(&current)?;
            return Ok(ProjectionWriteOutcome::Superseded);
        }
        let projected_revision = record_revision.expect("stage2-ready revision was validated");
        let projected_fingerprint =
            record_fingerprint.expect("stage2-ready fingerprint was validated");
        let projected_id = record.id.clone();
        kogwistar_store::GraphWriteStore::upsert_graph_record_of_kind(
            &self.graph,
            namespace,
            kind,
            record,
        )
        .await?;
        match self.current_revision(namespace, entity_kind, &projected_id)? {
            Some(latest)
                if !latest.is_deleted()
                    && latest.revision == projected_revision
                    && Self::canonical_fingerprint(&latest)? == projected_fingerprint =>
            {
                self.clear_stage1_if_current(&latest)?;
                Ok(ProjectionWriteOutcome::Applied)
            }
            Some(latest) => {
                self.enqueue_current_revision(&latest)?;
                Ok(ProjectionWriteOutcome::Superseded)
            }
            None => {
                kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
                    &self.graph,
                    namespace,
                    kind,
                    &projected_id,
                )
                .await?;
                Ok(ProjectionWriteOutcome::Superseded)
            }
        }
    }

    async fn delete_with_revision_fence(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> Result<(bool, kogwistar_store::ProjectionWriteOutcome), StoreError> {
        use kogwistar_store::ProjectionWriteOutcome;

        let kind = Self::graph_kind(entity_kind)?;
        if let Some(current) = self.current_revision(namespace, entity_kind, id)?
            && !current.is_deleted()
        {
            self.enqueue_current_revision(&current)?;
            return Ok((false, ProjectionWriteOutcome::Superseded));
        }
        let deleted = kogwistar_store::GraphWriteStore::delete_graph_record_of_kind(
            &self.graph,
            namespace,
            kind,
            id,
        )
        .await?;
        if let Some(current) = self.current_revision(namespace, entity_kind, id)? {
            if !current.is_deleted() {
                self.enqueue_current_revision(&current)?;
                return Ok((deleted, ProjectionWriteOutcome::Superseded));
            }
            self.clear_stage1_through_deleted_revision(&current)?;
        }
        Ok((deleted, ProjectionWriteOutcome::Applied))
    }
}

impl kogwistar_store::GraphProjectionWriteStore for RevisionFencedChromaProjectionStore {
    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl std::future::Future<Output = Result<(), StoreError>> + Send {
        async move {
            self.upsert_with_revision_fence(namespace, entity_kind, record)
                .await
                .map(|_| ())
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl std::future::Future<Output = Result<bool, StoreError>> + Send {
        async move {
            self.delete_with_revision_fence(namespace, entity_kind, id)
                .await
                .map(|(deleted, _)| deleted)
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn upsert_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        record: GraphRecord,
    ) -> impl std::future::Future<
        Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>,
    > + Send {
        async move {
            self.upsert_with_revision_fence(namespace, entity_kind, record)
                .await
        }
    }

    // Preserve the trait's explicit opaque-future + Send contract at this adapter.
    #[allow(clippy::manual_async_fn)]
    fn delete_graph_projection_record_with_outcome(
        &self,
        namespace: &str,
        entity_kind: &str,
        id: &str,
    ) -> impl std::future::Future<
        Output = Result<kogwistar_store::ProjectionWriteOutcome, StoreError>,
    > + Send {
        async move {
            self.delete_with_revision_fence(namespace, entity_kind, id)
                .await
                .map(|(_, outcome)| outcome)
        }
    }
}

